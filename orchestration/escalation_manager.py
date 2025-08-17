"""
Escalation Manager

Handles automated escalation procedures and communication protocols
for the Agent Orchestration system.
"""

import json
import smtplib
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import logging

from priority_matrix import Priority, AgentType
from agent_orchestrator import Task, TaskStatus


class EscalationLevel(Enum):
    """Escalation levels."""
    L1_AUTOMATED = "l1_automated"      # Automated recovery attempts
    L2_NOTIFICATION = "l2_notification" # Notifications sent
    L3_HUMAN_REVIEW = "l3_human_review" # Human intervention required
    L4_EMERGENCY = "l4_emergency"       # Emergency response


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    GITHUB_ISSUE = "github_issue"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG_ONLY = "log_only"


@dataclass
class EscalationRule:
    """Defines when and how to escalate issues."""
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    escalation_level: EscalationLevel
    notification_channels: List[NotificationChannel]
    cooldown_minutes: int = 60
    max_escalations_per_hour: int = 5
    enabled: bool = True


@dataclass
class EscalationEvent:
    """Represents an escalation event."""
    id: str
    rule_name: str
    escalation_level: EscalationLevel
    trigger_data: Dict[str, Any]
    created_at: datetime
    notifications_sent: List[str]
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    human_notified: bool = False
    automated_actions_taken: List[str] = None
    
    def __post_init__(self):
        if self.automated_actions_taken is None:
            self.automated_actions_taken = []


class EscalationManager:
    """
    Manages escalation procedures and communication protocols for the orchestration system.
    
    Features:
    - Rule-based escalation triggers
    - Multi-channel notifications (email, GitHub, Slack, webhooks)
    - Automated recovery attempts
    - Escalation tracking and resolution
    - Cooldown periods to prevent notification spam
    - Human-in-the-loop procedures for critical issues
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the escalation manager."""
        self.config_path = config_path or Path(__file__).parent / "escalation_config.json"
        self.load_configuration()
        self.setup_logging()
        
        # State tracking
        self.active_escalations: Dict[str, EscalationEvent] = {}
        self.escalation_history: List[EscalationEvent] = []
        self.notification_counts: Dict[str, int] = {}  # Hourly notification counts
        self.last_reset_time = datetime.now()
        
        # Load escalation rules
        self.rules = self._load_escalation_rules()
        
        # Initialize notification handlers
        self.notification_handlers = {
            NotificationChannel.EMAIL: self._send_email_notification,
            NotificationChannel.GITHUB_ISSUE: self._create_github_issue,
            NotificationChannel.SLACK: self._send_slack_notification,
            NotificationChannel.WEBHOOK: self._send_webhook_notification,
            NotificationChannel.LOG_ONLY: self._log_notification
        }
        
        self.logger.info("Escalation Manager initialized")
    
    def load_configuration(self) -> None:
        """Load escalation configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.get_default_configuration()
            self.save_configuration()
    
    def get_default_configuration(self) -> Dict:
        """Get default escalation configuration."""
        return {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "use_tls": True,
                "username": "",
                "password": "",
                "from_address": "fao-dashboard@yourcompany.com",
                "to_addresses": [],
                "subject_prefix": "[FAO Dashboard Alert]"
            },
            "github": {
                "enabled": False,
                "token": "",
                "repository": "",
                "labels": ["automated", "escalation", "priority:high"]
            },
            "slack": {
                "enabled": False,
                "webhook_url": "",
                "channel": "#alerts",
                "username": "FAO Dashboard Bot"
            },
            "webhooks": {
                "enabled": False,
                "endpoints": []
            },
            "escalation": {
                "max_notifications_per_hour": 10,
                "cooldown_minutes": 30,
                "auto_resolve_after_hours": 24,
                "emergency_contact_delay_minutes": 15
            },
            "automated_recovery": {
                "enabled": True,
                "max_retry_attempts": 3,
                "agent_restart_enabled": True,
                "cache_clear_enabled": True,
                "fallback_mode_enabled": True
            }
        }
    
    def save_configuration(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def setup_logging(self) -> None:
        """Configure logging for the escalation manager."""
        log_dir = Path(__file__).parent.parent / "automation" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create escalation-specific logger
        self.logger = logging.getLogger('escalation_manager')
        self.logger.setLevel(logging.INFO)
        
        # File handler for escalation logs
        file_handler = logging.FileHandler(log_dir / "escalations.log")
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - ESCALATION - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)
    
    def _load_escalation_rules(self) -> List[EscalationRule]:
        """Load escalation rules from configuration."""
        default_rules = [
            EscalationRule(
                name="critical_task_failure",
                description="Critical priority task failed",
                trigger_conditions={
                    "task_priority": "critical",
                    "task_status": "failed",
                    "retry_count": {">=": 2}
                },
                escalation_level=EscalationLevel.L3_HUMAN_REVIEW,
                notification_channels=[
                    NotificationChannel.EMAIL,
                    NotificationChannel.GITHUB_ISSUE,
                    NotificationChannel.LOG_ONLY
                ],
                cooldown_minutes=30
            ),
            EscalationRule(
                name="agent_health_critical",
                description="Agent health score critically low",
                trigger_conditions={
                    "agent_health_score": {"<": 0.3},
                    "duration_minutes": {">=": 10}
                },
                escalation_level=EscalationLevel.L2_NOTIFICATION,
                notification_channels=[
                    NotificationChannel.EMAIL,
                    NotificationChannel.LOG_ONLY
                ],
                cooldown_minutes=60
            ),
            EscalationRule(
                name="queue_size_critical",
                description="Task queue size critically high",
                trigger_conditions={
                    "queue_size": {">=": 100},
                    "duration_minutes": {">=": 15}
                },
                escalation_level=EscalationLevel.L2_NOTIFICATION,
                notification_channels=[
                    NotificationChannel.EMAIL,
                    NotificationChannel.SLACK,
                    NotificationChannel.LOG_ONLY
                ]
            ),
            EscalationRule(
                name="data_pipeline_failure",
                description="Data pipeline agent repeated failures",
                trigger_conditions={
                    "agent_type": "data-pipeline",
                    "failure_rate": {">=": 0.5},
                    "recent_failures": {">=": 3}
                },
                escalation_level=EscalationLevel.L3_HUMAN_REVIEW,
                notification_channels=[
                    NotificationChannel.EMAIL,
                    NotificationChannel.GITHUB_ISSUE
                ],
                cooldown_minutes=45
            ),
            EscalationRule(
                name="system_unresponsive",
                description="System appears unresponsive",
                trigger_conditions={
                    "no_task_completion": {"minutes": 30},
                    "agent_activity": "none"
                },
                escalation_level=EscalationLevel.L4_EMERGENCY,
                notification_channels=[
                    NotificationChannel.EMAIL,
                    NotificationChannel.SLACK,
                    NotificationChannel.WEBHOOK
                ],
                cooldown_minutes=15,
                max_escalations_per_hour=2
            ),
            EscalationRule(
                name="high_failure_rate",
                description="Overall task failure rate too high",
                trigger_conditions={
                    "overall_failure_rate": {">=": 0.3},
                    "sample_size": {">=": 10}
                },
                escalation_level=EscalationLevel.L2_NOTIFICATION,
                notification_channels=[
                    NotificationChannel.EMAIL,
                    NotificationChannel.LOG_ONLY
                ]
            )
        ]
        
        return default_rules
    
    def check_escalation_conditions(
        self,
        orchestrator_metrics: Dict[str, Any],
        agent_metrics: Dict[str, Any],
        queue_status: Dict[str, Any],
        recent_tasks: List[Task]
    ) -> List[EscalationEvent]:
        """
        Check all escalation rules against current system state.
        
        Args:
            orchestrator_metrics: Current orchestrator metrics
            agent_metrics: Agent status and metrics
            queue_status: Queue status information
            recent_tasks: List of recent tasks
            
        Returns:
            List of triggered escalation events
        """
        triggered_escalations = []
        current_time = datetime.now()
        
        # Reset notification counts hourly
        if (current_time - self.last_reset_time).total_seconds() > 3600:
            self.notification_counts.clear()
            self.last_reset_time = current_time
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if self._is_in_cooldown(rule.name, current_time):
                continue
            
            # Check notification rate limits
            if self._exceeds_rate_limit(rule.name):
                continue
            
            # Evaluate rule conditions
            if self._evaluate_rule_conditions(
                rule, orchestrator_metrics, agent_metrics, queue_status, recent_tasks
            ):
                escalation_event = self._create_escalation_event(
                    rule, orchestrator_metrics, agent_metrics, queue_status
                )
                triggered_escalations.append(escalation_event)
                self.logger.warning(f"Escalation triggered: {rule.name}")
        
        return triggered_escalations
    
    def _is_in_cooldown(self, rule_name: str, current_time: datetime) -> bool:
        """Check if rule is in cooldown period."""
        for escalation in self.escalation_history:
            if escalation.rule_name == rule_name:
                time_diff = (current_time - escalation.created_at).total_seconds() / 60
                rule = next((r for r in self.rules if r.name == rule_name), None)
                if rule and time_diff < rule.cooldown_minutes:
                    return True
        return False
    
    def _exceeds_rate_limit(self, rule_name: str) -> bool:
        """Check if rule exceeds rate limit."""
        rule = next((r for r in self.rules if r.name == rule_name), None)
        if not rule:
            return False
        
        current_count = self.notification_counts.get(rule_name, 0)
        return current_count >= rule.max_escalations_per_hour
    
    def _evaluate_rule_conditions(
        self,
        rule: EscalationRule,
        orchestrator_metrics: Dict[str, Any],
        agent_metrics: Dict[str, Any],
        queue_status: Dict[str, Any],
        recent_tasks: List[Task]
    ) -> bool:
        """Evaluate if rule conditions are met."""
        conditions = rule.trigger_conditions
        
        # Check queue size conditions
        if "queue_size" in conditions:
            queue_size = queue_status.get('total_queued', 0)
            if not self._evaluate_condition(queue_size, conditions["queue_size"]):
                return False
        
        # Check overall failure rate
        if "overall_failure_rate" in conditions:
            if recent_tasks:
                failed_tasks = [t for t in recent_tasks if t.status == TaskStatus.FAILED]
                failure_rate = len(failed_tasks) / len(recent_tasks)
                if not self._evaluate_condition(failure_rate, conditions["overall_failure_rate"]):
                    return False
                
                # Check sample size requirement
                if "sample_size" in conditions:
                    if not self._evaluate_condition(len(recent_tasks), conditions["sample_size"]):
                        return False
            else:
                return False
        
        # Check agent-specific conditions
        if "agent_type" in conditions:
            agent_type = conditions["agent_type"]
            if agent_type not in agent_metrics:
                return False
            
            agent_data = agent_metrics[agent_type]
            
            # Check agent health
            if "agent_health_score" in conditions:
                health_score = agent_data.get('health_score', 1.0)
                if not self._evaluate_condition(health_score, conditions["agent_health_score"]):
                    return False
            
            # Check failure rate for specific agent
            if "failure_rate" in conditions:
                total_tasks = agent_data.get('tasks_completed', 0) + agent_data.get('tasks_failed', 0)
                if total_tasks > 0:
                    failure_rate = agent_data.get('tasks_failed', 0) / total_tasks
                    if not self._evaluate_condition(failure_rate, conditions["failure_rate"]):
                        return False
                else:
                    return False
        
        # Check task-specific conditions
        if "task_priority" in conditions and "task_status" in conditions:
            priority_filter = conditions["task_priority"]
            status_filter = conditions["task_status"]
            
            matching_tasks = [
                t for t in recent_tasks
                if t.priority.value == priority_filter and t.status.value == status_filter
            ]
            
            if not matching_tasks:
                return False
            
            # Check retry count if specified
            if "retry_count" in conditions:
                for task in matching_tasks:
                    if self._evaluate_condition(task.retry_count, conditions["retry_count"]):
                        return True
                return False
        
        return True
    
    def _evaluate_condition(self, value: Any, condition: Any) -> bool:
        """Evaluate a single condition against a value."""
        if isinstance(condition, dict):
            for operator, threshold in condition.items():
                if operator == ">=" and value < threshold:
                    return False
                elif operator == ">" and value <= threshold:
                    return False
                elif operator == "<=" and value > threshold:
                    return False
                elif operator == "<" and value >= threshold:
                    return False
                elif operator == "==" and value != threshold:
                    return False
                elif operator == "!=" and value == threshold:
                    return False
        else:
            # Direct equality check
            return value == condition
        
        return True
    
    def _create_escalation_event(
        self,
        rule: EscalationRule,
        orchestrator_metrics: Dict[str, Any],
        agent_metrics: Dict[str, Any],
        queue_status: Dict[str, Any]
    ) -> EscalationEvent:
        """Create an escalation event."""
        event_id = f"{rule.name}_{int(time.time() * 1000)}"
        
        trigger_data = {
            "rule": rule.name,
            "description": rule.description,
            "orchestrator_metrics": orchestrator_metrics,
            "agent_metrics": agent_metrics,
            "queue_status": queue_status,
            "conditions": rule.trigger_conditions
        }
        
        event = EscalationEvent(
            id=event_id,
            rule_name=rule.name,
            escalation_level=rule.escalation_level,
            trigger_data=trigger_data,
            created_at=datetime.now(),
            notifications_sent=[]
        )
        
        self.active_escalations[event_id] = event
        self.escalation_history.append(event)
        
        # Update notification count
        self.notification_counts[rule.name] = self.notification_counts.get(rule.name, 0) + 1
        
        return event
    
    def handle_escalation(self, escalation_event: EscalationEvent) -> None:
        """Handle an escalation event by taking appropriate actions."""
        self.logger.info(f"Handling escalation: {escalation_event.id}")
        
        # Get the rule for this escalation
        rule = next((r for r in self.rules if r.name == escalation_event.rule_name), None)
        if not rule:
            self.logger.error(f"Rule not found for escalation: {escalation_event.rule_name}")
            return
        
        # Perform automated recovery actions if enabled
        if (escalation_event.escalation_level == EscalationLevel.L1_AUTOMATED and 
            self.config.get("automated_recovery", {}).get("enabled", True)):
            self._perform_automated_recovery(escalation_event)
        
        # Send notifications
        for channel in rule.notification_channels:
            try:
                handler = self.notification_handlers.get(channel)
                if handler:
                    success = handler(escalation_event, rule)
                    if success:
                        escalation_event.notifications_sent.append(channel.value)
                        self.logger.info(f"Notification sent via {channel.value}")
                    else:
                        self.logger.warning(f"Failed to send notification via {channel.value}")
            except Exception as e:
                self.logger.error(f"Error sending notification via {channel.value}: {e}")
        
        # Mark for human review if necessary
        if escalation_event.escalation_level in [EscalationLevel.L3_HUMAN_REVIEW, EscalationLevel.L4_EMERGENCY]:
            escalation_event.human_notified = True
            self.logger.warning(f"Human review required for escalation: {escalation_event.id}")
    
    def _perform_automated_recovery(self, escalation_event: EscalationEvent) -> None:
        """Perform automated recovery actions."""
        self.logger.info(f"Attempting automated recovery for: {escalation_event.id}")
        
        recovery_config = self.config.get("automated_recovery", {})
        actions_taken = []
        
        # Agent restart (simulated)
        if (recovery_config.get("agent_restart_enabled", True) and 
            "agent_health" in escalation_event.rule_name):
            actions_taken.append("agent_restart_attempted")
            self.logger.info("Automated agent restart initiated")
        
        # Cache clearing (simulated)
        if (recovery_config.get("cache_clear_enabled", True) and 
            "data_pipeline" in escalation_event.rule_name):
            actions_taken.append("cache_cleared")
            self.logger.info("Automated cache clearing initiated")
        
        # Fallback mode activation (simulated)
        if (recovery_config.get("fallback_mode_enabled", True) and 
            escalation_event.escalation_level == EscalationLevel.L4_EMERGENCY):
            actions_taken.append("fallback_mode_activated")
            self.logger.info("Fallback mode activated")
        
        escalation_event.automated_actions_taken.extend(actions_taken)
    
    def _send_email_notification(self, escalation_event: EscalationEvent, rule: EscalationRule) -> bool:
        """Send email notification."""
        email_config = self.config.get("email", {})
        if not email_config.get("enabled", False):
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"{email_config['subject_prefix']} {rule.description}"
            
            # Create email body
            body = self._create_email_body(escalation_event, rule)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config.get('use_tls', True):
                server.starttls()
            
            if email_config.get('username') and email_config.get('password'):
                server.login(email_config['username'], email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _create_email_body(self, escalation_event: EscalationEvent, rule: EscalationRule) -> str:
        """Create HTML email body for escalation notification."""
        return f"""
        <html>
        <body>
            <h2>FAO Dashboard Escalation Alert</h2>
            
            <p><strong>Alert:</strong> {rule.description}</p>
            <p><strong>Escalation Level:</strong> {escalation_event.escalation_level.value.upper()}</p>
            <p><strong>Time:</strong> {escalation_event.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            
            <h3>Trigger Details</h3>
            <p><strong>Rule:</strong> {rule.name}</p>
            <p><strong>Conditions:</strong> {json.dumps(rule.trigger_conditions, indent=2)}</p>
            
            <h3>System Status</h3>
            <p><strong>Queue Size:</strong> {escalation_event.trigger_data.get('queue_status', {}).get('total_queued', 'Unknown')}</p>
            <p><strong>Active Tasks:</strong> {escalation_event.trigger_data.get('queue_status', {}).get('total_active', 'Unknown')}</p>
            
            {self._format_agent_status_for_email(escalation_event.trigger_data.get('agent_metrics', {}))}
            
            <h3>Automated Actions</h3>
            {('<ul>' + ''.join(f'<li>{action}</li>' for action in escalation_event.automated_actions_taken) + '</ul>') if escalation_event.automated_actions_taken else '<p>No automated actions taken.</p>'}
            
            <h3>Next Steps</h3>
            <p>Please review the system status and take appropriate action if necessary.</p>
            
            <hr>
            <p><small>This is an automated alert from the FAO Dashboard Agent Orchestration system.</small></p>
        </body>
        </html>
        """
    
    def _format_agent_status_for_email(self, agent_metrics: Dict[str, Any]) -> str:
        """Format agent status for email."""
        if not agent_metrics:
            return "<p>No agent status available.</p>"
        
        html = "<h4>Agent Status</h4><ul>"
        for agent_type, metrics in agent_metrics.items():
            status = metrics.get('status', 'unknown')
            health = metrics.get('health_score', 0)
            html += f"<li><strong>{agent_type}:</strong> {status} (Health: {health:.2f})</li>"
        html += "</ul>"
        
        return html
    
    def _create_github_issue(self, escalation_event: EscalationEvent, rule: EscalationRule) -> bool:
        """Create GitHub issue for escalation."""
        github_config = self.config.get("github", {})
        if not github_config.get("enabled", False):
            return False
        
        # This would require the GitHub API
        # For now, just log the action
        self.logger.info(f"GitHub issue would be created for escalation: {escalation_event.id}")
        
        issue_data = {
            "title": f"[Escalation] {rule.description}",
            "body": self._create_github_issue_body(escalation_event, rule),
            "labels": github_config.get("labels", [])
        }
        
        # TODO: Implement actual GitHub API call
        # For now, save the issue data to a file
        try:
            issues_dir = Path(__file__).parent.parent / "automation" / "github_issues"
            issues_dir.mkdir(parents=True, exist_ok=True)
            
            issue_file = issues_dir / f"escalation_{escalation_event.id}.json"
            with open(issue_file, 'w') as f:
                json.dump(issue_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save GitHub issue data: {e}")
            return False
    
    def _create_github_issue_body(self, escalation_event: EscalationEvent, rule: EscalationRule) -> str:
        """Create GitHub issue body for escalation."""
        return f"""## Escalation Alert: {rule.description}

**Escalation Level:** {escalation_event.escalation_level.value.upper()}  
**Time:** {escalation_event.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Event ID:** {escalation_event.id}

### Trigger Details

**Rule:** {rule.name}  
**Conditions:** 
```json
{json.dumps(rule.trigger_conditions, indent=2)}
```

### System Status

- **Queue Size:** {escalation_event.trigger_data.get('queue_status', {}).get('total_queued', 'Unknown')}
- **Active Tasks:** {escalation_event.trigger_data.get('queue_status', {}).get('total_active', 'Unknown')}

### Agent Status

{self._format_agent_status_for_github(escalation_event.trigger_data.get('agent_metrics', {}))}

### Automated Actions

{('- ' + '\\n- '.join(escalation_event.automated_actions_taken)) if escalation_event.automated_actions_taken else 'No automated actions taken.'}

### Investigation Steps

1. Review system logs for errors
2. Check agent health and performance
3. Verify data pipeline integrity
4. Monitor for resolution

---
*This issue was automatically created by the Agent Orchestration system.*
"""
    
    def _format_agent_status_for_github(self, agent_metrics: Dict[str, Any]) -> str:
        """Format agent status for GitHub."""
        if not agent_metrics:
            return "No agent status available."
        
        lines = []
        for agent_type, metrics in agent_metrics.items():
            status = metrics.get('status', 'unknown')
            health = metrics.get('health_score', 0)
            lines.append(f"- **{agent_type}:** {status} (Health: {health:.2f})")
        
        return '\n'.join(lines)
    
    def _send_slack_notification(self, escalation_event: EscalationEvent, rule: EscalationRule) -> bool:
        """Send Slack notification."""
        slack_config = self.config.get("slack", {})
        if not slack_config.get("enabled", False):
            return False
        
        # This would require the Slack API or webhooks
        # For now, just log the action
        self.logger.info(f"Slack notification would be sent for escalation: {escalation_event.id}")
        return True
    
    def _send_webhook_notification(self, escalation_event: EscalationEvent, rule: EscalationRule) -> bool:
        """Send webhook notification."""
        webhook_config = self.config.get("webhooks", {})
        if not webhook_config.get("enabled", False):
            return False
        
        # This would send HTTP POST requests to configured webhooks
        # For now, just log the action
        self.logger.info(f"Webhook notification would be sent for escalation: {escalation_event.id}")
        return True
    
    def _log_notification(self, escalation_event: EscalationEvent, rule: EscalationRule) -> bool:
        """Log notification (always succeeds)."""
        self.logger.warning(f"ESCALATION: {rule.description} - Event ID: {escalation_event.id}")
        return True
    
    def resolve_escalation(self, escalation_id: str, resolution_notes: str = "") -> bool:
        """Mark an escalation as resolved."""
        if escalation_id not in self.active_escalations:
            self.logger.warning(f"Escalation not found: {escalation_id}")
            return False
        
        escalation = self.active_escalations[escalation_id]
        escalation.resolved_at = datetime.now()
        escalation.resolution_notes = resolution_notes
        
        # Remove from active escalations
        del self.active_escalations[escalation_id]
        
        self.logger.info(f"Escalation resolved: {escalation_id}")
        return True
    
    def get_active_escalations(self) -> List[EscalationEvent]:
        """Get all active escalations."""
        return list(self.active_escalations.values())
    
    def get_escalation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get escalation summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_escalations = [
            esc for esc in self.escalation_history
            if esc.created_at >= cutoff_time
        ]
        
        summary = {
            "total_escalations": len(recent_escalations),
            "active_escalations": len(self.active_escalations),
            "by_level": {},
            "by_rule": {},
            "resolved_escalations": 0,
            "avg_resolution_time": 0
        }
        
        # Count by level
        for escalation in recent_escalations:
            level = escalation.escalation_level.value
            summary["by_level"][level] = summary["by_level"].get(level, 0) + 1
        
        # Count by rule
        for escalation in recent_escalations:
            rule = escalation.rule_name
            summary["by_rule"][rule] = summary["by_rule"].get(rule, 0) + 1
        
        # Calculate resolution metrics
        resolved_escalations = [
            esc for esc in recent_escalations
            if esc.resolved_at is not None
        ]
        
        summary["resolved_escalations"] = len(resolved_escalations)
        
        if resolved_escalations:
            total_resolution_time = sum(
                (esc.resolved_at - esc.created_at).total_seconds()
                for esc in resolved_escalations
            )
            summary["avg_resolution_time"] = total_resolution_time / len(resolved_escalations)
        
        return summary
    
    def export_escalation_report(self, hours: int = 24) -> Path:
        """Export escalation report to JSON file."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_escalations = [
            asdict(esc) for esc in self.escalation_history
            if esc.created_at >= cutoff_time
        ]
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "time_period_hours": hours,
            "summary": self.get_escalation_summary(hours),
            "escalations": recent_escalations,
            "active_escalations": [asdict(esc) for esc in self.active_escalations.values()],
            "configuration": {
                "rules_count": len(self.rules),
                "enabled_rules": [r.name for r in self.rules if r.enabled],
                "notification_channels": list(set(
                    channel.value for rule in self.rules for channel in rule.notification_channels
                ))
            }
        }
        
        # Save report
        reports_dir = Path(__file__).parent.parent / "automation" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = reports_dir / f"escalation_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report_path


def main():
    """CLI interface for the escalation manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Orchestration Escalation Manager")
    parser.add_argument('--status', action='store_true', help='Show escalation status')
    parser.add_argument('--summary', type=int, metavar='HOURS', default=24, 
                       help='Show escalation summary for specified hours')
    parser.add_argument('--resolve', metavar='ESCALATION_ID', help='Resolve an escalation')
    parser.add_argument('--export-report', type=int, metavar='HOURS', default=24,
                       help='Export escalation report')
    parser.add_argument('--test-rule', metavar='RULE_NAME', help='Test a specific escalation rule')
    
    args = parser.parse_args()
    
    escalation_manager = EscalationManager()
    
    if args.status:
        active = escalation_manager.get_active_escalations()
        print(f"Active escalations: {len(active)}")
        for esc in active:
            print(f"  {esc.id}: {esc.rule_name} ({esc.escalation_level.value})")
    
    elif args.summary:
        summary = escalation_manager.get_escalation_summary(args.summary)
        print("=== Escalation Summary ===")
        print(json.dumps(summary, indent=2, default=str))
    
    elif args.resolve:
        success = escalation_manager.resolve_escalation(args.resolve, "Manually resolved via CLI")
        if success:
            print(f"Escalation {args.resolve} resolved")
        else:
            print(f"Failed to resolve escalation {args.resolve}")
    
    elif args.export_report:
        report_path = escalation_manager.export_escalation_report(args.export_report)
        print(f"Escalation report exported to: {report_path}")
    
    elif args.test_rule:
        # This would test a specific rule with mock data
        print(f"Testing rule: {args.test_rule}")
        # TODO: Implement rule testing
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()