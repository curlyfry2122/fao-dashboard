"""
Monitoring Dashboard

Real-time monitoring and status reporting for the Agent Orchestration system.
Provides comprehensive insights into agent performance, task execution, and system health.
"""

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
from dataclasses import asdict

from agent_orchestrator import AgentOrchestrator, AgentStatus, TaskStatus
from priority_matrix import Priority, AgentType


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for the agent orchestration system.
    
    Features:
    - Real-time agent status monitoring
    - Task execution metrics and trends
    - Performance analytics and insights
    - Health checks and alerting
    - Historical data analysis
    - Automated reporting
    """
    
    def __init__(self, orchestrator: AgentOrchestrator):
        """Initialize the monitoring dashboard."""
        self.orchestrator = orchestrator
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metric snapshots
        self.alert_history = deque(maxlen=100)     # Keep last 100 alerts
        self.performance_baselines = {}
        
        # Monitoring configuration
        self.alert_thresholds = {
            'task_failure_rate': 0.15,      # 15% failure rate
            'queue_size_warning': 50,       # 50 tasks in queue
            'queue_size_critical': 100,     # 100 tasks in queue
            'agent_health_warning': 0.7,    # 70% health score
            'agent_health_critical': 0.3,   # 30% health score
            'task_timeout_warning': 1800,   # 30 minutes
            'response_time_warning': 600    # 10 minutes average
        }
        
        self.reports_dir = Path(__file__).parent.parent / "automation" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize baseline metrics
        self._initialize_baselines()
    
    def _initialize_baselines(self) -> None:
        """Initialize performance baselines."""
        self.performance_baselines = {
            'avg_task_completion_time': {
                AgentType.DATA_PIPELINE: 240,      # 4 minutes
                AgentType.UI_DASHBOARD: 360,       # 6 minutes
                AgentType.DEPLOYMENT: 180,         # 3 minutes
                AgentType.PERFORMANCE_MONITOR: 480, # 8 minutes
                AgentType.SECURITY_SCAN: 120,      # 2 minutes
                AgentType.DEPENDENCY_UPDATE: 60    # 1 minute
            },
            'expected_success_rate': 0.95,  # 95% success rate
            'max_queue_time': 1800,         # 30 minutes max in queue
            'agent_utilization_target': 0.7 # 70% utilization target
        }
    
    def collect_metrics_snapshot(self) -> Dict[str, Any]:
        """Collect a comprehensive metrics snapshot."""
        timestamp = datetime.now()
        
        # Get orchestrator metrics
        orchestrator_metrics = self.orchestrator.get_metrics()
        queue_status = self.orchestrator.get_queue_status()
        
        # Agent-specific metrics
        agent_metrics = {}
        for agent_type, agent in self.orchestrator.agents.items():
            agent_metrics[agent_type.value] = {
                'status': agent.status.value,
                'health_score': agent.health_score,
                'current_load': agent.current_load,
                'max_load': agent.max_concurrent_tasks,
                'utilization': agent.current_load / agent.max_concurrent_tasks,
                'tasks_completed': agent.tasks_completed,
                'tasks_failed': agent.tasks_failed,
                'success_rate': (
                    agent.tasks_completed / max(1, agent.tasks_completed + agent.tasks_failed)
                ),
                'last_active': agent.last_active.isoformat() if agent.last_active else None
            }
        
        # Task execution metrics
        completed_tasks = self.orchestrator.completed_tasks
        task_metrics = self._calculate_task_metrics(completed_tasks)
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(completed_tasks)
        
        # Health indicators
        health_indicators = self._calculate_health_indicators(agent_metrics, queue_status)
        
        snapshot = {
            'timestamp': timestamp.isoformat(),
            'orchestrator': orchestrator_metrics,
            'queue': queue_status,
            'agents': agent_metrics,
            'tasks': task_metrics,
            'performance': performance_metrics,
            'health': health_indicators,
            'alerts': self._check_alerts(agent_metrics, queue_status, task_metrics)
        }
        
        # Store snapshot
        self.metrics_history.append(snapshot)
        
        return snapshot
    
    def _calculate_task_metrics(self, completed_tasks: List) -> Dict[str, Any]:
        """Calculate task execution metrics."""
        if not completed_tasks:
            return {
                'total_completed': 0,
                'success_rate': 0,
                'avg_completion_time': 0,
                'priority_distribution': {p.value: 0 for p in Priority},
                'agent_distribution': {a.value: 0 for a in AgentType}
            }
        
        # Recent tasks (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_tasks = [
            t for t in completed_tasks 
            if t.completed_at and t.completed_at >= cutoff_time
        ]
        
        if not recent_tasks:
            recent_tasks = completed_tasks[-50:]  # Fallback to last 50 tasks
        
        # Success rate
        successful_tasks = [t for t in recent_tasks if t.status == TaskStatus.COMPLETED]
        success_rate = len(successful_tasks) / len(recent_tasks) if recent_tasks else 0
        
        # Average completion time
        tasks_with_times = [
            t for t in successful_tasks 
            if t.started_at and t.completed_at
        ]
        avg_completion_time = 0
        if tasks_with_times:
            total_time = sum(
                (t.completed_at - t.started_at).total_seconds() 
                for t in tasks_with_times
            )
            avg_completion_time = total_time / len(tasks_with_times)
        
        # Priority distribution
        priority_dist = {p.value: 0 for p in Priority}
        for task in recent_tasks:
            priority_dist[task.priority.value] += 1
        
        # Agent distribution
        agent_dist = {a.value: 0 for a in AgentType}
        for task in recent_tasks:
            agent_dist[task.agent_type.value] += 1
        
        return {
            'total_completed': len(recent_tasks),
            'successful': len(successful_tasks),
            'failed': len(recent_tasks) - len(successful_tasks),
            'success_rate': success_rate,
            'avg_completion_time': avg_completion_time,
            'priority_distribution': priority_dist,
            'agent_distribution': agent_dist,
            'period': '24h'
        }
    
    def _calculate_performance_metrics(self, completed_tasks: List) -> Dict[str, Any]:
        """Calculate performance metrics and compare against baselines."""
        performance = {}
        
        # Recent tasks for performance analysis
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_tasks = [
            t for t in completed_tasks 
            if t.completed_at and t.completed_at >= cutoff_time
        ]
        
        # Performance by agent type
        agent_performance = {}
        for agent_type in AgentType:
            agent_tasks = [t for t in recent_tasks if t.agent_type == agent_type]
            
            if agent_tasks:
                successful_tasks = [t for t in agent_tasks if t.status == TaskStatus.COMPLETED]
                
                # Calculate metrics
                success_rate = len(successful_tasks) / len(agent_tasks)
                
                tasks_with_times = [
                    t for t in successful_tasks 
                    if t.started_at and t.completed_at
                ]
                avg_time = 0
                if tasks_with_times:
                    total_time = sum(
                        (t.completed_at - t.started_at).total_seconds() 
                        for t in tasks_with_times
                    )
                    avg_time = total_time / len(tasks_with_times)
                
                # Compare against baseline
                baseline_time = self.performance_baselines['avg_task_completion_time'].get(
                    agent_type, 300
                )
                baseline_success = self.performance_baselines['expected_success_rate']
                
                agent_performance[agent_type.value] = {
                    'tasks_processed': len(agent_tasks),
                    'success_rate': success_rate,
                    'avg_completion_time': avg_time,
                    'baseline_completion_time': baseline_time,
                    'baseline_success_rate': baseline_success,
                    'time_performance': 'good' if avg_time <= baseline_time * 1.2 else 'degraded',
                    'success_performance': 'good' if success_rate >= baseline_success * 0.9 else 'degraded'
                }
            else:
                agent_performance[agent_type.value] = {
                    'tasks_processed': 0,
                    'success_rate': 0,
                    'avg_completion_time': 0,
                    'time_performance': 'no_data',
                    'success_performance': 'no_data'
                }
        
        performance['agents'] = agent_performance
        
        # Overall system performance
        if recent_tasks:
            overall_success_rate = len([
                t for t in recent_tasks if t.status == TaskStatus.COMPLETED
            ]) / len(recent_tasks)
            
            performance['overall'] = {
                'success_rate': overall_success_rate,
                'baseline_success_rate': self.performance_baselines['expected_success_rate'],
                'performance_rating': (
                    'excellent' if overall_success_rate >= 0.95 else
                    'good' if overall_success_rate >= 0.90 else
                    'fair' if overall_success_rate >= 0.80 else
                    'poor'
                )
            }
        else:
            performance['overall'] = {
                'success_rate': 0,
                'performance_rating': 'no_data'
            }
        
        return performance
    
    def _calculate_health_indicators(
        self, agent_metrics: Dict, queue_status: Dict
    ) -> Dict[str, Any]:
        """Calculate system health indicators."""
        health = {
            'overall_status': 'healthy',
            'indicators': {},
            'warnings': [],
            'critical_issues': []
        }
        
        # Agent health
        agent_health_scores = [
            metrics['health_score'] for metrics in agent_metrics.values()
        ]
        avg_agent_health = sum(agent_health_scores) / len(agent_health_scores) if agent_health_scores else 0
        
        health['indicators']['agent_health'] = {
            'average_score': avg_agent_health,
            'status': (
                'healthy' if avg_agent_health >= 0.8 else
                'warning' if avg_agent_health >= 0.6 else
                'critical'
            )
        }
        
        # Queue health
        queue_size = queue_status.get('total_queued', 0)
        oldest_task_age = queue_status.get('oldest_task_age', 0)
        
        health['indicators']['queue_health'] = {
            'size': queue_size,
            'oldest_task_age_seconds': oldest_task_age,
            'status': (
                'healthy' if queue_size < 20 and oldest_task_age < 1800 else
                'warning' if queue_size < 50 and oldest_task_age < 3600 else
                'critical'
            )
        }
        
        # System utilization
        active_agents = len([
            agent for agent in agent_metrics.values() 
            if agent['status'] in ['busy', 'idle']
        ])
        total_agents = len(agent_metrics)
        
        avg_utilization = sum(
            agent['utilization'] for agent in agent_metrics.values()
        ) / total_agents if total_agents > 0 else 0
        
        health['indicators']['system_utilization'] = {
            'active_agents': active_agents,
            'total_agents': total_agents,
            'average_utilization': avg_utilization,
            'status': (
                'healthy' if 0.2 <= avg_utilization <= 0.8 else
                'warning' if avg_utilization <= 0.9 else
                'critical'
            )
        }
        
        # Determine overall status
        indicator_statuses = [
            indicator['status'] for indicator in health['indicators'].values()
        ]
        
        if 'critical' in indicator_statuses:
            health['overall_status'] = 'critical'
        elif 'warning' in indicator_statuses:
            health['overall_status'] = 'warning'
        
        return health
    
    def _check_alerts(
        self, agent_metrics: Dict, queue_status: Dict, task_metrics: Dict
    ) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        timestamp = datetime.now()
        
        # Queue size alerts
        queue_size = queue_status.get('total_queued', 0)
        if queue_size >= self.alert_thresholds['queue_size_critical']:
            alerts.append({
                'level': 'critical',
                'type': 'queue_size',
                'message': f'Queue size critical: {queue_size} tasks',
                'timestamp': timestamp.isoformat(),
                'value': queue_size,
                'threshold': self.alert_thresholds['queue_size_critical']
            })
        elif queue_size >= self.alert_thresholds['queue_size_warning']:
            alerts.append({
                'level': 'warning',
                'type': 'queue_size',
                'message': f'Queue size high: {queue_size} tasks',
                'timestamp': timestamp.isoformat(),
                'value': queue_size,
                'threshold': self.alert_thresholds['queue_size_warning']
            })
        
        # Agent health alerts
        for agent_type, metrics in agent_metrics.items():
            health_score = metrics['health_score']
            if health_score <= self.alert_thresholds['agent_health_critical']:
                alerts.append({
                    'level': 'critical',
                    'type': 'agent_health',
                    'message': f'Agent {agent_type} health critical: {health_score:.2f}',
                    'timestamp': timestamp.isoformat(),
                    'agent': agent_type,
                    'value': health_score,
                    'threshold': self.alert_thresholds['agent_health_critical']
                })
            elif health_score <= self.alert_thresholds['agent_health_warning']:
                alerts.append({
                    'level': 'warning',
                    'type': 'agent_health',
                    'message': f'Agent {agent_type} health degraded: {health_score:.2f}',
                    'timestamp': timestamp.isoformat(),
                    'agent': agent_type,
                    'value': health_score,
                    'threshold': self.alert_thresholds['agent_health_warning']
                })
        
        # Task failure rate alerts
        success_rate = task_metrics.get('success_rate', 1.0)
        failure_rate = 1.0 - success_rate
        if failure_rate > self.alert_thresholds['task_failure_rate']:
            alerts.append({
                'level': 'warning',
                'type': 'task_failure_rate',
                'message': f'High task failure rate: {failure_rate:.1%}',
                'timestamp': timestamp.isoformat(),
                'value': failure_rate,
                'threshold': self.alert_thresholds['task_failure_rate']
            })
        
        # Store alerts in history
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate a comprehensive status report."""
        snapshot = self.collect_metrics_snapshot()
        
        # Add trending data
        if len(self.metrics_history) >= 2:
            trends = self._calculate_trends()
            snapshot['trends'] = trends
        
        # Add recommendations
        snapshot['recommendations'] = self._generate_recommendations(snapshot)
        
        return snapshot
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate trends from historical data."""
        if len(self.metrics_history) < 10:
            return {'available': False, 'reason': 'insufficient_data'}
        
        # Get recent snapshots for trending
        recent_snapshots = list(self.metrics_history)[-10:]
        
        trends = {
            'available': True,
            'period': '10_snapshots',
            'task_completion': [],
            'queue_size': [],
            'agent_health': {},
            'success_rate': []
        }
        
        # Extract time series data
        for snapshot in recent_snapshots:
            timestamp = snapshot['timestamp']
            
            # Task completion trends
            trends['task_completion'].append({
                'timestamp': timestamp,
                'value': snapshot.get('tasks', {}).get('total_completed', 0)
            })
            
            # Queue size trends
            trends['queue_size'].append({
                'timestamp': timestamp,
                'value': snapshot.get('queue', {}).get('total_queued', 0)
            })
            
            # Success rate trends
            trends['success_rate'].append({
                'timestamp': timestamp,
                'value': snapshot.get('tasks', {}).get('success_rate', 0)
            })
            
            # Agent health trends
            for agent_type, metrics in snapshot.get('agents', {}).items():
                if agent_type not in trends['agent_health']:
                    trends['agent_health'][agent_type] = []
                
                trends['agent_health'][agent_type].append({
                    'timestamp': timestamp,
                    'value': metrics.get('health_score', 0)
                })
        
        # Calculate trend directions
        trends['analysis'] = {
            'queue_size_trend': self._calculate_trend_direction([
                point['value'] for point in trends['queue_size']
            ]),
            'success_rate_trend': self._calculate_trend_direction([
                point['value'] for point in trends['success_rate']
            ]),
            'completion_rate_trend': self._calculate_trend_direction([
                point['value'] for point in trends['task_completion']
            ])
        }
        
        return trends
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Simple linear regression slope
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        if abs(slope) < 0.01:  # Threshold for "stable"
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _generate_recommendations(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on current status."""
        recommendations = []
        
        # Queue size recommendations
        queue_size = snapshot.get('queue', {}).get('total_queued', 0)
        if queue_size > 30:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'High queue size detected',
                'description': f'Queue has {queue_size} pending tasks',
                'actions': [
                    'Consider scaling up agent capacity',
                    'Review task priorities and dependencies',
                    'Check for stuck or failed agents'
                ]
            })
        
        # Agent health recommendations
        agent_metrics = snapshot.get('agents', {})
        unhealthy_agents = [
            agent_type for agent_type, metrics in agent_metrics.items()
            if metrics.get('health_score', 1.0) < 0.7
        ]
        
        if unhealthy_agents:
            recommendations.append({
                'type': 'health',
                'priority': 'medium',
                'title': 'Agent health issues detected',
                'description': f'Agents with low health scores: {", ".join(unhealthy_agents)}',
                'actions': [
                    'Review agent logs for errors',
                    'Consider restarting unhealthy agents',
                    'Check system resources and dependencies'
                ]
            })
        
        # Performance recommendations
        performance = snapshot.get('performance', {})
        overall_success = performance.get('overall', {}).get('success_rate', 1.0)
        
        if overall_success < 0.9:
            recommendations.append({
                'type': 'reliability',
                'priority': 'high',
                'title': 'Low task success rate',
                'description': f'Success rate is {overall_success:.1%}',
                'actions': [
                    'Investigate failing tasks patterns',
                    'Review error logs and retry policies',
                    'Consider adjusting task timeouts',
                    'Validate system dependencies'
                ]
            })
        
        # Utilization recommendations
        health = snapshot.get('health', {})
        utilization = health.get('indicators', {}).get('system_utilization', {}).get('average_utilization', 0)
        
        if utilization < 0.2:
            recommendations.append({
                'type': 'efficiency',
                'priority': 'low',
                'title': 'Low system utilization',
                'description': f'Average utilization is {utilization:.1%}',
                'actions': [
                    'Consider reducing agent capacity if consistently low',
                    'Review task scheduling and distribution',
                    'Optimize task execution times'
                ]
            })
        elif utilization > 0.9:
            recommendations.append({
                'type': 'capacity',
                'priority': 'medium',
                'title': 'High system utilization',
                'description': f'Average utilization is {utilization:.1%}',
                'actions': [
                    'Consider increasing agent capacity',
                    'Review task priorities and load balancing',
                    'Monitor for performance degradation'
                ]
            })
        
        return recommendations
    
    def export_metrics_to_csv(self, hours: int = 24) -> Path:
        """Export recent metrics to CSV for analysis."""
        if not self.metrics_history:
            raise ValueError("No metrics data available")
        
        # Filter recent data
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            snapshot for snapshot in self.metrics_history
            if datetime.fromisoformat(snapshot['timestamp']) >= cutoff_time
        ]
        
        if not recent_data:
            recent_data = list(self.metrics_history)[-50:]  # Fallback to last 50 snapshots
        
        # Flatten data for CSV
        csv_data = []
        for snapshot in recent_data:
            base_row = {
                'timestamp': snapshot['timestamp'],
                'queue_size': snapshot.get('queue', {}).get('total_queued', 0),
                'active_tasks': snapshot.get('queue', {}).get('total_active', 0),
                'completed_tasks': snapshot.get('tasks', {}).get('total_completed', 0),
                'success_rate': snapshot.get('tasks', {}).get('success_rate', 0),
                'avg_completion_time': snapshot.get('tasks', {}).get('avg_completion_time', 0)
            }
            
            # Add agent-specific metrics
            for agent_type, metrics in snapshot.get('agents', {}).items():
                base_row.update({
                    f'{agent_type}_health': metrics.get('health_score', 0),
                    f'{agent_type}_utilization': metrics.get('utilization', 0),
                    f'{agent_type}_tasks_completed': metrics.get('tasks_completed', 0),
                    f'{agent_type}_tasks_failed': metrics.get('tasks_failed', 0)
                })
            
            csv_data.append(base_row)
        
        # Create DataFrame and export
        df = pd.DataFrame(csv_data)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.reports_dir / f"orchestrator_metrics_{timestamp_str}.csv"
        
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def generate_html_report(self) -> Path:
        """Generate an HTML status report."""
        snapshot = self.generate_status_report()
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Agent Orchestration Status Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        .status-healthy {{ color: green; }}
        .status-warning {{ color: orange; }}
        .status-critical {{ color: red; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .alert-warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; }}
        .alert-critical {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Agent Orchestration Status Report</h1>
        <p><strong>Generated:</strong> {timestamp_str}</p>
        <p><strong>Overall Status:</strong> 
            <span class="status-{snapshot.get('health', {}).get('overall_status', 'unknown')}">
                {snapshot.get('health', {}).get('overall_status', 'Unknown').upper()}
            </span>
        </p>
    </div>
    
    <div class="section">
        <h2>System Overview</h2>
        <div class="metric">
            <strong>Queue Size:</strong> {snapshot.get('queue', {}).get('total_queued', 0)}
        </div>
        <div class="metric">
            <strong>Active Tasks:</strong> {snapshot.get('queue', {}).get('total_active', 0)}
        </div>
        <div class="metric">
            <strong>Success Rate:</strong> {snapshot.get('tasks', {}).get('success_rate', 0):.1%}
        </div>
        <div class="metric">
            <strong>Avg Completion Time:</strong> {snapshot.get('tasks', {}).get('avg_completion_time', 0):.0f}s
        </div>
    </div>
    
    <div class="section">
        <h2>Agent Status</h2>
        <table>
            <tr>
                <th>Agent</th>
                <th>Status</th>
                <th>Health Score</th>
                <th>Utilization</th>
                <th>Tasks Completed</th>
                <th>Tasks Failed</th>
                <th>Success Rate</th>
            </tr>
        """
        
        # Add agent rows
        for agent_type, metrics in snapshot.get('agents', {}).items():
            success_rate = metrics.get('success_rate', 0)
            html_content += f"""
            <tr>
                <td>{agent_type}</td>
                <td>{metrics.get('status', 'unknown')}</td>
                <td>{metrics.get('health_score', 0):.2f}</td>
                <td>{metrics.get('utilization', 0):.1%}</td>
                <td>{metrics.get('tasks_completed', 0)}</td>
                <td>{metrics.get('tasks_failed', 0)}</td>
                <td>{success_rate:.1%}</td>
            </tr>
            """
        
        html_content += """
        </table>
    </div>
    """
        
        # Add alerts section
        alerts = snapshot.get('alerts', [])
        if alerts:
            html_content += """
    <div class="section">
        <h2>Active Alerts</h2>
        """
            for alert in alerts:
                alert_class = f"alert-{alert['level']}"
                html_content += f"""
        <div class="alert {alert_class}">
            <strong>{alert['level'].upper()}:</strong> {alert['message']}
        </div>
                """
            html_content += "</div>"
        
        # Add recommendations section
        recommendations = snapshot.get('recommendations', [])
        if recommendations:
            html_content += """
    <div class="section">
        <h2>Recommendations</h2>
        """
            for rec in recommendations:
                html_content += f"""
        <div style="margin-bottom: 15px;">
            <h4>{rec['title']} (Priority: {rec['priority']})</h4>
            <p>{rec['description']}</p>
            <ul>
        """
                for action in rec['actions']:
                    html_content += f"<li>{action}</li>"
                html_content += "</ul></div>"
            html_content += "</div>"
        
        html_content += """
</body>
</html>
        """
        
        # Save HTML report
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = self.reports_dir / f"orchestrator_report_{timestamp_str}.html"
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
        ]
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'by_level': {'critical': 0, 'warning': 0, 'info': 0},
                'by_type': {},
                'recent_alerts': []
            }
        
        # Count by level
        by_level = {'critical': 0, 'warning': 0, 'info': 0}
        for alert in recent_alerts:
            level = alert.get('level', 'info')
            by_level[level] = by_level.get(level, 0) + 1
        
        # Count by type
        by_type = {}
        for alert in recent_alerts:
            alert_type = alert.get('type', 'unknown')
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'by_level': by_level,
            'by_type': by_type,
            'recent_alerts': recent_alerts[-10:]  # Last 10 alerts
        }


def main():
    """CLI interface for the monitoring dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Orchestration Monitoring Dashboard")
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--alerts', action='store_true', help='Show recent alerts')
    parser.add_argument('--export-csv', type=int, metavar='HOURS', help='Export metrics to CSV')
    parser.add_argument('--generate-report', action='store_true', help='Generate HTML report')
    parser.add_argument('--trends', action='store_true', help='Show trends analysis')
    
    args = parser.parse_args()
    
    # Mock orchestrator for demonstration
    from agent_orchestrator import AgentOrchestrator
    orchestrator = AgentOrchestrator()
    dashboard = MonitoringDashboard(orchestrator)
    
    if args.status:
        status = dashboard.generate_status_report()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.alerts:
        alerts = dashboard.get_alert_summary()
        print("=== Recent Alerts Summary ===")
        print(f"Total alerts (24h): {alerts['total_alerts']}")
        print(f"By level: {alerts['by_level']}")
        print(f"By type: {alerts['by_type']}")
        
        if alerts['recent_alerts']:
            print("\nRecent alerts:")
            for alert in alerts['recent_alerts']:
                print(f"  {alert['level'].upper()}: {alert['message']}")
    
    elif args.export_csv:
        try:
            csv_path = dashboard.export_metrics_to_csv(args.export_csv)
            print(f"Metrics exported to: {csv_path}")
        except ValueError as e:
            print(f"Export failed: {e}")
    
    elif args.generate_report:
        html_path = dashboard.generate_html_report()
        print(f"HTML report generated: {html_path}")
    
    elif args.trends:
        snapshot = dashboard.collect_metrics_snapshot()
        trends = snapshot.get('trends', {})
        
        if trends.get('available'):
            print("=== Trends Analysis ===")
            analysis = trends.get('analysis', {})
            for metric, trend in analysis.items():
                print(f"{metric}: {trend}")
        else:
            print("Trends analysis not available: insufficient data")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()