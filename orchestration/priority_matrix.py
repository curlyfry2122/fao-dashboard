"""
Agent Orchestration Priority Matrix

This module implements intelligent issue classification and agent routing
for the FAO Food Price Index Dashboard project.
"""

import json
import re
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class Priority(Enum):
    """Issue priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentType(Enum):
    """Available agent types for handling different issue categories."""
    DATA_PIPELINE = "data-pipeline"
    UI_DASHBOARD = "ui-dashboard"
    DEPLOYMENT = "deployment"
    PERFORMANCE_MONITOR = "performance-monitor"
    SECURITY_SCAN = "security-scan"
    DEPENDENCY_UPDATE = "dependency-update"
    PYTEST_TEST_GENERATOR = "pytest-test-generator"
    GENERAL = "general"


@dataclass
class IssueClassification:
    """Classification result for an issue."""
    priority: Priority
    agent_type: AgentType
    escalate: bool
    confidence: float
    reasoning: List[str]
    estimated_resolution_time: timedelta
    required_expertise: List[str]
    dependencies: List[str]


@dataclass
class AgentCapability:
    """Defines an agent's capabilities and constraints."""
    agent_type: AgentType
    max_concurrent_issues: int
    expertise_areas: List[str]
    average_resolution_time: timedelta
    available_hours: List[int]  # Hours of day when agent is active
    escalation_threshold: int  # Max issues before escalation


class PriorityMatrix:
    """
    Intelligent priority matrix for classifying issues and routing to appropriate agents.
    
    This system analyzes issue content, context, and historical patterns to:
    1. Determine issue priority level
    2. Route to appropriate specialized agent
    3. Estimate resolution time and required resources
    4. Trigger escalation when necessary
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the priority matrix with configuration."""
        self.config_path = config_path or Path(__file__).parent / "matrix_config.yml"
        self.load_configuration()
        self.load_historical_data()
    
    def load_configuration(self) -> None:
        """Load priority matrix configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self.get_default_configuration()
            self.save_configuration(config)
        
        self.keywords = config.get('keywords', {})
        self.weight_factors = config.get('weight_factors', {})
        self.escalation_rules = config.get('escalation_rules', {})
        self.agent_capabilities = {
            AgentType(agent_type): AgentCapability(**capability)
            for agent_type, capability in config.get('agent_capabilities', {}).items()
        }
    
    def get_default_configuration(self) -> Dict:
        """Get default priority matrix configuration."""
        return {
            'keywords': {
                'critical': [
                    'down', 'outage', 'crash', 'critical', 'urgent', 'emergency',
                    'data loss', 'security breach', 'vulnerability', 'production failure',
                    'memory leak', 'infinite loop', 'deadlock', 'corruption'
                ],
                'high': [
                    'error', 'bug', 'failure', 'broken', 'not working', 'exception',
                    'timeout', 'slow performance', 'high cpu', 'high memory',
                    'incorrect data', 'missing data', 'authentication failed'
                ],
                'medium': [
                    'enhancement', 'feature request', 'improvement', 'optimization',
                    'refactor', 'update', 'upgrade', 'documentation',
                    'usability', 'ui improvement', 'performance tuning'
                ],
                'low': [
                    'question', 'discussion', 'clarification', 'minor', 'cosmetic',
                    'typo', 'spelling', 'style', 'formatting', 'comment'
                ]
            },
            'agent_keywords': {
                'data-pipeline': [
                    'data', 'fao', 'fetch', 'download', 'pipeline', 'cache', 'excel',
                    'csv', 'parsing', 'etl', 'transform', 'load', 'import',
                    'data quality', 'missing values', 'data validation'
                ],
                'ui-dashboard': [
                    'ui', 'streamlit', 'dashboard', 'chart', 'visualization', 'plot',
                    'graph', 'display', 'interface', 'frontend', 'user experience',
                    'layout', 'styling', 'responsive', 'mobile'
                ],
                'deployment': [
                    'deploy', 'deployment', 'docker', 'container', 'heroku', 'cloud',
                    'ci/cd', 'build', 'release', 'production', 'staging',
                    'environment', 'configuration', 'setup'
                ],
                'performance-monitor': [
                    'performance', 'slow', 'fast', 'memory', 'cpu', 'optimization',
                    'bottleneck', 'latency', 'throughput', 'scalability',
                    'caching', 'efficiency', 'resource usage'
                ],
                'security-scan': [
                    'security', 'vulnerability', 'cve', 'auth', 'authentication',
                    'authorization', 'permission', 'access', 'encrypt', 'decrypt',
                    'ssl', 'tls', 'token', 'credential', 'privacy'
                ],
                'dependency-update': [
                    'dependency', 'package', 'library', 'version', 'update',
                    'upgrade', 'requirements', 'pip', 'npm', 'outdated',
                    'compatibility', 'breaking change'
                ]
            },
            'weight_factors': {
                'title_multiplier': 2.0,
                'label_multiplier': 1.5,
                'body_multiplier': 1.0,
                'author_reputation_multiplier': 0.2,
                'time_sensitivity_multiplier': 1.3
            },
            'escalation_rules': {
                'critical_immediate': True,
                'high_after_hours': 2,
                'medium_after_hours': 24,
                'low_after_hours': 168,
                'max_agent_load': 5,
                'cross_team_threshold': 3
            },
            'agent_capabilities': {
                'data-pipeline': {
                    'agent_type': 'data-pipeline',
                    'max_concurrent_issues': 3,
                    'expertise_areas': ['data processing', 'ETL', 'caching', 'FAO integration'],
                    'average_resolution_time': {'days': 0, 'hours': 4, 'minutes': 0},
                    'available_hours': list(range(24)),  # 24/7 for automated agent
                    'escalation_threshold': 2
                },
                'ui-dashboard': {
                    'agent_type': 'ui-dashboard',
                    'max_concurrent_issues': 4,
                    'expertise_areas': ['Streamlit', 'UI/UX', 'visualization', 'frontend'],
                    'average_resolution_time': {'days': 0, 'hours': 6, 'minutes': 0},
                    'available_hours': list(range(9, 18)),  # Business hours
                    'escalation_threshold': 3
                },
                'deployment': {
                    'agent_type': 'deployment',
                    'max_concurrent_issues': 2,
                    'expertise_areas': ['Docker', 'CI/CD', 'cloud deployment', 'DevOps'],
                    'average_resolution_time': {'days': 0, 'hours': 3, 'minutes': 0},
                    'available_hours': list(range(24)),  # 24/7 for deployment issues
                    'escalation_threshold': 1
                },
                'performance-monitor': {
                    'agent_type': 'performance-monitor',
                    'max_concurrent_issues': 3,
                    'expertise_areas': ['performance optimization', 'monitoring', 'profiling'],
                    'average_resolution_time': {'days': 0, 'hours': 8, 'minutes': 0},
                    'available_hours': list(range(24)),  # 24/7 monitoring
                    'escalation_threshold': 2
                },
                'security-scan': {
                    'agent_type': 'security-scan',
                    'max_concurrent_issues': 2,
                    'expertise_areas': ['security analysis', 'vulnerability assessment'],
                    'average_resolution_time': {'days': 0, 'hours': 2, 'minutes': 0},
                    'available_hours': list(range(24)),  # 24/7 for security
                    'escalation_threshold': 1
                },
                'dependency-update': {
                    'agent_type': 'dependency-update',
                    'max_concurrent_issues': 5,
                    'expertise_areas': ['dependency management', 'version control'],
                    'average_resolution_time': {'days': 0, 'hours': 1, 'minutes': 0},
                    'available_hours': list(range(24)),  # 24/7 automated updates
                    'escalation_threshold': 4
                }
            }
        }
    
    def save_configuration(self, config: Dict) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def load_historical_data(self) -> None:
        """Load historical issue data for pattern analysis."""
        history_path = self.config_path.parent / "issue_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.issue_history = json.load(f)
        else:
            self.issue_history = []
    
    def save_historical_data(self) -> None:
        """Save historical issue data."""
        history_path = self.config_path.parent / "issue_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.issue_history, f, indent=2, default=str)
    
    def classify_issue(
        self,
        title: str,
        body: str = "",
        labels: Optional[List[str]] = None,
        author: Optional[str] = None,
        created_at: Optional[datetime] = None
    ) -> IssueClassification:
        """
        Classify an issue and determine priority, agent assignment, and escalation needs.
        
        Args:
            title: Issue title
            body: Issue description/body
            labels: List of issue labels
            author: Issue author username
            created_at: When the issue was created
            
        Returns:
            IssueClassification with priority, agent type, and routing information
        """
        if labels is None:
            labels = []
        if created_at is None:
            created_at = datetime.now()
        
        # Normalize text for analysis
        title_lower = title.lower().strip()
        body_lower = body.lower().strip() if body else ""
        combined_text = f"{title_lower} {body_lower}"
        
        # Calculate priority score
        priority_scores = self._calculate_priority_scores(title_lower, body_lower, labels)
        priority = self._determine_priority(priority_scores)
        
        # Determine agent type
        agent_scores = self._calculate_agent_scores(combined_text, labels)
        agent_type = self._determine_agent_type(agent_scores)
        
        # Calculate confidence based on keyword matches and pattern strength
        confidence = self._calculate_confidence(priority_scores, agent_scores)
        
        # Determine if escalation is needed
        escalate = self._should_escalate(priority, agent_type, created_at, author)
        
        # Estimate resolution time
        estimated_time = self._estimate_resolution_time(priority, agent_type, combined_text)
        
        # Determine required expertise
        expertise = self._determine_required_expertise(agent_type, combined_text)
        
        # Identify dependencies
        dependencies = self._identify_dependencies(combined_text, labels)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            priority_scores, agent_scores, priority, agent_type, escalate
        )
        
        classification = IssueClassification(
            priority=priority,
            agent_type=agent_type,
            escalate=escalate,
            confidence=confidence,
            reasoning=reasoning,
            estimated_resolution_time=estimated_time,
            required_expertise=expertise,
            dependencies=dependencies
        )
        
        # Record for historical analysis
        self._record_classification(title, body, labels, classification)
        
        return classification
    
    def _calculate_priority_scores(
        self, title: str, body: str, labels: List[str]
    ) -> Dict[str, float]:
        """Calculate priority scores based on keyword matching."""
        scores = {priority: 0.0 for priority in Priority}
        
        for priority_level, keywords in self.keywords.items():
            if priority_level in [p.value for p in Priority]:
                score = 0.0
                
                # Title keywords (higher weight)
                for keyword in keywords:
                    if keyword in title:
                        score += self.weight_factors.get('title_multiplier', 2.0)
                
                # Body keywords
                for keyword in keywords:
                    if keyword in body:
                        score += self.weight_factors.get('body_multiplier', 1.0)
                
                # Label keywords
                for label in labels:
                    if any(keyword in label.lower() for keyword in keywords):
                        score += self.weight_factors.get('label_multiplier', 1.5)
                
                scores[Priority(priority_level)] = score
        
        return scores
    
    def _calculate_agent_scores(
        self, combined_text: str, labels: List[str]
    ) -> Dict[AgentType, float]:
        """Calculate agent type scores based on keyword matching."""
        scores = {agent_type: 0.0 for agent_type in AgentType}
        
        for agent_key, keywords in self.keywords.get('agent_keywords', {}).items():
            if agent_key in [a.value for a in AgentType]:
                agent_type = AgentType(agent_key)
                score = 0.0
                
                # Text keywords
                for keyword in keywords:
                    if keyword in combined_text:
                        score += 1.0
                
                # Label keywords
                for label in labels:
                    if any(keyword in label.lower() for keyword in keywords):
                        score += 1.5
                
                scores[agent_type] = score
        
        return scores
    
    def _determine_priority(self, priority_scores: Dict[Priority, float]) -> Priority:
        """Determine priority based on scores."""
        if not priority_scores or all(score == 0 for score in priority_scores.values()):
            return Priority.MEDIUM
        
        # Get the priority with the highest score
        max_priority = max(priority_scores.items(), key=lambda x: x[1])
        
        # If no strong signals, default to medium
        if max_priority[1] < 1.0:
            return Priority.MEDIUM
        
        return max_priority[0]
    
    def _determine_agent_type(self, agent_scores: Dict[AgentType, float]) -> AgentType:
        """Determine agent type based on scores."""
        if not agent_scores or all(score == 0 for score in agent_scores.values()):
            return AgentType.GENERAL
        
        # Get the agent type with the highest score
        max_agent = max(agent_scores.items(), key=lambda x: x[1])
        
        # If no strong signals, default to general
        if max_agent[1] < 1.0:
            return AgentType.GENERAL
        
        return max_agent[0]
    
    def _calculate_confidence(
        self, priority_scores: Dict[Priority, float], agent_scores: Dict[AgentType, float]
    ) -> float:
        """Calculate classification confidence score."""
        priority_max = max(priority_scores.values()) if priority_scores else 0
        priority_total = sum(priority_scores.values()) if priority_scores else 1
        
        agent_max = max(agent_scores.values()) if agent_scores else 0
        agent_total = sum(agent_scores.values()) if agent_scores else 1
        
        # Confidence based on how dominant the top choice is
        priority_confidence = priority_max / priority_total if priority_total > 0 else 0
        agent_confidence = agent_max / agent_total if agent_total > 0 else 0
        
        # Average confidence, capped at 1.0
        return min(1.0, (priority_confidence + agent_confidence) / 2)
    
    def _should_escalate(
        self, priority: Priority, agent_type: AgentType, created_at: datetime, author: Optional[str]
    ) -> bool:
        """Determine if issue should be escalated."""
        # Critical issues always escalate immediately
        if priority == Priority.CRITICAL:
            return True
        
        # High priority issues escalate after threshold
        if priority == Priority.HIGH:
            threshold_hours = self.escalation_rules.get('high_after_hours', 2)
            if (datetime.now() - created_at).total_seconds() / 3600 > threshold_hours:
                return True
        
        # Check agent load
        if agent_type in self.agent_capabilities:
            capability = self.agent_capabilities[agent_type]
            # This would be implemented with actual issue tracking
            # For now, we'll use a simple heuristic
            if priority in [Priority.CRITICAL, Priority.HIGH]:
                return True
        
        return False
    
    def _estimate_resolution_time(
        self, priority: Priority, agent_type: AgentType, combined_text: str
    ) -> timedelta:
        """Estimate resolution time based on priority, agent type, and complexity."""
        # Base time from agent capabilities
        if agent_type in self.agent_capabilities:
            base_time = self.agent_capabilities[agent_type].average_resolution_time
        else:
            base_time = timedelta(hours=4)  # Default
        
        # Priority multipliers
        priority_multipliers = {
            Priority.CRITICAL: 0.5,  # Rush job
            Priority.HIGH: 0.8,
            Priority.MEDIUM: 1.0,
            Priority.LOW: 1.5
        }
        
        # Complexity indicators
        complexity_keywords = [
            'integration', 'complex', 'multiple', 'architecture', 'design',
            'refactor', 'migration', 'performance', 'scalability'
        ]
        
        complexity_multiplier = 1.0
        for keyword in complexity_keywords:
            if keyword in combined_text:
                complexity_multiplier += 0.2
        
        # Cap complexity multiplier
        complexity_multiplier = min(2.0, complexity_multiplier)
        
        # Calculate final estimate
        multiplier = priority_multipliers.get(priority, 1.0) * complexity_multiplier
        estimated_time = timedelta(seconds=base_time.total_seconds() * multiplier)
        
        return estimated_time
    
    def _determine_required_expertise(self, agent_type: AgentType, combined_text: str) -> List[str]:
        """Determine required expertise areas."""
        if agent_type in self.agent_capabilities:
            base_expertise = self.agent_capabilities[agent_type].expertise_areas[:]
        else:
            base_expertise = ['general development']
        
        # Add specific expertise based on content
        additional_expertise = []
        
        if 'database' in combined_text or 'sql' in combined_text:
            additional_expertise.append('database administration')
        if 'api' in combined_text or 'rest' in combined_text:
            additional_expertise.append('API development')
        if 'machine learning' in combined_text or 'ml' in combined_text:
            additional_expertise.append('machine learning')
        if 'statistics' in combined_text or 'analysis' in combined_text:
            additional_expertise.append('statistical analysis')
        
        return base_expertise + additional_expertise
    
    def _identify_dependencies(self, combined_text: str, labels: List[str]) -> List[str]:
        """Identify issue dependencies and blockers."""
        dependencies = []
        
        # Common dependency patterns
        if 'depends on' in combined_text or 'blocked by' in combined_text:
            dependencies.append('external dependency')
        
        if 'infrastructure' in combined_text or 'server' in combined_text:
            dependencies.append('infrastructure')
        
        if 'approval' in combined_text or 'review' in combined_text:
            dependencies.append('approval process')
        
        if 'third party' in combined_text or 'vendor' in combined_text:
            dependencies.append('third party service')
        
        # Check labels for dependencies
        for label in labels:
            if 'dependency' in label.lower() or 'blocked' in label.lower():
                dependencies.append('labeled dependency')
        
        return dependencies
    
    def _generate_reasoning(
        self,
        priority_scores: Dict[Priority, float],
        agent_scores: Dict[AgentType, float],
        final_priority: Priority,
        final_agent: AgentType,
        escalate: bool
    ) -> List[str]:
        """Generate human-readable reasoning for the classification."""
        reasoning = []
        
        # Priority reasoning
        if final_priority == Priority.CRITICAL:
            reasoning.append("Classified as CRITICAL due to keywords indicating system failure or security issues")
        elif final_priority == Priority.HIGH:
            reasoning.append("Classified as HIGH priority due to error indicators and functional impact")
        elif final_priority == Priority.MEDIUM:
            reasoning.append("Classified as MEDIUM priority - enhancement or improvement request")
        else:
            reasoning.append("Classified as LOW priority - minor issue or question")
        
        # Agent assignment reasoning
        if final_agent != AgentType.GENERAL:
            reasoning.append(f"Assigned to {final_agent.value} agent based on domain-specific keywords")
        else:
            reasoning.append("Assigned to general agent - no specific domain identified")
        
        # Escalation reasoning
        if escalate:
            reasoning.append("Marked for escalation due to priority level or time sensitivity")
        
        return reasoning
    
    def _record_classification(
        self, title: str, body: str, labels: List[str], classification: IssueClassification
    ) -> None:
        """Record classification for historical analysis."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'body_length': len(body),
            'labels': labels,
            'classification': asdict(classification)
        }
        
        self.issue_history.append(record)
        
        # Keep only last 1000 records
        if len(self.issue_history) > 1000:
            self.issue_history = self.issue_history[-1000:]
        
        # Save periodically
        if len(self.issue_history) % 10 == 0:
            self.save_historical_data()
    
    def get_agent_workload(self, agent_type: AgentType) -> Dict[str, Union[int, float]]:
        """Get current workload for an agent type."""
        # This would integrate with actual issue tracking
        # For now, return mock data
        if agent_type in self.agent_capabilities:
            capability = self.agent_capabilities[agent_type]
            return {
                'current_issues': 2,  # Mock data
                'max_capacity': capability.max_concurrent_issues,
                'utilization': 0.4,  # 40% utilization
                'avg_resolution_time': capability.average_resolution_time.total_seconds() / 3600
            }
        return {'current_issues': 0, 'max_capacity': 1, 'utilization': 0.0, 'avg_resolution_time': 4.0}
    
    def suggest_optimization(self) -> List[str]:
        """Suggest optimizations based on historical data."""
        suggestions = []
        
        if len(self.issue_history) < 10:
            suggestions.append("Collect more historical data for better analysis")
            return suggestions
        
        # Analyze recent patterns
        recent_issues = self.issue_history[-50:]  # Last 50 issues
        
        # Priority distribution
        priorities = [issue['classification']['priority'] for issue in recent_issues]
        critical_count = priorities.count('critical')
        
        if critical_count > len(recent_issues) * 0.1:  # More than 10% critical
            suggestions.append("High number of critical issues - review processes to prevent escalation")
        
        # Agent distribution
        agents = [issue['classification']['agent_type'] for issue in recent_issues]
        general_count = agents.count('general')
        
        if general_count > len(recent_issues) * 0.3:  # More than 30% general
            suggestions.append("Many issues assigned to general agent - refine classification keywords")
        
        # Resolution time patterns
        # This would require actual resolution data
        suggestions.append("Consider implementing feedback loop for resolution time accuracy")
        
        return suggestions


def main():
    """CLI interface for the priority matrix."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FAO Dashboard Priority Matrix")
    parser.add_argument('--classify', help='Classify an issue', nargs=2, 
                       metavar=('TITLE', 'BODY'))
    parser.add_argument('--config', help='Show current configuration', action='store_true')
    parser.add_argument('--optimize', help='Show optimization suggestions', action='store_true')
    parser.add_argument('--workload', help='Show agent workloads', action='store_true')
    
    args = parser.parse_args()
    
    matrix = PriorityMatrix()
    
    if args.classify:
        title, body = args.classify
        classification = matrix.classify_issue(title, body)
        
        print("=== Issue Classification ===")
        print(f"Priority: {classification.priority.value.upper()}")
        print(f"Agent: {classification.agent_type.value}")
        print(f"Escalate: {'Yes' if classification.escalate else 'No'}")
        print(f"Confidence: {classification.confidence:.2f}")
        print(f"Estimated Resolution: {classification.estimated_resolution_time}")
        print(f"Required Expertise: {', '.join(classification.required_expertise)}")
        print(f"Dependencies: {', '.join(classification.dependencies) if classification.dependencies else 'None'}")
        print("\nReasoning:")
        for reason in classification.reasoning:
            print(f"  - {reason}")
    
    elif args.config:
        print("=== Priority Matrix Configuration ===")
        print(f"Configuration file: {matrix.config_path}")
        print(f"Historical records: {len(matrix.issue_history)}")
        print(f"Agent types: {len(matrix.agent_capabilities)}")
        
        print("\nAgent Capabilities:")
        for agent_type, capability in matrix.agent_capabilities.items():
            print(f"  {agent_type.value}:")
            print(f"    Max issues: {capability.max_concurrent_issues}")
            print(f"    Avg resolution: {capability.average_resolution_time}")
            print(f"    Expertise: {', '.join(capability.expertise_areas)}")
    
    elif args.workload:
        print("=== Agent Workload Status ===")
        for agent_type in AgentType:
            workload = matrix.get_agent_workload(agent_type)
            utilization = workload['utilization'] * 100
            print(f"{agent_type.value}:")
            print(f"  Current: {workload['current_issues']}/{workload['max_capacity']} issues")
            print(f"  Utilization: {utilization:.1f}%")
            print(f"  Avg resolution: {workload['avg_resolution_time']:.1f}h")
    
    elif args.optimize:
        suggestions = matrix.suggest_optimization()
        print("=== Optimization Suggestions ===")
        if suggestions:
            for suggestion in suggestions:
                print(f"  - {suggestion}")
        else:
            print("  No specific optimizations suggested at this time")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()