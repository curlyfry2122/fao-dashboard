#!/usr/bin/env python3
"""
Agent Orchestrator CLI Management Tool

This script provides a command-line interface for managing the FAO Dashboard
Agent Orchestration system, including starting/stopping agents, monitoring
status, and managing tasks.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.agent_orchestrator import AgentOrchestrator, AgentStatus, TaskStatus
from orchestration.priority_matrix import PriorityMatrix, Priority, AgentType
from orchestration.escalation_manager import EscalationManager
from orchestration.monitoring_dashboard import MonitoringDashboard


class OrchestratorCLI:
    """Command-line interface for the Agent Orchestrator."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.orchestrator = None
        self.priority_matrix = None
        self.escalation_manager = None
        self.monitoring_dashboard = None
        
        # CLI state
        self.verbose = False
        self.config_path = Path(__file__).parent / "orchestrator_config.json"
        
    def setup_orchestrator(self, config_path: Optional[Path] = None) -> None:
        """Set up orchestrator components."""
        if config_path:
            self.config_path = config_path
        
        try:
            self.orchestrator = AgentOrchestrator(self.config_path)
            self.priority_matrix = PriorityMatrix()
            self.escalation_manager = EscalationManager()
            self.monitoring_dashboard = MonitoringDashboard(self.orchestrator)
            
            if self.verbose:
                print("✅ Orchestrator components initialized successfully")
                
        except Exception as e:
            print(f"❌ Failed to initialize orchestrator: {e}")
            sys.exit(1)
    
    def cmd_start(self, args) -> None:
        """Start the orchestrator."""
        print("🚀 Starting FAO Dashboard Agent Orchestrator...")
        
        if args.daemon:
            print("Daemon mode not implemented yet. Starting in foreground.")
        
        try:
            self.orchestrator.start()
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal, stopping orchestrator...")
            self.orchestrator.stop()
        except Exception as e:
            print(f"❌ Orchestrator failed: {e}")
            sys.exit(1)
    
    def cmd_stop(self, args) -> None:
        """Stop the orchestrator."""
        print("🛑 Stopping orchestrator...")
        if self.orchestrator:
            self.orchestrator.stop()
            print("✅ Orchestrator stopped")
        else:
            print("ℹ️ No orchestrator instance found")
    
    def cmd_status(self, args) -> None:
        """Show orchestrator status."""
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        print("📊 FAO Dashboard Agent Orchestrator Status")
        print("=" * 50)
        
        # Overall status
        status_report = self.orchestrator.get_status_report()
        running = status_report['orchestrator']['running']
        uptime = status_report['orchestrator']['uptime']
        
        print(f"Status: {'🟢 Running' if running else '🔴 Stopped'}")
        print(f"Uptime: {uptime:.1f} seconds")
        
        # Queue status
        queue_status = self.orchestrator.get_queue_status()
        print(f"\n📋 Queue Status:")
        print(f"  Queued: {queue_status['total_queued']}")
        print(f"  Active: {queue_status['total_active']}")
        print(f"  Completed: {queue_status['total_completed']}")
        
        if queue_status['oldest_task_age'] > 0:
            print(f"  Oldest task: {queue_status['oldest_task_age']:.1f}s ago")
        
        # Agent status
        print(f"\n🤖 Agent Status:")
        for agent_type, agent_info in status_report['agents'].items():
            status_emoji = {
                'idle': '🟢',
                'busy': '🟡', 
                'error': '🔴',
                'offline': '⚫',
                'maintenance': '🟠'
            }.get(agent_info['status'], '❓')
            
            print(f"  {status_emoji} {agent_type}: {agent_info['status']} "
                  f"(Health: {agent_info['health_score']:.2f}, "
                  f"Load: {agent_info['current_load']}/{agent_info['max_concurrent_tasks']})")
        
        # Recent metrics
        if args.detailed:
            metrics = self.orchestrator.get_metrics()
            if metrics:
                print(f"\n📈 Metrics (last update: {metrics.get('timestamp', 'N/A')}):")
                task_metrics = metrics.get('tasks', {})
                print(f"  Success rate: {task_metrics.get('success_rate', 0):.1%}")
                print(f"  Avg completion time: {task_metrics.get('avg_completion_time_seconds', 0):.1f}s")
    
    def cmd_submit(self, args) -> None:
        """Submit a task to the orchestrator."""
        try:
            agent_type = AgentType(args.agent)
        except ValueError:
            print(f"❌ Invalid agent type: {args.agent}")
            print(f"Available agents: {[a.value for a in AgentType]}")
            return
        
        try:
            priority = Priority(args.priority)
        except ValueError:
            print(f"❌ Invalid priority: {args.priority}")
            print(f"Available priorities: {[p.value for p in Priority]}")
            return
        
        # Parse parameters
        parameters = {}
        if args.params:
            try:
                parameters = json.loads(args.params)
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON parameters: {args.params}")
                return
        
        # Submit task
        task_id = self.orchestrator.submit_task(
            agent_type=agent_type,
            title=args.title,
            description=args.description,
            priority=priority,
            parameters=parameters
        )
        
        print(f"✅ Task submitted: {task_id}")
        
        if args.wait:
            print("⏳ Waiting for task completion...")
            self._wait_for_task(task_id, timeout=args.timeout)
    
    def cmd_classify(self, args) -> None:
        """Classify an issue using the priority matrix."""
        classification = self.priority_matrix.classify_issue(
            title=args.title,
            body=args.body or "",
            labels=args.labels or []
        )
        
        print("🎯 Issue Classification Result")
        print("=" * 40)
        print(f"Priority: {classification.priority.value.upper()}")
        print(f"Agent: {classification.agent_type.value}")
        print(f"Escalate: {'Yes' if classification.escalate else 'No'}")
        print(f"Confidence: {classification.confidence:.2f}")
        print(f"Est. Resolution Time: {classification.estimated_resolution_time}")
        
        if classification.required_expertise:
            print(f"Required Expertise: {', '.join(classification.required_expertise)}")
        
        if classification.dependencies:
            print(f"Dependencies: {', '.join(classification.dependencies)}")
        
        print("\nReasoning:")
        for reason in classification.reasoning:
            print(f"  • {reason}")
        
        if args.submit:
            task_id = self.orchestrator.submit_issue_task(
                title=args.title,
                body=args.body or "",
                labels=args.labels or []
            )
            print(f"\n✅ Task auto-submitted: {task_id}")
    
    def cmd_tasks(self, args) -> None:
        """List and manage tasks."""
        if args.task_id:
            # Show specific task
            task = self.orchestrator.get_task_status(args.task_id)
            if task:
                print(f"📋 Task Details: {args.task_id}")
                print("=" * 40)
                print(f"Title: {task.title}")
                print(f"Agent: {task.agent_type.value}")
                print(f"Priority: {task.priority.value}")
                print(f"Status: {task.status.value}")
                print(f"Created: {task.created_at}")
                
                if task.started_at:
                    print(f"Started: {task.started_at}")
                if task.completed_at:
                    print(f"Completed: {task.completed_at}")
                    duration = (task.completed_at - task.started_at).total_seconds()
                    print(f"Duration: {duration:.1f}s")
                
                if task.error:
                    print(f"Error: {task.error}")
                
                if task.result:
                    print(f"Result: {json.dumps(task.result, indent=2)}")
            else:
                print(f"❌ Task not found: {args.task_id}")
        else:
            # List tasks
            queue_status = self.orchestrator.get_queue_status()
            print(f"📋 Task Overview")
            print("=" * 40)
            print(f"Queued: {queue_status['total_queued']}")
            print(f"Active: {queue_status['total_active']}")
            print(f"Completed: {queue_status['total_completed']}")
            
            # Show recent completed tasks
            completed_tasks = self.orchestrator.completed_tasks[-10:]
            if completed_tasks:
                print(f"\n🏁 Recent Completed Tasks:")
                for task in completed_tasks:
                    status_emoji = '✅' if task.status == TaskStatus.COMPLETED else '❌'
                    print(f"  {status_emoji} {task.id}: {task.title} ({task.agent_type.value})")
    
    def cmd_agents(self, args) -> None:
        """Manage agents."""
        if args.agent_type:
            # Show specific agent
            try:
                agent_type = AgentType(args.agent_type)
                agent = self.orchestrator.get_agent_status(agent_type)
                
                if agent:
                    print(f"🤖 Agent Details: {args.agent_type}")
                    print("=" * 40)
                    print(f"Status: {agent.status.value}")
                    print(f"Health Score: {agent.health_score:.2f}")
                    print(f"Current Load: {agent.current_load}/{agent.max_concurrent_tasks}")
                    print(f"Tasks Completed: {agent.tasks_completed}")
                    print(f"Tasks Failed: {agent.tasks_failed}")
                    
                    if agent.tasks_completed > 0:
                        success_rate = agent.tasks_completed / (agent.tasks_completed + agent.tasks_failed)
                        print(f"Success Rate: {success_rate:.1%}")
                    
                    if agent.last_active:
                        print(f"Last Active: {agent.last_active}")
                    
                    print(f"Capabilities: {', '.join(agent.capabilities)}")
                else:
                    print(f"❌ Agent not found: {args.agent_type}")
            except ValueError:
                print(f"❌ Invalid agent type: {args.agent_type}")
                print(f"Available agents: {[a.value for a in AgentType]}")
        else:
            # List all agents
            print("🤖 Agent Overview")
            print("=" * 40)
            
            for agent_type, agent in self.orchestrator.agents.items():
                status_indicator = {
                    AgentStatus.IDLE: '🟢',
                    AgentStatus.BUSY: '🟡',
                    AgentStatus.ERROR: '🔴',
                    AgentStatus.OFFLINE: '⚫',
                    AgentStatus.MAINTENANCE: '🟠'
                }.get(agent.status, '❓')
                
                utilization = agent.current_load / agent.max_concurrent_tasks * 100
                print(f"{status_indicator} {agent_type.value}: {agent.status.value} "
                      f"(Load: {utilization:.0f}%, Health: {agent.health_score:.2f})")
    
    def cmd_monitor(self, args) -> None:
        """Real-time monitoring."""
        print("📊 FAO Dashboard Agent Monitoring")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        try:
            while True:
                # Clear screen (basic)
                print("\033[2J\033[H", end="")
                
                # Generate monitoring snapshot
                snapshot = self.monitoring_dashboard.generate_status_report()
                
                print(f"📊 Status Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 60)
                
                # System health
                health = snapshot.get('health', {})
                overall_status = health.get('overall_status', 'unknown')
                status_emoji = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴'}.get(overall_status, '❓')
                print(f"Overall Health: {status_emoji} {overall_status.upper()}")
                
                # Queue metrics
                queue = snapshot.get('queue', {})
                print(f"Queue: {queue.get('total_queued', 0)} pending, {queue.get('total_active', 0)} active")
                
                # Task metrics
                tasks = snapshot.get('tasks', {})
                print(f"Tasks: {tasks.get('success_rate', 0):.1%} success rate, "
                      f"{tasks.get('avg_completion_time', 0):.0f}s avg time")
                
                # Agent health summary
                agents = snapshot.get('agents', {})
                healthy_agents = sum(1 for a in agents.values() if a.get('health_score', 0) > 0.7)
                print(f"Agents: {healthy_agents}/{len(agents)} healthy")
                
                # Alerts
                alerts = snapshot.get('alerts', [])
                if alerts:
                    print(f"\n🚨 Active Alerts ({len(alerts)}):")
                    for alert in alerts[:3]:  # Show top 3 alerts
                        level_emoji = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}.get(alert['level'], '❓')
                        print(f"  {level_emoji} {alert['message']}")
                
                # Recommendations
                recommendations = snapshot.get('recommendations', [])
                if recommendations:
                    print(f"\n💡 Recommendations:")
                    for rec in recommendations[:2]:  # Show top 2 recommendations
                        print(f"  • {rec['title']} ({rec['priority']})")
                
                print(f"\nNext update in {args.interval}s...")
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
    
    def cmd_report(self, args) -> None:
        """Generate reports."""
        if args.format == 'html':
            report_path = self.monitoring_dashboard.generate_html_report()
            print(f"📄 HTML report generated: {report_path}")
        elif args.format == 'csv':
            report_path = self.monitoring_dashboard.export_metrics_to_csv(hours=args.hours)
            print(f"📊 CSV report generated: {report_path}")
        elif args.format == 'json':
            status_report = self.monitoring_dashboard.generate_status_report()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = Path(__file__).parent.parent / "automation" / "reports" / f"status_report_{timestamp}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(status_report, f, indent=2, default=str)
            
            print(f"📋 JSON report generated: {report_path}")
        
        if args.open_report:
            import webbrowser
            webbrowser.open(f"file://{report_path.absolute()}")
    
    def _wait_for_task(self, task_id: str, timeout: int = 300) -> None:
        """Wait for a task to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.orchestrator.get_task_status(task_id)
            if not task:
                print(f"❌ Task not found: {task_id}")
                return
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task.status == TaskStatus.COMPLETED:
                    print(f"✅ Task completed successfully")
                    if task.result:
                        print(f"Result: {json.dumps(task.result, indent=2)}")
                else:
                    print(f"❌ Task {task.status.value}")
                    if task.error:
                        print(f"Error: {task.error}")
                return
            
            print(f"⏳ Task status: {task.status.value}")
            time.sleep(5)
        
        print(f"⏰ Timeout waiting for task completion")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FAO Dashboard Agent Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    # Start orchestrator
  %(prog)s status --detailed        # Show detailed status
  %(prog)s submit data-pipeline "Fetch FAO data" "Get latest data"
  %(prog)s classify "Dashboard broken" --submit
  %(prog)s monitor --interval 10    # Real-time monitoring
  %(prog)s report --format html --open
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--config', type=Path, help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the orchestrator')
    start_parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    # Stop command
    subparsers.add_parser('stop', help='Stop the orchestrator')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show orchestrator status')
    status_parser.add_argument('--detailed', action='store_true', help='Show detailed metrics')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a task')
    submit_parser.add_argument('agent', help='Agent type')
    submit_parser.add_argument('title', help='Task title')
    submit_parser.add_argument('description', help='Task description')
    submit_parser.add_argument('--priority', default='medium', help='Task priority')
    submit_parser.add_argument('--params', help='JSON parameters')
    submit_parser.add_argument('--wait', action='store_true', help='Wait for completion')
    submit_parser.add_argument('--timeout', type=int, default=300, help='Wait timeout')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify an issue')
    classify_parser.add_argument('title', help='Issue title')
    classify_parser.add_argument('--body', help='Issue body')
    classify_parser.add_argument('--labels', nargs='*', help='Issue labels')
    classify_parser.add_argument('--submit', action='store_true', help='Auto-submit classified issue')
    
    # Tasks command
    tasks_parser = subparsers.add_parser('tasks', help='List and manage tasks')
    tasks_parser.add_argument('--task-id', help='Show specific task')
    
    # Agents command
    agents_parser = subparsers.add_parser('agents', help='Manage agents')
    agents_parser.add_argument('--agent-type', help='Show specific agent')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Real-time monitoring')
    monitor_parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--format', choices=['html', 'csv', 'json'], default='html', help='Report format')
    report_parser.add_argument('--hours', type=int, default=24, help='Report time range in hours')
    report_parser.add_argument('--open-report', action='store_true', help='Open report after generation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = OrchestratorCLI()
    cli.verbose = args.verbose
    
    # Setup orchestrator (except for some commands that don't need it)
    if args.command not in ['help']:
        cli.setup_orchestrator(args.config)
    
    # Execute command
    command_handler = getattr(cli, f'cmd_{args.command.replace("-", "_")}', None)
    if command_handler:
        try:
            command_handler(args)
        except KeyboardInterrupt:
            print("\n👋 Operation cancelled")
        except Exception as e:
            print(f"❌ Command failed: {e}")
            if cli.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()