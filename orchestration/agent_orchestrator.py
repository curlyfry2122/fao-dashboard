"""
Agent Orchestrator

Central coordination system for managing multiple agents across the FAO Dashboard project.
Handles agent lifecycle, task distribution, escalation, and monitoring.
"""

import asyncio
import json
import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from threading import Lock, Event

from priority_matrix import PriorityMatrix, Priority, AgentType, IssueClassification


class AgentStatus(Enum):
    """Agent status states."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"


@dataclass
class Task:
    """Represents a task to be executed by an agent."""
    id: str
    agent_type: AgentType
    priority: Priority
    title: str
    description: str
    parameters: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    escalated: bool = False
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class Agent:
    """Represents an agent and its current state."""
    agent_type: AgentType
    status: AgentStatus
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_active: Optional[datetime] = None
    capabilities: List[str] = None
    max_concurrent_tasks: int = 1
    current_load: int = 0
    health_score: float = 1.0
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class AgentOrchestrator:
    """
    Central orchestrator for managing multiple agents and coordinating their work.
    
    Features:
    - Task queuing and distribution based on priority and agent capabilities
    - Agent health monitoring and load balancing
    - Automatic escalation and retry logic
    - Real-time status reporting and metrics
    - Graceful shutdown and recovery
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the orchestrator."""
        self.config_path = config_path or Path(__file__).parent / "orchestrator_config.json"
        self.priority_matrix = PriorityMatrix()
        
        # Core state
        self.agents: Dict[AgentType, Agent] = {}
        self.task_queue: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.metrics: Dict[str, Any] = {}
        
        # Threading and coordination
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.shutdown_event = Event()
        self.state_lock = Lock()
        self.running = False
        
        # Configuration
        self.load_configuration()
        self.setup_logging()
        self.initialize_agents()
        
        # Metrics tracking
        self.metrics_start_time = datetime.now()
        self.last_metrics_update = datetime.now()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def load_configuration(self) -> None:
        """Load orchestrator configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.get_default_configuration()
            self.save_configuration()
    
    def get_default_configuration(self) -> Dict:
        """Get default orchestrator configuration."""
        return {
            "orchestrator": {
                "max_queue_size": 100,
                "task_timeout_seconds": 3600,
                "health_check_interval": 60,
                "metrics_update_interval": 300,
                "max_concurrent_tasks": 20,
                "escalation_enabled": True,
                "auto_retry_enabled": True
            },
            "agents": {
                "data-pipeline": {
                    "enabled": True,
                    "max_concurrent_tasks": 3,
                    "command": "python3 -m agents.data_pipeline_agent",
                    "timeout": 1800,
                    "health_check_endpoint": "/health",
                    "capabilities": ["data_fetching", "cache_management", "validation"]
                },
                "ui-dashboard": {
                    "enabled": True,
                    "max_concurrent_tasks": 2,
                    "command": "python3 -m agents.ui_dashboard_agent",
                    "timeout": 1200,
                    "capabilities": ["ui_testing", "frontend_validation", "user_experience"]
                },
                "deployment": {
                    "enabled": True,
                    "max_concurrent_tasks": 1,
                    "command": "python3 -m agents.deployment_agent",
                    "timeout": 900,
                    "capabilities": ["docker_build", "deployment", "infrastructure"]
                },
                "performance-monitor": {
                    "enabled": True,
                    "max_concurrent_tasks": 2,
                    "command": "python3 -m agents.performance_agent",
                    "timeout": 2400,
                    "capabilities": ["performance_testing", "benchmarking", "optimization"]
                },
                "security-scan": {
                    "enabled": True,
                    "max_concurrent_tasks": 1,
                    "command": "python3 -m agents.security_agent",
                    "timeout": 600,
                    "capabilities": ["vulnerability_scanning", "security_analysis", "compliance"]
                },
                "dependency-update": {
                    "enabled": True,
                    "max_concurrent_tasks": 3,
                    "command": "python3 -m agents.dependency_agent",
                    "timeout": 300,
                    "capabilities": ["dependency_checking", "version_management", "compatibility"]
                }
            },
            "escalation": {
                "critical_immediate": True,
                "high_after_minutes": 30,
                "medium_after_hours": 4,
                "low_after_hours": 24,
                "max_escalation_attempts": 3,
                "escalation_contacts": []
            },
            "notifications": {
                "enabled": True,
                "channels": ["github_issues", "logs"],
                "critical_immediate": True,
                "summary_interval_hours": 6
            }
        }
    
    def save_configuration(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def setup_logging(self) -> None:
        """Configure logging for the orchestrator."""
        log_dir = Path(__file__).parent.parent / "automation" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "orchestrator.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Agent Orchestrator initialized")
    
    def initialize_agents(self) -> None:
        """Initialize all configured agents."""
        for agent_type_str, agent_config in self.config.get("agents", {}).items():
            if not agent_config.get("enabled", True):
                continue
            
            try:
                agent_type = AgentType(agent_type_str)
                agent = Agent(
                    agent_type=agent_type,
                    status=AgentStatus.IDLE,
                    capabilities=agent_config.get("capabilities", []),
                    max_concurrent_tasks=agent_config.get("max_concurrent_tasks", 1)
                )
                self.agents[agent_type] = agent
                self.logger.info(f"Initialized agent: {agent_type.value}")
                
            except ValueError as e:
                self.logger.warning(f"Unknown agent type: {agent_type_str}")
    
    def submit_task(
        self,
        agent_type: AgentType,
        title: str,
        description: str,
        priority: Priority = Priority.MEDIUM,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Submit a new task for execution.
        
        Args:
            agent_type: Type of agent to handle the task
            title: Task title
            description: Task description
            priority: Task priority level
            parameters: Additional parameters for the task
            dependencies: List of task IDs this task depends on
            
        Returns:
            Task ID
        """
        if parameters is None:
            parameters = {}
        if dependencies is None:
            dependencies = []
        
        task_id = f"{agent_type.value}_{int(time.time() * 1000)}"
        
        task = Task(
            id=task_id,
            agent_type=agent_type,
            priority=priority,
            title=title,
            description=description,
            parameters=parameters,
            created_at=datetime.now(),
            dependencies=dependencies
        )
        
        with self.state_lock:
            # Check queue size limits
            max_queue_size = self.config.get("orchestrator", {}).get("max_queue_size", 100)
            if len(self.task_queue) >= max_queue_size:
                self.logger.warning(f"Task queue full ({max_queue_size}), rejecting task: {task_id}")
                raise RuntimeError("Task queue is full")
            
            self.task_queue.append(task)
            self.logger.info(f"Task submitted: {task_id} ({priority.value}) - {title}")
        
        return task_id
    
    def submit_issue_task(self, title: str, body: str = "", labels: Optional[List[str]] = None) -> str:
        """
        Submit a task based on issue classification.
        
        Args:
            title: Issue title
            body: Issue body/description
            labels: Issue labels
            
        Returns:
            Task ID
        """
        classification = self.priority_matrix.classify_issue(title, body, labels)
        
        return self.submit_task(
            agent_type=classification.agent_type,
            title=title,
            description=body,
            priority=classification.priority,
            parameters={
                "classification": asdict(classification),
                "labels": labels or [],
                "auto_escalate": classification.escalate
            }
        )
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get current status of a task."""
        with self.state_lock:
            # Check active tasks
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            
            # Check queue
            for task in self.task_queue:
                if task.id == task_id:
                    return task
            
            # Check completed tasks
            for task in self.completed_tasks:
                if task.id == task_id:
                    return task
        
        return None
    
    def get_agent_status(self, agent_type: AgentType) -> Optional[Agent]:
        """Get current status of an agent."""
        return self.agents.get(agent_type)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and metrics."""
        with self.state_lock:
            # Count tasks by priority
            priority_counts = {p.value: 0 for p in Priority}
            for task in self.task_queue:
                priority_counts[task.priority.value] += 1
            
            # Count tasks by agent type
            agent_counts = {}
            for task in self.task_queue:
                agent_type = task.agent_type.value
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
            
            return {
                "total_queued": len(self.task_queue),
                "total_active": len(self.active_tasks),
                "total_completed": len(self.completed_tasks),
                "priority_distribution": priority_counts,
                "agent_distribution": agent_counts,
                "oldest_task_age": (
                    (datetime.now() - self.task_queue[0].created_at).total_seconds()
                    if self.task_queue else 0
                )
            }
    
    def start(self) -> None:
        """Start the orchestrator main loop."""
        if self.running:
            self.logger.warning("Orchestrator is already running")
            return
        
        self.running = True
        self.logger.info("Starting Agent Orchestrator")
        
        try:
            # Start background tasks
            health_check_future = self.executor.submit(self._health_check_loop)
            metrics_future = self.executor.submit(self._metrics_loop)
            task_processor_future = self.executor.submit(self._task_processor_loop)
            
            # Wait for shutdown signal
            while self.running and not self.shutdown_event.is_set():
                time.sleep(1)
            
            self.logger.info("Shutting down orchestrator...")
            
            # Cancel background tasks
            health_check_future.cancel()
            metrics_future.cancel()
            task_processor_future.cancel()
            
            # Wait for active tasks to complete (with timeout)
            self._wait_for_active_tasks(timeout=30)
            
        except Exception as e:
            self.logger.error(f"Orchestrator error: {e}")
        finally:
            self.running = False
            self.executor.shutdown(wait=True)
            self.logger.info("Agent Orchestrator stopped")
    
    def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        self.logger.info("Stopping Agent Orchestrator...")
        self.running = False
        self.shutdown_event.set()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()
    
    def _task_processor_loop(self) -> None:
        """Main task processing loop."""
        while self.running and not self.shutdown_event.is_set():
            try:
                self._process_pending_tasks()
                self._check_task_timeouts()
                self._handle_escalations()
                time.sleep(5)  # Process every 5 seconds
            except Exception as e:
                self.logger.error(f"Task processor error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _process_pending_tasks(self) -> None:
        """Process pending tasks in the queue."""
        with self.state_lock:
            if not self.task_queue:
                return
            
            # Sort by priority and creation time
            self.task_queue.sort(key=lambda t: (t.priority.value, t.created_at))
            
            # Try to assign tasks to available agents
            tasks_to_process = []
            for i, task in enumerate(self.task_queue):
                agent = self.agents.get(task.agent_type)
                if not agent or agent.status != AgentStatus.IDLE:
                    continue
                
                if agent.current_load >= agent.max_concurrent_tasks:
                    continue
                
                # Check dependencies
                if self._check_task_dependencies(task):
                    tasks_to_process.append((i, task))
                    agent.current_load += 1
                    
                    # Don't overload - process one task per iteration per agent
                    break
            
            # Remove assigned tasks from queue and start execution
            for i, task in reversed(tasks_to_process):  # Reverse to maintain indices
                del self.task_queue[i]
                self.active_tasks[task.id] = task
                self._start_task_execution(task)
    
    def _check_task_dependencies(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            # Check if dependency is completed
            found_completed = any(
                t.id == dep_id and t.status == TaskStatus.COMPLETED
                for t in self.completed_tasks
            )
            if not found_completed:
                return False
        
        return True
    
    def _start_task_execution(self, task: Task) -> None:
        """Start execution of a task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        agent = self.agents[task.agent_type]
        agent.status = AgentStatus.BUSY
        agent.current_task = task.id
        agent.last_active = datetime.now()
        
        self.logger.info(f"Starting task execution: {task.id}")
        
        # Submit task execution to thread pool
        future = self.executor.submit(self._execute_task, task)
        future.add_done_callback(lambda f: self._task_completion_callback(task.id, f))
    
    def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task (mock implementation)."""
        # This is a mock implementation
        # In a real system, this would invoke the actual agent
        
        agent_config = self.config.get("agents", {}).get(task.agent_type.value, {})
        timeout = agent_config.get("timeout", 300)
        
        self.logger.info(f"Executing task {task.id} with agent {task.agent_type.value}")
        
        try:
            # Simulate task execution
            execution_time = min(10, timeout / 60)  # Mock: 10 seconds or 1/60 of timeout
            time.sleep(execution_time)
            
            # Mock result
            result = {
                "status": "success",
                "message": f"Task {task.id} completed successfully",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Task {task.id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task.id} failed: {e}")
            raise
    
    def _task_completion_callback(self, task_id: str, future) -> None:
        """Handle task completion."""
        with self.state_lock:
            task = self.active_tasks.get(task_id)
            if not task:
                return
            
            agent = self.agents.get(task.agent_type)
            if agent:
                agent.status = AgentStatus.IDLE
                agent.current_task = None
                agent.current_load = max(0, agent.current_load - 1)
                agent.last_active = datetime.now()
            
            try:
                result = future.result()
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.now()
                
                if agent:
                    agent.tasks_completed += 1
                
                self.logger.info(f"Task completed: {task_id}")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                
                if agent:
                    agent.tasks_failed += 1
                
                self.logger.error(f"Task failed: {task_id} - {e}")
                
                # Handle retry logic
                self._handle_task_retry(task)
            
            # Move from active to completed
            del self.active_tasks[task_id]
            self.completed_tasks.append(task)
            
            # Keep only recent completed tasks
            if len(self.completed_tasks) > 1000:
                self.completed_tasks = self.completed_tasks[-1000:]
    
    def _handle_task_retry(self, task: Task) -> None:
        """Handle task retry logic."""
        if not self.config.get("orchestrator", {}).get("auto_retry_enabled", True):
            return
        
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            task.started_at = None
            task.error = None
            
            # Add back to queue with delay
            self.task_queue.append(task)
            self.logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
        else:
            self.logger.warning(f"Task {task.id} exceeded max retries, marking for escalation")
            task.escalated = True
            self._escalate_task(task)
    
    def _check_task_timeouts(self) -> None:
        """Check for timed out tasks."""
        timeout_threshold = self.config.get("orchestrator", {}).get("task_timeout_seconds", 3600)
        current_time = datetime.now()
        
        with self.state_lock:
            timed_out_tasks = []
            for task_id, task in self.active_tasks.items():
                if task.started_at:
                    elapsed = (current_time - task.started_at).total_seconds()
                    if elapsed > timeout_threshold:
                        timed_out_tasks.append(task_id)
            
            for task_id in timed_out_tasks:
                task = self.active_tasks[task_id]
                self.logger.warning(f"Task timeout: {task_id}")
                task.status = TaskStatus.FAILED
                task.error = "Task timeout"
                self._handle_task_retry(task)
    
    def _handle_escalations(self) -> None:
        """Handle task escalations based on configured rules."""
        if not self.config.get("orchestrator", {}).get("escalation_enabled", True):
            return
        
        escalation_config = self.config.get("escalation", {})
        current_time = datetime.now()
        
        with self.state_lock:
            tasks_to_escalate = []
            
            # Check queue for tasks that need escalation
            for task in self.task_queue:
                should_escalate = False
                age_minutes = (current_time - task.created_at).total_seconds() / 60
                
                if task.priority == Priority.CRITICAL and escalation_config.get("critical_immediate", True):
                    should_escalate = True
                elif (task.priority == Priority.HIGH and 
                      age_minutes > escalation_config.get("high_after_minutes", 30)):
                    should_escalate = True
                elif (task.priority == Priority.MEDIUM and 
                      age_minutes > escalation_config.get("medium_after_hours", 4) * 60):
                    should_escalate = True
                elif (task.priority == Priority.LOW and 
                      age_minutes > escalation_config.get("low_after_hours", 24) * 60):
                    should_escalate = True
                
                if should_escalate and not task.escalated:
                    tasks_to_escalate.append(task)
            
            for task in tasks_to_escalate:
                self._escalate_task(task)
    
    def _escalate_task(self, task: Task) -> None:
        """Escalate a task."""
        task.escalated = True
        task.status = TaskStatus.ESCALATED
        
        self.logger.warning(f"Escalating task: {task.id} - {task.title}")
        
        # TODO: Implement actual escalation logic (notifications, etc.)
        # For now, just log the escalation
        
        escalation_data = {
            "task_id": task.id,
            "task_title": task.title,
            "priority": task.priority.value,
            "agent_type": task.agent_type.value,
            "age_minutes": (datetime.now() - task.created_at).total_seconds() / 60,
            "retry_count": task.retry_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save escalation record
        escalation_file = Path(__file__).parent.parent / "automation" / "logs" / "escalations.jsonl"
        escalation_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(escalation_file, 'a') as f:
            f.write(json.dumps(escalation_data) + '\n')
    
    def _health_check_loop(self) -> None:
        """Background health check loop."""
        interval = self.config.get("orchestrator", {}).get("health_check_interval", 60)
        
        while self.running and not self.shutdown_event.is_set():
            try:
                self._perform_health_checks()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                time.sleep(interval)
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all agents."""
        for agent_type, agent in self.agents.items():
            try:
                # Simple health check based on last activity
                if agent.last_active:
                    inactive_minutes = (datetime.now() - agent.last_active).total_seconds() / 60
                    if inactive_minutes > 60:  # 1 hour inactive
                        agent.health_score = max(0.1, agent.health_score - 0.1)
                    else:
                        agent.health_score = min(1.0, agent.health_score + 0.1)
                
                # Check if agent has too many failures
                if agent.tasks_completed > 0:
                    failure_rate = agent.tasks_failed / (agent.tasks_completed + agent.tasks_failed)
                    if failure_rate > 0.5:  # More than 50% failure rate
                        agent.health_score = max(0.1, agent.health_score - 0.2)
                
                # Update agent status based on health
                if agent.health_score < 0.3:
                    agent.status = AgentStatus.ERROR
                elif agent.status == AgentStatus.ERROR and agent.health_score > 0.7:
                    agent.status = AgentStatus.IDLE
                
            except Exception as e:
                self.logger.error(f"Health check failed for {agent_type.value}: {e}")
                agent.status = AgentStatus.ERROR
    
    def _metrics_loop(self) -> None:
        """Background metrics collection loop."""
        interval = self.config.get("orchestrator", {}).get("metrics_update_interval", 300)
        
        while self.running and not self.shutdown_event.is_set():
            try:
                self._update_metrics()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Metrics update error: {e}")
                time.sleep(interval)
    
    def _update_metrics(self) -> None:
        """Update orchestrator metrics."""
        current_time = datetime.now()
        uptime = (current_time - self.metrics_start_time).total_seconds()
        
        with self.state_lock:
            # Calculate task metrics
            total_tasks = len(self.completed_tasks) + len(self.active_tasks) + len(self.task_queue)
            completed_tasks = len(self.completed_tasks)
            failed_tasks = sum(1 for t in self.completed_tasks if t.status == TaskStatus.FAILED)
            
            # Calculate average completion time
            completed_with_times = [
                t for t in self.completed_tasks 
                if t.completed_at and t.started_at
            ]
            avg_completion_time = 0
            if completed_with_times:
                total_time = sum(
                    (t.completed_at - t.started_at).total_seconds() 
                    for t in completed_with_times
                )
                avg_completion_time = total_time / len(completed_with_times)
            
            # Agent metrics
            agent_metrics = {}
            for agent_type, agent in self.agents.items():
                agent_metrics[agent_type.value] = {
                    "status": agent.status.value,
                    "health_score": agent.health_score,
                    "tasks_completed": agent.tasks_completed,
                    "tasks_failed": agent.tasks_failed,
                    "current_load": agent.current_load,
                    "max_load": agent.max_concurrent_tasks,
                    "utilization": agent.current_load / agent.max_concurrent_tasks
                }
            
            self.metrics = {
                "timestamp": current_time.isoformat(),
                "uptime_seconds": uptime,
                "orchestrator_status": "running" if self.running else "stopped",
                "tasks": {
                    "total": total_tasks,
                    "completed": completed_tasks,
                    "failed": failed_tasks,
                    "active": len(self.active_tasks),
                    "queued": len(self.task_queue),
                    "success_rate": completed_tasks / max(1, total_tasks) * 100,
                    "avg_completion_time_seconds": avg_completion_time
                },
                "agents": agent_metrics,
                "queue_status": self.get_queue_status()
            }
        
        # Save metrics to file
        metrics_file = Path(__file__).parent.parent / "automation" / "logs" / "orchestrator_metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
        
        self.last_metrics_update = current_time
    
    def _wait_for_active_tasks(self, timeout: int = 30) -> None:
        """Wait for active tasks to complete during shutdown."""
        start_time = time.time()
        
        while self.active_tasks and (time.time() - start_time) < timeout:
            self.logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            time.sleep(1)
        
        if self.active_tasks:
            self.logger.warning(f"Shutdown timeout reached, {len(self.active_tasks)} tasks still active")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current orchestrator metrics."""
        return self.metrics.copy()
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        return {
            "orchestrator": {
                "running": self.running,
                "uptime": (datetime.now() - self.metrics_start_time).total_seconds(),
                "last_metrics_update": self.last_metrics_update.isoformat()
            },
            "agents": {
                agent_type.value: asdict(agent) 
                for agent_type, agent in self.agents.items()
            },
            "queue": self.get_queue_status(),
            "metrics": self.metrics
        }


def main():
    """CLI interface for the orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FAO Dashboard Agent Orchestrator")
    parser.add_argument('--start', action='store_true', help='Start the orchestrator')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--submit', nargs=3, metavar=('AGENT', 'TITLE', 'DESCRIPTION'),
                       help='Submit a task')
    parser.add_argument('--task-status', metavar='TASK_ID', help='Get task status')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    orchestrator = AgentOrchestrator(config_path)
    
    if args.start:
        orchestrator.start()
    elif args.status:
        status = orchestrator.get_status_report()
        print(json.dumps(status, indent=2, default=str))
    elif args.submit:
        agent_type_str, title, description = args.submit
        try:
            agent_type = AgentType(agent_type_str)
            task_id = orchestrator.submit_task(agent_type, title, description)
            print(f"Task submitted: {task_id}")
        except ValueError:
            print(f"Unknown agent type: {agent_type_str}")
            print(f"Available: {[a.value for a in AgentType]}")
    elif args.task_status:
        task = orchestrator.get_task_status(args.task_status)
        if task:
            print(json.dumps(asdict(task), indent=2, default=str))
        else:
            print(f"Task not found: {args.task_status}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()