"""
Task Master AI MCP tool adapter for LangGraph agents.
Provides task management, research capabilities, and project coordination.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    REVIEW = "review"
    DONE = "done"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Task:
    """Task representation"""
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    assignee: Optional[str] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None


@dataclass
class Project:
    """Project representation"""
    id: str
    name: str
    description: str
    tasks: List[Task] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "active"


class TaskMasterAdapter:
    """Adapter for Task Master AI MCP tool"""

    def __init__(self, tool_executor=None):
        self.tool_executor = tool_executor
        self._projects = {}
        self._tasks = {}
        self._task_counter = 1

    async def initialize_project(
        self,
        project_name: str,
        description: str = "",
        rules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Initialize a new Task Master project"""
        try:
            project_id = f"proj_{len(self._projects) + 1}"
            
            project = Project(
                id=project_id,
                name=project_name,
                description=description,
                tags=rules or ["cursor"]
            )
            
            self._projects[project_id] = project
            
            logger.info(f"Initialized project: {project_name}")
            return {
                "success": True,
                "project_id": project_id,
                "project_name": project_name,
                "message": f"Project '{project_name}' initialized successfully"
            }

        except Exception as e:
            logger.error(f"Failed to initialize project: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_id": None
            }

    async def create_task(
        self,
        title: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[List[str]] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new task"""
        try:
            task_id = str(self._task_counter)
            self._task_counter += 1

            task = Task(
                id=task_id,
                title=title,
                description=description,
                priority=priority,
                dependencies=dependencies or []
            )

            self._tasks[task_id] = task

            # Add to project if specified
            if project_id and project_id in self._projects:
                self._projects[project_id].tasks.append(task)

            logger.debug(f"Created task: {title}")
            return {
                "success": True,
                "task_id": task_id,
                "title": title,
                "status": task.status.value,
                "priority": task.priority.value
            }

        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": None
            }

    async def get_tasks(
        self,
        status: Optional[TaskStatus] = None,
        project_id: Optional[str] = None,
        with_subtasks: bool = False
    ) -> Dict[str, Any]:
        """Get tasks with optional filtering"""
        try:
            tasks = list(self._tasks.values())

            # Filter by status
            if status:
                tasks = [t for t in tasks if t.status == status]

            # Filter by project
            if project_id and project_id in self._projects:
                project_task_ids = [t.id for t in self._projects[project_id].tasks]
                tasks = [t for t in tasks if t.id in project_task_ids]

            # Format task data
            task_data = []
            for task in tasks:
                task_info = {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "dependencies": task.dependencies,
                    "tags": task.tags,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat()
                }

                if with_subtasks and task.subtasks:
                    task_info["subtasks"] = [
                        {
                            "id": st.id,
                            "title": st.title,
                            "status": st.status.value
                        }
                        for st in task.subtasks
                    ]

                task_data.append(task_info)

            return {
                "success": True,
                "tasks": task_data,
                "total_count": len(task_data)
            }

        except Exception as e:
            logger.error(f"Failed to get tasks: {e}")
            return {
                "success": False,
                "error": str(e),
                "tasks": []
            }

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus
    ) -> Dict[str, Any]:
        """Update task status"""
        try:
            if task_id not in self._tasks:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }

            task = self._tasks[task_id]
            old_status = task.status
            task.status = status
            task.updated_at = datetime.utcnow()

            logger.debug(f"Updated task {task_id} status: {old_status.value} -> {status.value}")
            return {
                "success": True,
                "task_id": task_id,
                "old_status": old_status.value,
                "new_status": status.value,
                "updated_at": task.updated_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def expand_task(
        self,
        task_id: str,
        num_subtasks: int = 3,
        research: bool = False
    ) -> Dict[str, Any]:
        """Expand task into subtasks"""
        try:
            if task_id not in self._tasks:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }

            task = self._tasks[task_id]

            # Generate subtasks based on task description
            subtasks = []
            for i in range(num_subtasks):
                subtask_id = f"{task_id}.{i + 1}"
                subtask = Task(
                    id=subtask_id,
                    title=f"Subtask {i + 1} of {task.title}",
                    description=f"Implementation step {i + 1} for: {task.description}",
                    priority=task.priority
                )
                subtasks.append(subtask)
                self._tasks[subtask_id] = subtask

            task.subtasks = subtasks
            task.updated_at = datetime.utcnow()

            logger.debug(f"Expanded task {task_id} into {num_subtasks} subtasks")
            return {
                "success": True,
                "task_id": task_id,
                "subtasks_created": num_subtasks,
                "subtask_ids": [st.id for st in subtasks]
            }

        except Exception as e:
            logger.error(f"Failed to expand task: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_next_task(
        self,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get the next task to work on based on dependencies and priority"""
        try:
            tasks = list(self._tasks.values())

            # Filter by project if specified
            if project_id and project_id in self._projects:
                project_task_ids = [t.id for t in self._projects[project_id].tasks]
                tasks = [t for t in tasks if t.id in project_task_ids]

            # Filter for pending tasks
            pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]

            if not pending_tasks:
                return {
                    "success": True,
                    "message": "No pending tasks found",
                    "next_task": None
                }

            # Find tasks with no pending dependencies
            available_tasks = []
            for task in pending_tasks:
                deps_satisfied = True
                for dep_id in task.dependencies:
                    if dep_id in self._tasks:
                        dep_task = self._tasks[dep_id]
                        if dep_task.status not in [TaskStatus.DONE, TaskStatus.CANCELLED]:
                            deps_satisfied = False
                            break

                if deps_satisfied:
                    available_tasks.append(task)

            if not available_tasks:
                return {
                    "success": True,
                    "message": "No tasks available (waiting on dependencies)",
                    "next_task": None
                }

            # Sort by priority (high -> medium -> low)
            priority_order = {TaskPriority.HIGH: 3, TaskPriority.MEDIUM: 2, TaskPriority.LOW: 1}
            available_tasks.sort(key=lambda t: priority_order[t.priority], reverse=True)

            next_task = available_tasks[0]

            return {
                "success": True,
                "next_task": {
                    "id": next_task.id,
                    "title": next_task.title,
                    "description": next_task.description,
                    "priority": next_task.priority.value,
                    "dependencies": next_task.dependencies
                }
            }

        except Exception as e:
            logger.error(f"Failed to get next task: {e}")
            return {
                "success": False,
                "error": str(e),
                "next_task": None
            }

    async def research_topic(
        self,
        topic: str,
        context: Optional[str] = None,
        depth: str = "medium"
    ) -> Dict[str, Any]:
        """Perform research on a topic"""
        try:
            # Simulated research results
            research_results = {
                "topic": topic,
                "context": context,
                "depth": depth,
                "findings": [
                    f"Key insight about {topic}: Implementation patterns and best practices",
                    f"Technical considerations for {topic}: Performance and scalability factors",
                    f"Integration approaches for {topic}: Compatibility and maintenance aspects"
                ],
                "resources": [
                    f"Documentation: Official {topic} documentation",
                    f"Examples: Code samples and tutorials for {topic}",
                    f"Community: Forums and discussions about {topic}"
                ],
                "recommendations": [
                    f"Start with basic {topic} implementation",
                    f"Follow established patterns for {topic}",
                    f"Test thoroughly before production deployment"
                ]
            }

            logger.debug(f"Research completed for topic: {topic}")
            return {
                "success": True,
                "research": research_results
            }

        except Exception as e:
            logger.error(f"Research failed for topic {topic}: {e}")
            return {
                "success": False,
                "error": str(e),
                "research": None
            }

    async def add_dependency(
        self,
        task_id: str,
        depends_on_id: str
    ) -> Dict[str, Any]:
        """Add dependency relationship between tasks"""
        try:
            if task_id not in self._tasks:
                return {"success": False, "error": f"Task not found: {task_id}"}

            if depends_on_id not in self._tasks:
                return {"success": False, "error": f"Dependency task not found: {depends_on_id}"}

            task = self._tasks[task_id]
            if depends_on_id not in task.dependencies:
                task.dependencies.append(depends_on_id)
                task.updated_at = datetime.utcnow()

            logger.debug(f"Added dependency: {task_id} depends on {depends_on_id}")
            return {
                "success": True,
                "task_id": task_id,
                "depends_on": depends_on_id
            }

        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            return {"success": False, "error": str(e)}

    async def get_project_status(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """Get project status and progress"""
        try:
            if project_id not in self._projects:
                return {"success": False, "error": f"Project not found: {project_id}"}

            project = self._projects[project_id]
            tasks = project.tasks

            status_counts = {}
            for status in TaskStatus:
                status_counts[status.value] = len([t for t in tasks if t.status == status])

            total_tasks = len(tasks)
            completed_tasks = status_counts.get("done", 0)
            progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

            return {
                "success": True,
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "status": project.status,
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "progress_percentage": round(progress, 2),
                    "status_breakdown": status_counts,
                    "created_at": project.created_at.isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Failed to get project status: {e}")
            return {"success": False, "error": str(e)}


# Utility functions for integration with agents

async def create_agent_task(
    adapter: TaskMasterAdapter,
    agent_name: str,
    task_description: str,
    priority: TaskPriority = TaskPriority.MEDIUM
) -> Dict[str, Any]:
    """Create a task specifically for an agent"""
    try:
        title = f"[{agent_name}] {task_description}"
        result = await adapter.create_task(
            title=title,
            description=f"Task assigned to {agent_name}: {task_description}",
            priority=priority
        )

        if result["success"]:
            # Add agent-specific tags
            task_id = result["task_id"]
            task = adapter._tasks[task_id]
            task.tags.extend(["agent-task", agent_name.lower()])
            task.assignee = agent_name

        return result

    except Exception as e:
        logger.error(f"Failed to create agent task: {e}")
        return {"success": False, "error": str(e)}


async def sync_with_shrimp(
    adapter: TaskMasterAdapter,
    shrimp_tasks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Synchronize tasks with Shrimp Task Manager"""
    try:
        synced_count = 0
        errors = []

        for shrimp_task in shrimp_tasks:
            try:
                # Convert Shrimp task to TaskMaster format
                result = await adapter.create_task(
                    title=shrimp_task.get("name", "Unnamed Task"),
                    description=shrimp_task.get("description", ""),
                    priority=TaskPriority.MEDIUM
                )

                if result["success"]:
                    synced_count += 1
                else:
                    errors.append(f"Failed to sync task: {result.get('error', 'Unknown error')}")

            except Exception as e:
                errors.append(f"Error syncing task {shrimp_task.get('id', 'unknown')}: {str(e)}")

        return {
            "success": len(errors) == 0,
            "synced_tasks": synced_count,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Task synchronization failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "synced_tasks": 0
        }


# Create default adapter instance
default_taskmaster = TaskMasterAdapter()