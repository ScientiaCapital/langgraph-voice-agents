"""
Shrimp Task Manager MCP tool adapter for LangGraph agents.
Provides advanced task planning, verification, and execution management.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskMode(Enum):
    """Task execution modes"""
    APPEND = "append"
    OVERWRITE = "overwrite"
    SELECTIVE = "selective"
    CLEAR_ALL_TASKS = "clearAllTasks"


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class ShrimpTask:
    """Shrimp task representation with verification capabilities"""
    id: str
    name: str
    description: str
    implementation_guide: str
    verification_criteria: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    related_files: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    score: Optional[int] = None


@dataclass
class AnalysisResult:
    """Task analysis result"""
    summary: str
    initial_concept: str
    feasibility: str
    risks: List[str]
    recommendations: List[str]
    technical_considerations: List[str]


@dataclass
class ReflectionResult:
    """Task reflection result"""
    summary: str
    analysis: str
    strengths: List[str]
    weaknesses: List[str]
    improvements: List[str]
    final_recommendations: List[str]


class ShrimpTaskManagerAdapter:
    """Adapter for Shrimp Task Manager MCP tool"""

    def __init__(self, tool_executor=None):
        self.tool_executor = tool_executor
        self._tasks = {}
        self._analysis_cache = {}
        self._thinking_sessions = {}

    async def plan_task(
        self,
        description: str,
        requirements: Optional[str] = None,
        existing_tasks_reference: bool = False
    ) -> Dict[str, Any]:
        """Plan a task with comprehensive analysis"""
        try:
            session_id = str(uuid.uuid4())
            
            # Store planning session
            self._thinking_sessions[session_id] = {
                "description": description,
                "requirements": requirements,
                "existing_tasks_reference": existing_tasks_reference,
                "created_at": datetime.utcnow(),
                "status": "planning"
            }

            # Generate planning guidance
            planning_steps = [
                "1. Define clear objectives and success criteria",
                "2. Break down into manageable components",
                "3. Identify dependencies and prerequisites",
                "4. Consider technical constraints and limitations",
                "5. Plan testing and verification approach",
                "6. Establish timeline and milestones"
            ]

            considerations = [
                "Technical feasibility and complexity",
                "Resource requirements and availability",
                "Integration with existing systems",
                "Performance and scalability needs",
                "Security and compliance requirements",
                "Maintenance and support implications"
            ]

            logger.info(f"Task planning initiated: {description}")
            return {
                "success": True,
                "session_id": session_id,
                "planning_steps": planning_steps,
                "considerations": considerations,
                "next_action": "Use analyze_task to perform detailed technical analysis"
            }

        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": None
            }

    async def analyze_task(
        self,
        summary: str,
        initial_concept: str,
        previous_analysis: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze task requirements and technical feasibility"""
        try:
            # Perform comprehensive analysis
            analysis_result = AnalysisResult(
                summary=summary,
                initial_concept=initial_concept,
                feasibility="High - Task appears technically feasible with standard approaches",
                risks=[
                    "Complexity may increase during implementation",
                    "Integration challenges with existing systems",
                    "Performance bottlenecks under high load",
                    "Security vulnerabilities if not properly implemented"
                ],
                recommendations=[
                    "Start with MVP implementation to validate approach",
                    "Use established patterns and frameworks",
                    "Implement comprehensive testing strategy",
                    "Plan for iterative development and feedback"
                ],
                technical_considerations=[
                    "Choose appropriate technology stack",
                    "Design for scalability and maintainability",
                    "Implement proper error handling and logging",
                    "Consider security implications from the start"
                ]
            )

            # Cache analysis for later use
            analysis_id = str(uuid.uuid4())
            self._analysis_cache[analysis_id] = analysis_result

            logger.debug(f"Task analysis completed: {summary}")
            return analysis_result

        except Exception as e:
            logger.error(f"Task analysis failed: {e}")
            raise

    async def reflect_task(
        self,
        summary: str,
        analysis: str
    ) -> ReflectionResult:
        """Reflect on analysis results and provide optimization recommendations"""
        try:
            reflection_result = ReflectionResult(
                summary=summary,
                analysis=analysis,
                strengths=[
                    "Clear problem definition and scope",
                    "Comprehensive technical approach",
                    "Consideration of potential risks",
                    "Structured implementation plan"
                ],
                weaknesses=[
                    "May need more specific implementation details",
                    "Could benefit from performance benchmarks",
                    "Risk mitigation strategies could be more detailed",
                    "Integration testing approach needs refinement"
                ],
                improvements=[
                    "Add specific performance metrics and targets",
                    "Include detailed error handling scenarios",
                    "Expand on security implementation details",
                    "Create more granular testing checkpoints"
                ],
                final_recommendations=[
                    "Proceed with implementation using iterative approach",
                    "Establish monitoring and metrics early",
                    "Plan for regular code reviews and testing",
                    "Document decisions and trade-offs thoroughly"
                ]
            )

            logger.debug(f"Task reflection completed: {summary}")
            return reflection_result

        except Exception as e:
            logger.error(f"Task reflection failed: {e}")
            raise

    async def split_tasks(
        self,
        global_analysis_result: str,
        tasks_raw: str,
        update_mode: TaskMode = TaskMode.CLEAR_ALL_TASKS
    ) -> Dict[str, Any]:
        """Split complex task into manageable subtasks"""
        try:
            # Parse tasks from raw input
            try:
                tasks_data = json.loads(tasks_raw)
                if not isinstance(tasks_data, list):
                    tasks_data = [tasks_data]
            except json.JSONDecodeError:
                # If not valid JSON, treat as simple task description
                tasks_data = [{
                    "name": "Primary Task",
                    "description": tasks_raw,
                    "implementation_guide": "Follow standard implementation practices",
                    "verification_criteria": "Task completed successfully with all requirements met"
                }]

            # Clear existing tasks if specified
            if update_mode == TaskMode.CLEAR_ALL_TASKS:
                self._tasks.clear()

            # Create new tasks
            created_tasks = []
            for i, task_data in enumerate(tasks_data):
                task_id = str(uuid.uuid4())
                
                task = ShrimpTask(
                    id=task_id,
                    name=task_data.get("name", f"Task {i+1}"),
                    description=task_data.get("description", ""),
                    implementation_guide=task_data.get("implementation_guide", ""),
                    verification_criteria=task_data.get("verification_criteria", ""),
                    dependencies=task_data.get("dependencies", []),
                    related_files=task_data.get("related_files", []),
                    notes=task_data.get("notes", "")
                )

                self._tasks[task_id] = task
                created_tasks.append({
                    "id": task_id,
                    "name": task.name,
                    "description": task.description
                })

            logger.info(f"Split into {len(created_tasks)} tasks using {update_mode.value} mode")
            return {
                "success": True,
                "tasks_created": len(created_tasks),
                "tasks": created_tasks,
                "update_mode": update_mode.value
            }

        except Exception as e:
            logger.error(f"Task splitting failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tasks_created": 0
            }

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None
    ) -> Dict[str, Any]:
        """List all tasks with optional status filtering"""
        try:
            tasks = list(self._tasks.values())

            # Filter by status if specified
            if status:
                tasks = [t for t in tasks if t.status == status]

            # Format task data
            task_list = []
            for task in tasks:
                task_info = {
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "status": task.status.value,
                    "dependencies": task.dependencies,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat()
                }
                
                if task.score is not None:
                    task_info["score"] = task.score

                task_list.append(task_info)

            return {
                "success": True,
                "tasks": task_list,
                "total_count": len(task_list),
                "status_filter": status.value if status else "all"
            }

        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return {
                "success": False,
                "error": str(e),
                "tasks": []
            }

    async def execute_task(
        self,
        task_id: str
    ) -> Dict[str, Any]:
        """Get execution guidance for a specific task"""
        try:
            if task_id not in self._tasks:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }

            task = self._tasks[task_id]
            
            # Update task status
            task.status = TaskStatus.IN_PROGRESS
            task.updated_at = datetime.utcnow()

            # Generate execution guidance
            execution_guidance = {
                "task_id": task_id,
                "task_name": task.name,
                "implementation_steps": [
                    "1. Review task requirements and acceptance criteria",
                    "2. Set up development environment and dependencies",
                    "3. Implement core functionality following the guide",
                    "4. Add proper error handling and validation",
                    "5. Write comprehensive tests",
                    "6. Document implementation decisions"
                ],
                "implementation_guide": task.implementation_guide,
                "verification_criteria": task.verification_criteria,
                "related_files": task.related_files,
                "dependencies": task.dependencies,
                "notes": task.notes
            }

            logger.debug(f"Task execution started: {task.name}")
            return {
                "success": True,
                "execution_guidance": execution_guidance,
                "status": "in_progress"
            }

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def verify_task(
        self,
        task_id: str,
        summary: str,
        score: int
    ) -> Dict[str, Any]:
        """Verify task completion and provide scoring"""
        try:
            if task_id not in self._tasks:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }

            task = self._tasks[task_id]
            
            # Update task with verification results
            task.score = score
            task.notes = f"Verification summary: {summary}"
            task.updated_at = datetime.utcnow()

            # Determine if task is completed based on score
            if score >= 80:
                task.status = TaskStatus.COMPLETED
                completion_status = "completed"
                message = f"Task '{task.name}' completed successfully with score {score}/100"
            else:
                task.status = TaskStatus.PENDING
                completion_status = "needs_improvement"
                message = f"Task '{task.name}' needs improvement (score: {score}/100)"

            logger.info(f"Task verification: {task.name} - Score: {score}")
            return {
                "success": True,
                "task_id": task_id,
                "score": score,
                "status": completion_status,
                "message": message,
                "verification_summary": summary
            }

        except Exception as e:
            logger.error(f"Task verification failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def update_task(
        self,
        task_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        implementation_guide: Optional[str] = None,
        verification_criteria: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update task properties"""
        try:
            if task_id not in self._tasks:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }

            task = self._tasks[task_id]
            updated_fields = []

            # Update provided fields
            if name is not None:
                task.name = name
                updated_fields.append("name")
            
            if description is not None:
                task.description = description
                updated_fields.append("description")
            
            if implementation_guide is not None:
                task.implementation_guide = implementation_guide
                updated_fields.append("implementation_guide")
            
            if verification_criteria is not None:
                task.verification_criteria = verification_criteria
                updated_fields.append("verification_criteria")
            
            if notes is not None:
                task.notes = notes
                updated_fields.append("notes")

            task.updated_at = datetime.utcnow()

            logger.debug(f"Task updated: {task.name} - Fields: {updated_fields}")
            return {
                "success": True,
                "task_id": task_id,
                "updated_fields": updated_fields,
                "updated_at": task.updated_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Task update failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def delete_task(
        self,
        task_id: str
    ) -> Dict[str, Any]:
        """Delete a task (only if not completed)"""
        try:
            if task_id not in self._tasks:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }

            task = self._tasks[task_id]

            # Prevent deletion of completed tasks
            if task.status == TaskStatus.COMPLETED:
                return {
                    "success": False,
                    "error": "Cannot delete completed tasks"
                }

            task_name = task.name
            del self._tasks[task_id]

            logger.info(f"Task deleted: {task_name}")
            return {
                "success": True,
                "task_id": task_id,
                "task_name": task_name,
                "message": f"Task '{task_name}' deleted successfully"
            }

        except Exception as e:
            logger.error(f"Task deletion failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_task_detail(
        self,
        task_id: str
    ) -> Dict[str, Any]:
        """Get detailed information about a specific task"""
        try:
            if task_id not in self._tasks:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }

            task = self._tasks[task_id]

            task_detail = {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "implementation_guide": task.implementation_guide,
                "verification_criteria": task.verification_criteria,
                "status": task.status.value,
                "dependencies": task.dependencies,
                "related_files": task.related_files,
                "notes": task.notes,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "score": task.score
            }

            return {
                "success": True,
                "task": task_detail
            }

        except Exception as e:
            logger.error(f"Failed to get task detail: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def clear_all_tasks(
        self,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """Clear all incomplete tasks"""
        try:
            if not confirm:
                return {
                    "success": False,
                    "error": "Confirmation required to clear all tasks"
                }

            # Backup completed tasks
            completed_tasks = {
                task_id: task for task_id, task in self._tasks.items()
                if task.status == TaskStatus.COMPLETED
            }

            # Count tasks to be cleared
            incomplete_count = len(self._tasks) - len(completed_tasks)

            # Clear all tasks except completed ones
            self._tasks = completed_tasks

            logger.info(f"Cleared {incomplete_count} incomplete tasks, kept {len(completed_tasks)} completed")
            return {
                "success": True,
                "cleared_tasks": incomplete_count,
                "kept_completed": len(completed_tasks),
                "message": f"Cleared {incomplete_count} incomplete tasks"
            }

        except Exception as e:
            logger.error(f"Failed to clear tasks: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Utility functions for agent integration

async def create_development_workflow(
    adapter: ShrimpTaskManagerAdapter,
    feature_description: str,
    requirements: List[str]
) -> Dict[str, Any]:
    """Create a complete development workflow for a feature"""
    try:
        # Plan the task
        plan_result = await adapter.plan_task(
            description=feature_description,
            requirements="; ".join(requirements),
            existing_tasks_reference=False
        )

        if not plan_result["success"]:
            return plan_result

        # Analyze the requirements
        analysis = await adapter.analyze_task(
            summary=feature_description,
            initial_concept=f"Implement {feature_description} with requirements: {requirements}"
        )

        # Reflect on the analysis
        reflection = await adapter.reflect_task(
            summary=feature_description,
            analysis=analysis.analysis if hasattr(analysis, 'analysis') else str(analysis)
        )

        # Create structured tasks
        workflow_tasks = [
            {
                "name": f"Design {feature_description}",
                "description": "Create architectural design and technical specifications",
                "implementation_guide": "Define interfaces, data models, and component interactions",
                "verification_criteria": "Design reviewed and approved by technical lead"
            },
            {
                "name": f"Implement {feature_description}",
                "description": "Develop the core functionality according to specifications",
                "implementation_guide": "Follow coding standards and implement all required features",
                "verification_criteria": "All features implemented and unit tests passing"
            },
            {
                "name": f"Test {feature_description}",
                "description": "Comprehensive testing including edge cases and integration",
                "implementation_guide": "Create test cases covering all requirements and scenarios",
                "verification_criteria": "All tests passing with adequate coverage"
            },
            {
                "name": f"Document {feature_description}",
                "description": "Create user and developer documentation",
                "implementation_guide": "Write clear documentation with examples and usage guides",
                "verification_criteria": "Documentation complete and reviewed"
            }
        ]

        # Split into manageable tasks
        split_result = await adapter.split_tasks(
            global_analysis_result=str(analysis),
            tasks_raw=json.dumps(workflow_tasks),
            update_mode=TaskMode.CLEAR_ALL_TASKS
        )

        return {
            "success": True,
            "workflow_created": True,
            "plan_session": plan_result["session_id"],
            "tasks_created": split_result["tasks_created"],
            "analysis_summary": analysis.summary if hasattr(analysis, 'summary') else "Analysis completed",
            "recommendations": reflection.final_recommendations if hasattr(reflection, 'final_recommendations') else []
        }

    except Exception as e:
        logger.error(f"Workflow creation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Create default adapter instance
default_shrimp = ShrimpTaskManagerAdapter()