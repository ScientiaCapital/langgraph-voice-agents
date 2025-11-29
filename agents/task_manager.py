"""
Task Manager - Voice-enabled productivity assistant.

Handles:
- Task creation and organization
- Priority and deadline management
- Project breakdown and planning
- Status updates and reminders
- Focus and productivity coaching

Uses Gemini for fast responses, Claude for complex planning.
NO OpenAI dependencies.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from langgraph.graph import StateGraph, END

from agents.voice_agent import VoiceAgent, VoiceAgentConfig
from core.base_graph import AgentState, AgentMode
from llm import LLMProvider
from llm.router import TaskType

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    URGENT = "urgent"      # Do now
    HIGH = "high"          # Do today
    MEDIUM = "medium"      # Do this week
    LOW = "low"            # Do when possible
    SOMEDAY = "someday"    # Maybe later


class TaskStatus(Enum):
    """Task status"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A single task"""
    id: str
    title: str
    description: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.TODO
    due_date: Optional[str] = None
    project: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "due_date": self.due_date,
            "project": self.project,
            "tags": self.tags,
            "subtasks": self.subtasks,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class Project:
    """A project containing tasks"""
    id: str
    name: str
    description: Optional[str] = None
    tasks: List[str] = field(default_factory=list)  # Task IDs
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TaskManager(VoiceAgent):
    """
    Voice-enabled task and project manager.

    Features:
    - Natural language task creation
    - Smart priority detection
    - Project breakdown
    - Status reporting
    - Focus coaching

    Uses Gemini for quick operations, Claude for planning.
    """

    def __init__(
        self,
        name: str = "Taskmaster",
        mode: AgentMode = AgentMode.VOICE,
    ):
        """
        Initialize Task Manager.

        Args:
            name: Assistant's name
            mode: Voice or text mode
        """
        config = VoiceAgentConfig(
            agent_name=name,
            agent_description="a productivity and task management assistant",
            system_prompt=self._create_system_prompt(name),
            default_provider=LLMProvider.GEMINI,  # Fast for quick operations
            task_type=TaskType.FAST,
            greeting_message=f"Hello! I'm {name}, your task manager. What would you like to accomplish today?",
            max_history_turns=15,
        )

        super().__init__(config=config, mode=mode)

        # Task storage (in-memory for now)
        self._tasks: Dict[str, Task] = {}
        self._projects: Dict[str, Project] = {}
        self._next_task_id = 1

        logger.info(f"Task Manager '{name}' initialized")

    def _create_system_prompt(self, name: str) -> str:
        """Create system prompt for task manager"""
        return f"""You are {name}, a voice-enabled task and productivity assistant.

YOUR ROLE:
- Help users capture and organize tasks
- Break down projects into actionable steps
- Provide status updates and reminders
- Coach on productivity and focus

VOICE INTERACTION STYLE:
1. Be concise and action-oriented
2. Confirm task details before adding
3. Use natural time references ("tomorrow", "next week")
4. Celebrate completed tasks
5. Gently nudge on overdue items

TASK MANAGEMENT APPROACH:
- Every task should be specific and actionable
- Help clarify vague tasks into clear actions
- Suggest priorities based on urgency and importance
- Break large tasks into smaller steps

When users mention tasks:
- Identify the action needed
- Detect priority from words like "urgent", "important", "whenever"
- Extract any deadlines mentioned
- Suggest project grouping if related to existing work

Current date/time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}
"""

    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        task_id = f"task_{self._next_task_id}"
        self._next_task_id += 1
        return task_id

    async def add_task(
        self,
        title: str,
        description: Optional[str] = None,
        priority: Optional[str] = None,
        due_date: Optional[str] = None,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Task:
        """
        Add a new task.

        Args:
            title: Task title
            description: Optional description
            priority: Priority level
            due_date: Due date
            project: Project name
            tags: Task tags

        Returns:
            Created task
        """
        # Parse priority
        task_priority = TaskPriority.MEDIUM
        if priority:
            try:
                task_priority = TaskPriority(priority.lower())
            except ValueError:
                pass

        # Create task
        task = Task(
            id=self._generate_task_id(),
            title=title,
            description=description,
            priority=task_priority,
            due_date=due_date,
            project=project,
            tags=tags or [],
        )

        self._tasks[task.id] = task

        # Add to project if specified
        if project:
            await self._ensure_project(project)
            self._projects[project].tasks.append(task.id)

        logger.info(f"Task added: {task.title} ({task.id})")

        # Confirmation message
        response = f"Got it! I've added '{title}' to your tasks"
        if priority:
            response += f" with {priority} priority"
        if due_date:
            response += f", due {due_date}"
        response += "."

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return task

    async def _ensure_project(self, name: str):
        """Ensure project exists"""
        if name not in self._projects:
            self._projects[name] = Project(
                id=name.lower().replace(" ", "_"),
                name=name,
            )

    async def complete_task(self, task_id: str) -> Optional[Task]:
        """
        Mark a task as complete.

        Args:
            task_id: Task ID to complete

        Returns:
            Completed task or None
        """
        task = self._tasks.get(task_id)
        if not task:
            if self.mode == AgentMode.VOICE:
                await self.speak("I couldn't find that task.")
            return None

        task.status = TaskStatus.DONE
        task.completed_at = datetime.now().isoformat()

        response = f"Excellent! '{task.title}' is now complete. "

        # Check remaining tasks
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.TODO]
        if pending:
            response += f"You have {len(pending)} tasks remaining."
        else:
            response += "You've completed all your tasks!"

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        logger.info(f"Task completed: {task.title}")
        return task

    async def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[Task]:
        """
        List tasks with optional filters.

        Args:
            status: Filter by status
            priority: Filter by priority
            project: Filter by project

        Returns:
            Matching tasks
        """
        tasks = list(self._tasks.values())

        # Apply filters
        if status:
            try:
                s = TaskStatus(status.lower())
                tasks = [t for t in tasks if t.status == s]
            except ValueError:
                pass

        if priority:
            try:
                p = TaskPriority(priority.lower())
                tasks = [t for t in tasks if t.priority == p]
            except ValueError:
                pass

        if project:
            tasks = [t for t in tasks if t.project and project.lower() in t.project.lower()]

        # Sort by priority
        priority_order = {
            TaskPriority.URGENT: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
            TaskPriority.SOMEDAY: 4,
        }
        tasks.sort(key=lambda t: priority_order.get(t.priority, 5))

        return tasks

    async def get_status(self) -> str:
        """
        Get current task status summary.

        Returns:
            Status summary
        """
        all_tasks = list(self._tasks.values())
        todo = [t for t in all_tasks if t.status == TaskStatus.TODO]
        in_progress = [t for t in all_tasks if t.status == TaskStatus.IN_PROGRESS]
        done_today = [
            t for t in all_tasks
            if t.status == TaskStatus.DONE
            and t.completed_at
            and t.completed_at.startswith(datetime.now().strftime('%Y-%m-%d'))
        ]

        # Build response
        parts = []

        if not all_tasks:
            response = "You don't have any tasks yet. What would you like to work on?"
        else:
            if in_progress:
                parts.append(f"You're currently working on {len(in_progress)} task" +
                           ("s" if len(in_progress) > 1 else ""))
            if todo:
                urgent = [t for t in todo if t.priority == TaskPriority.URGENT]
                if urgent:
                    parts.append(f"{len(urgent)} urgent task" + ("s" if len(urgent) > 1 else "") + " needs attention")
                parts.append(f"{len(todo)} task" + ("s" if len(todo) > 1 else "") + " pending")
            if done_today:
                parts.append(f"you've completed {len(done_today)} task" +
                           ("s" if len(done_today) > 1 else "") + " today")

            response = "Here's your status: " + ", ".join(parts) + "."

            # Suggest next action
            if urgent := [t for t in todo if t.priority == TaskPriority.URGENT]:
                response += f" You might want to focus on: {urgent[0].title}"

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    async def break_down_project(
        self,
        project_name: str,
        description: str,
    ) -> List[Task]:
        """
        Break down a project into tasks using AI.

        Args:
            project_name: Name of the project
            description: What the project involves

        Returns:
            List of created tasks
        """
        # Use Claude for complex planning
        system = """Break down this project into specific, actionable tasks.
For each task, specify:
- A clear title (action verb + object)
- Suggested priority (urgent/high/medium/low)
- Any dependencies on other tasks

Return a JSON array of tasks like:
[{"title": "...", "priority": "...", "depends_on": null or "task title"}]

Keep tasks concrete and achievable. Aim for 3-8 tasks."""

        response = await self.generate_response(
            user_message=f"Project: {project_name}\nDescription: {description}",
            system=system,
            provider=LLMProvider.CLAUDE,  # Use Claude for planning
            task_type=TaskType.REASONING,
        )

        # Parse tasks from response
        created_tasks = []
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                task_data = json.loads(json_match.group())
                for item in task_data:
                    task = await self.add_task(
                        title=item.get("title", "Untitled task"),
                        priority=item.get("priority", "medium"),
                        project=project_name,
                    )
                    created_tasks.append(task)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse project breakdown: {e}")

        # Report results
        if created_tasks:
            summary = f"I've broken down '{project_name}' into {len(created_tasks)} tasks. "
            summary += f"The first task is: {created_tasks[0].title}"
        else:
            summary = f"I'll help you plan '{project_name}'. Let's start by identifying the key deliverables."

        self._add_to_history("user", f"Break down project: {project_name}")
        self._add_to_history("assistant", summary)

        if self.mode == AgentMode.VOICE:
            await self.speak(summary)

        return created_tasks

    async def focus_coaching(self, current_task: Optional[str] = None) -> str:
        """
        Provide focus and productivity coaching.

        Args:
            current_task: What user is working on

        Returns:
            Coaching advice
        """
        # Get context
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.TODO]
        in_progress = [t for t in self._tasks.values() if t.status == TaskStatus.IN_PROGRESS]

        context = f"""
User has {len(pending)} pending tasks and {len(in_progress)} in progress.
Current time: {datetime.now().strftime('%I:%M %p')}
"""
        if current_task:
            context += f"Currently working on: {current_task}"

        system = """You're a friendly productivity coach. Give brief, actionable advice.
Consider:
- Time of day and energy levels
- Number of pending tasks
- The importance of focus and breaks

Keep it conversational and encouraging. One or two sentences max."""

        response = await self.generate_response(
            user_message=f"I need help focusing. {context}",
            system=system,
            task_type=TaskType.FAST,
        )

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    async def process_input(self, input_data: str) -> str:
        """
        Process task-related input with smart routing.

        Args:
            input_data: User's message

        Returns:
            Response
        """
        input_lower = input_data.lower()

        # Detect intent
        if any(word in input_lower for word in ["add", "create", "new task", "remind me", "i need to"]):
            # Parse task from natural language
            return await self._parse_and_add_task(input_data)

        if any(word in input_lower for word in ["done", "complete", "finished", "mark complete"]):
            return await self._handle_completion(input_data)

        if any(word in input_lower for word in ["status", "what's left", "remaining", "pending", "my tasks"]):
            return await self.get_status()

        if any(word in input_lower for word in ["break down", "plan", "project"]):
            return await self._handle_project(input_data)

        if any(word in input_lower for word in ["focus", "help me concentrate", "distracted"]):
            return await self.focus_coaching()

        if any(word in input_lower for word in ["list", "show", "what are"]):
            tasks = await self.list_tasks()
            return await self._speak_task_list(tasks)

        # Default to conversation
        return await super().process_input(input_data)

    async def _parse_and_add_task(self, input_data: str) -> str:
        """Parse natural language and add task"""
        system = """Extract task details from the user's message.
Return JSON with: {"title": "...", "priority": "medium", "due_date": null or "tomorrow" etc}
Keep the title actionable and specific."""

        response = await self.generate_response(
            user_message=input_data,
            system=system,
            task_type=TaskType.FAST,
        )

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                task = await self.add_task(
                    title=data.get("title", input_data[:50]),
                    priority=data.get("priority"),
                    due_date=data.get("due_date"),
                )
                return f"Added: {task.title}"
        except Exception as e:
            logger.error(f"Task parsing failed: {e}")

        # Fallback - add as-is
        task = await self.add_task(title=input_data[:100])
        return f"Added: {task.title}"

    async def _handle_completion(self, input_data: str) -> str:
        """Handle task completion"""
        # Try to find which task
        input_lower = input_data.lower()

        # Check for most recent task
        if "that" in input_lower or "it" in input_lower:
            in_progress = [t for t in self._tasks.values() if t.status == TaskStatus.IN_PROGRESS]
            if in_progress:
                task = await self.complete_task(in_progress[0].id)
                return f"Completed: {task.title}" if task else "Couldn't find that task."

        # Search by title
        for task in self._tasks.values():
            if task.status == TaskStatus.TODO and task.title.lower() in input_lower:
                completed = await self.complete_task(task.id)
                return f"Completed: {completed.title}" if completed else ""

        # Default - ask for clarification
        tasks = [t for t in self._tasks.values() if t.status == TaskStatus.TODO]
        if tasks:
            return f"Which task did you complete? You have: {', '.join(t.title for t in tasks[:3])}"
        return "You don't have any pending tasks!"

    async def _handle_project(self, input_data: str) -> str:
        """Handle project breakdown request"""
        # Extract project name and description
        return await self.break_down_project("New Project", input_data)

    async def _speak_task_list(self, tasks: List[Task]) -> str:
        """Convert task list to speech"""
        if not tasks:
            response = "You don't have any tasks right now."
        elif len(tasks) == 1:
            response = f"You have one task: {tasks[0].title}"
        else:
            response = f"You have {len(tasks)} tasks. "
            for i, task in enumerate(tasks[:5]):  # Max 5 for voice
                if i == 0:
                    response += f"First, {task.title}. "
                elif i == len(tasks[:5]) - 1:
                    response += f"And finally, {task.title}."
                else:
                    response += f"Then, {task.title}. "

            if len(tasks) > 5:
                response += f" Plus {len(tasks) - 5} more."

        if self.mode == AgentMode.VOICE:
            await self.speak(response)

        return response

    def _build_graph(self) -> StateGraph:
        """Build LangGraph for task manager"""
        graph = StateGraph(AgentState)

        graph.add_node("parse", self._parse_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("respond", self._respond_node)

        graph.set_entry_point("parse")
        graph.add_edge("parse", "execute")
        graph.add_edge("execute", "respond")
        graph.add_edge("respond", END)

        return graph

    async def _parse_node(self, state: dict) -> dict:
        """Parse task intent"""
        task = state.get("current_task", "")
        context = state.get("task_context", {})

        if "add" in task.lower() or "create" in task.lower():
            context["intent"] = "add"
        elif "complete" in task.lower() or "done" in task.lower():
            context["intent"] = "complete"
        elif "list" in task.lower() or "show" in task.lower():
            context["intent"] = "list"
        else:
            context["intent"] = "general"

        state["task_context"] = context
        return state

    async def _execute_node(self, state: dict) -> dict:
        """Execute task action"""
        return state

    async def _respond_node(self, state: dict) -> dict:
        """Generate response"""
        task = state.get("current_task", "")
        if task:
            response = await self.process_input(task)
            state["messages"] = state.get("messages", []) + [
                {"role": "assistant", "content": response}
            ]
        return state


# Factory function
def create_task_manager(
    name: str = "Taskmaster",
    voice_mode: bool = True,
) -> TaskManager:
    """
    Create a Task Manager instance.

    Args:
        name: Assistant's name
        voice_mode: Whether to enable voice mode

    Returns:
        Configured TaskManager
    """
    return TaskManager(
        name=name,
        mode=AgentMode.VOICE if voice_mode else AgentMode.TEXT,
    )
