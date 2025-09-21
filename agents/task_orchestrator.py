"""
Task Orchestrator Agent - Voice-enabled strategic planning and coordination.
Manages high-level task delegation and workflow orchestration across multiple agents.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from langgraph import StateGraph, START, END
from langgraph.graph import Graph

from ..core.base_graph import BaseAgent, AgentState, AgentMode, MultiModalMixin
from ..core.state_management import StateManager, StateStatus, StateMetadata
from ..tools.sequential_thinking_tools import SequentialThinkingAdapter, ThinkingStage
from ..tools.serena_tools import SerenaAdapter
from ..tools.context7_tools import Context7Adapter
from ..tools.taskmaster_tools import TaskMasterAdapter, TaskPriority
from ..tools.shrimp_tools import ShrimpTaskManagerAdapter, TaskMode
from ..voice.livekit_client import LiveKitClient, LiveKitConfig

logger = logging.getLogger(__name__)


class OrchestrationStrategy(Enum):
    """Task orchestration strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class TaskComplexity(Enum):
    """Task complexity levels for delegation"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


@dataclass
class OrchestrationPlan:
    """Strategic plan for task orchestration"""
    strategy: OrchestrationStrategy
    tasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    resource_allocation: Dict[str, str]
    timeline: Dict[str, str]
    risk_assessment: List[str]
    success_criteria: List[str]


class TaskOrchestratorAgent(BaseAgent, MultiModalMixin):
    """
    Strategic task orchestration agent with voice capabilities.
    Follows mandatory MCP tool order: Sequential Thinking → Serena → Context7
    """

    def __init__(
        self,
        agent_id: str = "task-orchestrator",
        state_manager: Optional[StateManager] = None,
        livekit_config: Optional[LiveKitConfig] = None
    ):
        super().__init__(agent_id, state_manager)
        MultiModalMixin.__init__(self, livekit_config)

        # Initialize MCP tool adapters in mandatory order
        self.sequential_thinking = SequentialThinkingAdapter()
        self.serena = SerenaAdapter()
        self.context7 = Context7Adapter()
        self.taskmaster = TaskMasterAdapter()
        self.shrimp = ShrimpTaskManagerAdapter()

        # Orchestration state
        self.active_plans = {}
        self.delegated_tasks = {}
        self.agent_capabilities = {
            "task-executor": ["implementation", "coding", "testing"],
            "task-checker": ["verification", "validation", "quality-assurance"]
        }

        # Voice command handlers
        self.voice_commands = {
            "plan task": self._handle_plan_task_voice,
            "delegate task": self._handle_delegate_task_voice,
            "check status": self._handle_status_check_voice,
            "emergency stop": self._handle_emergency_stop_voice
        }

    def create_graph(self) -> StateGraph:
        """Create the orchestrator workflow graph"""
        workflow = StateGraph(AgentState)

        # Add nodes for each orchestration phase
        workflow.add_node("think_strategically", self._think_strategically)
        workflow.add_node("analyze_codebase", self._analyze_codebase)
        workflow.add_node("research_best_practices", self._research_best_practices)
        workflow.add_node("create_orchestration_plan", self._create_orchestration_plan)
        workflow.add_node("delegate_tasks", self._delegate_tasks)
        workflow.add_node("monitor_progress", self._monitor_progress)
        workflow.add_node("coordinate_agents", self._coordinate_agents)
        workflow.add_node("finalize_results", self._finalize_results)

        # Add edges following the mandatory MCP tool order
        workflow.add_edge(START, "think_strategically")
        workflow.add_edge("think_strategically", "analyze_codebase")
        workflow.add_edge("analyze_codebase", "research_best_practices")
        workflow.add_edge("research_best_practices", "create_orchestration_plan")
        workflow.add_edge("create_orchestration_plan", "delegate_tasks")
        workflow.add_edge("delegate_tasks", "monitor_progress")
        workflow.add_edge("monitor_progress", "coordinate_agents")
        workflow.add_edge("coordinate_agents", "finalize_results")
        workflow.add_edge("finalize_results", END)

        # Add conditional edges for iteration
        workflow.add_conditional_edges(
            "monitor_progress",
            self._should_continue_monitoring,
            {
                "continue": "coordinate_agents",
                "complete": "finalize_results",
                "escalate": "think_strategically"
            }
        )

        return workflow

    async def _think_strategically(self, state: AgentState) -> AgentState:
        """Strategic thinking using Sequential Thinking MCP tool"""
        try:
            logger.info("Starting strategic thinking phase")
            
            # Use Sequential Thinking for strategic planning
            thinking_result = await self.sequential_thinking.plan_implementation(
                feature_description=state.current_request,
                context=state.context
            )

            if thinking_result.success:
                # Store thinking process results
                state.mcp_results["sequential_thinking"] = {
                    "steps": [step.thought for step in thinking_result.steps],
                    "stages_covered": [stage.value for stage in thinking_result.stages_covered],
                    "conclusion": thinking_result.final_conclusion,
                    "total_steps": thinking_result.total_steps
                }

                # Voice feedback if in voice mode
                if state.mode == AgentMode.VOICE and self.livekit_client:
                    await self.speak(f"Strategic analysis complete. Identified {thinking_result.total_steps} key considerations for {state.current_request}")

                logger.info(f"Strategic thinking completed: {thinking_result.total_steps} steps")
            else:
                state.errors.append(f"Strategic thinking failed: {thinking_result.error_message}")

        except Exception as e:
            error_msg = f"Strategic thinking error: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _analyze_codebase(self, state: AgentState) -> AgentState:
        """Analyze codebase using Serena MCP tool"""
        try:
            logger.info("Starting codebase analysis phase")

            # Use Serena for codebase intelligence
            analysis_result = await self.serena.analyze_project_structure(".")
            
            if analysis_result["success"]:
                # Store codebase analysis
                state.mcp_results["serena"] = {
                    "project_structure": analysis_result.get("structure", {}),
                    "key_files": analysis_result.get("key_files", []),
                    "technologies": analysis_result.get("technologies", []),
                    "complexity_score": analysis_result.get("complexity", 0),
                    "recommendations": analysis_result.get("recommendations", [])
                }

                # Voice feedback
                if state.mode == AgentMode.VOICE and self.livekit_client:
                    tech_count = len(analysis_result.get("technologies", []))
                    await self.speak(f"Codebase analysis complete. Found {tech_count} technologies and {len(analysis_result.get('key_files', []))} key files.")

                logger.info("Codebase analysis completed successfully")
            else:
                state.errors.append(f"Codebase analysis failed: {analysis_result.get('error', 'Unknown error')}")

        except Exception as e:
            error_msg = f"Codebase analysis error: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _research_best_practices(self, state: AgentState) -> AgentState:
        """Research best practices using Context7 MCP tool"""
        try:
            logger.info("Starting best practices research phase")

            # Extract technologies from Serena analysis
            technologies = state.mcp_results.get("serena", {}).get("technologies", [])
            
            research_results = {}
            for tech in technologies[:3]:  # Limit to top 3 technologies
                tech_research = await self.context7.get_best_practices(tech)
                if tech_research["success"]:
                    research_results[tech] = tech_research["best_practices"]

            # Store research results
            state.mcp_results["context7"] = {
                "researched_technologies": list(research_results.keys()),
                "best_practices": research_results,
                "documentation_available": len(research_results) > 0
            }

            # Voice feedback
            if state.mode == AgentMode.VOICE and self.livekit_client:
                await self.speak(f"Research complete. Gathered best practices for {len(research_results)} technologies.")

            logger.info(f"Best practices research completed for {len(research_results)} technologies")

        except Exception as e:
            error_msg = f"Best practices research error: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _create_orchestration_plan(self, state: AgentState) -> AgentState:
        """Create comprehensive orchestration plan"""
        try:
            logger.info("Creating orchestration plan")

            # Analyze task complexity
            complexity = self._assess_task_complexity(state)
            
            # Determine orchestration strategy
            strategy = self._select_orchestration_strategy(complexity, state)

            # Create plan using collected intelligence
            plan = OrchestrationPlan(
                strategy=strategy,
                tasks=self._decompose_tasks(state),
                dependencies=self._identify_dependencies(state),
                resource_allocation=self._allocate_resources(state),
                timeline=self._create_timeline(state),
                risk_assessment=self._assess_risks(state),
                success_criteria=self._define_success_criteria(state)
            )

            # Store orchestration plan
            state.orchestration_plan = plan
            state.context["orchestration_plan"] = {
                "strategy": plan.strategy.value,
                "task_count": len(plan.tasks),
                "estimated_duration": plan.timeline.get("total_duration", "unknown"),
                "complexity": complexity.value
            }

            # Voice feedback
            if state.mode == AgentMode.VOICE and self.livekit_client:
                await self.speak(f"Orchestration plan created. {len(plan.tasks)} tasks identified using {strategy.value} strategy.")

            logger.info(f"Orchestration plan created: {len(plan.tasks)} tasks, {strategy.value} strategy")

        except Exception as e:
            error_msg = f"Orchestration planning error: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _delegate_tasks(self, state: AgentState) -> AgentState:
        """Delegate tasks to appropriate agents"""
        try:
            logger.info("Starting task delegation")

            if not hasattr(state, 'orchestration_plan'):
                state.errors.append("No orchestration plan available for delegation")
                return state

            plan = state.orchestration_plan
            delegated_count = 0

            for task in plan.tasks:
                # Determine best agent for task
                assigned_agent = self._assign_task_to_agent(task)
                
                if assigned_agent:
                    # Create task in TaskMaster
                    task_result = await self.taskmaster.create_task(
                        title=task["title"],
                        description=task["description"],
                        priority=TaskPriority(task.get("priority", "medium"))
                    )

                    if task_result["success"]:
                        # Store delegation info
                        task_id = task_result["task_id"]
                        self.delegated_tasks[task_id] = {
                            "assigned_agent": assigned_agent,
                            "task_data": task,
                            "status": "delegated",
                            "delegated_at": datetime.utcnow()
                        }
                        delegated_count += 1

            # Update state
            state.context["delegated_tasks"] = {
                "total_delegated": delegated_count,
                "agents_involved": list(set(dt["assigned_agent"] for dt in self.delegated_tasks.values())),
                "delegation_complete": delegated_count == len(plan.tasks)
            }

            # Voice feedback
            if state.mode == AgentMode.VOICE and self.livekit_client:
                await self.speak(f"Task delegation complete. {delegated_count} tasks assigned to specialized agents.")

            logger.info(f"Task delegation completed: {delegated_count}/{len(plan.tasks)} tasks delegated")

        except Exception as e:
            error_msg = f"Task delegation error: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _monitor_progress(self, state: AgentState) -> AgentState:
        """Monitor progress of delegated tasks"""
        try:
            logger.info("Monitoring task progress")

            progress_summary = {
                "total_tasks": len(self.delegated_tasks),
                "completed": 0,
                "in_progress": 0,
                "pending": 0,
                "blocked": 0
            }

            # Check status of each delegated task
            for task_id, delegation_info in self.delegated_tasks.items():
                # Get task status from TaskMaster
                task_status = await self.taskmaster.get_tasks()
                
                if task_status["success"]:
                    for task in task_status["tasks"]:
                        if task["id"] == task_id:
                            status = task["status"]
                            if status == "done":
                                progress_summary["completed"] += 1
                            elif status == "in-progress":
                                progress_summary["in_progress"] += 1
                            elif status == "pending":
                                progress_summary["pending"] += 1
                            else:
                                progress_summary["blocked"] += 1
                            break

            # Update state with progress
            state.context["progress"] = progress_summary
            completion_rate = progress_summary["completed"] / progress_summary["total_tasks"] if progress_summary["total_tasks"] > 0 else 0

            # Voice feedback
            if state.mode == AgentMode.VOICE and self.livekit_client:
                await self.speak(f"Progress update: {progress_summary['completed']} of {progress_summary['total_tasks']} tasks completed. {int(completion_rate * 100)}% complete.")

            logger.info(f"Progress monitoring: {completion_rate:.1%} complete")

        except Exception as e:
            error_msg = f"Progress monitoring error: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _coordinate_agents(self, state: AgentState) -> AgentState:
        """Coordinate between agents and resolve conflicts"""
        try:
            logger.info("Coordinating agent activities")

            coordination_actions = []

            # Check for inter-agent dependencies
            progress = state.context.get("progress", {})
            if progress.get("blocked", 0) > 0:
                coordination_actions.append("resolve_blockers")

            # Check for resource conflicts
            if self._detect_resource_conflicts():
                coordination_actions.append("resolve_conflicts")

            # Rebalance workload if needed
            if self._needs_workload_rebalancing(progress):
                coordination_actions.append("rebalance_workload")

            # Execute coordination actions
            for action in coordination_actions:
                await self._execute_coordination_action(action, state)

            # Update state
            state.context["coordination"] = {
                "actions_taken": coordination_actions,
                "last_coordination": datetime.utcnow().isoformat(),
                "agents_coordinated": len(self.agent_capabilities)
            }

            # Voice feedback
            if state.mode == AgentMode.VOICE and self.livekit_client and coordination_actions:
                await self.speak(f"Coordination complete. Executed {len(coordination_actions)} coordination actions.")

            logger.info(f"Agent coordination completed: {len(coordination_actions)} actions taken")

        except Exception as e:
            error_msg = f"Agent coordination error: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _finalize_results(self, state: AgentState) -> AgentState:
        """Finalize orchestration results and provide summary"""
        try:
            logger.info("Finalizing orchestration results")

            # Compile final results
            final_results = {
                "orchestration_complete": True,
                "strategy_used": state.context.get("orchestration_plan", {}).get("strategy", "unknown"),
                "tasks_completed": state.context.get("progress", {}).get("completed", 0),
                "total_tasks": state.context.get("progress", {}).get("total_tasks", 0),
                "agents_involved": state.context.get("delegated_tasks", {}).get("agents_involved", []),
                "completion_time": datetime.utcnow().isoformat(),
                "success": len(state.errors) == 0
            }

            # Calculate success metrics
            completion_rate = final_results["tasks_completed"] / final_results["total_tasks"] if final_results["total_tasks"] > 0 else 0
            final_results["completion_rate"] = completion_rate

            # Store final results
            state.results = final_results
            state.status = "completed" if final_results["success"] else "failed"

            # Voice summary
            if state.mode == AgentMode.VOICE and self.livekit_client:
                summary = f"Orchestration complete. {int(completion_rate * 100)}% success rate with {len(final_results['agents_involved'])} agents involved."
                await self.speak(summary)

            logger.info(f"Orchestration finalized: {completion_rate:.1%} completion rate")

        except Exception as e:
            error_msg = f"Results finalization error: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    def _should_continue_monitoring(self, state: AgentState) -> str:
        """Determine if monitoring should continue"""
        progress = state.context.get("progress", {})
        completion_rate = progress.get("completed", 0) / progress.get("total_tasks", 1)
        
        if completion_rate >= 1.0:
            return "complete"
        elif progress.get("blocked", 0) > progress.get("total_tasks", 1) * 0.5:
            return "escalate"
        else:
            return "continue"

    def _assess_task_complexity(self, state: AgentState) -> TaskComplexity:
        """Assess task complexity based on analysis results"""
        # Get complexity indicators from MCP results
        serena_complexity = state.mcp_results.get("serena", {}).get("complexity_score", 0)
        thinking_steps = state.mcp_results.get("sequential_thinking", {}).get("total_steps", 0)
        
        if serena_complexity > 8 or thinking_steps > 10:
            return TaskComplexity.CRITICAL
        elif serena_complexity > 6 or thinking_steps > 7:
            return TaskComplexity.COMPLEX
        elif serena_complexity > 3 or thinking_steps > 4:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE

    def _select_orchestration_strategy(self, complexity: TaskComplexity, state: AgentState) -> OrchestrationStrategy:
        """Select appropriate orchestration strategy"""
        if complexity == TaskComplexity.CRITICAL:
            return OrchestrationStrategy.ADAPTIVE
        elif complexity == TaskComplexity.COMPLEX:
            return OrchestrationStrategy.HYBRID
        elif complexity == TaskComplexity.MODERATE:
            return OrchestrationStrategy.PARALLEL
        else:
            return OrchestrationStrategy.SEQUENTIAL

    def _decompose_tasks(self, state: AgentState) -> List[Dict[str, Any]]:
        """Decompose main request into manageable tasks"""
        # Use thinking results to create task breakdown
        thinking_steps = state.mcp_results.get("sequential_thinking", {}).get("steps", [])
        
        tasks = []
        for i, step in enumerate(thinking_steps):
            tasks.append({
                "id": f"task_{i+1}",
                "title": f"Step {i+1}: {step[:50]}...",
                "description": step,
                "priority": "high" if i < 2 else "medium",
                "estimated_duration": "2-4 hours",
                "skills_required": self._extract_skills_from_step(step)
            })

        return tasks

    def _identify_dependencies(self, state: AgentState) -> Dict[str, List[str]]:
        """Identify task dependencies"""
        # Simple sequential dependencies for now
        tasks = self._decompose_tasks(state)
        dependencies = {}
        
        for i, task in enumerate(tasks):
            if i > 0:
                dependencies[task["id"]] = [tasks[i-1]["id"]]
            else:
                dependencies[task["id"]] = []

        return dependencies

    def _allocate_resources(self, state: AgentState) -> Dict[str, str]:
        """Allocate resources (agents) to tasks"""
        tasks = self._decompose_tasks(state)
        allocation = {}
        
        for task in tasks:
            agent = self._assign_task_to_agent(task)
            allocation[task["id"]] = agent

        return allocation

    def _assign_task_to_agent(self, task: Dict[str, Any]) -> str:
        """Assign task to most suitable agent"""
        skills_required = task.get("skills_required", [])
        
        # Score agents based on capability match
        agent_scores = {}
        for agent, capabilities in self.agent_capabilities.items():
            score = len(set(skills_required) & set(capabilities))
            agent_scores[agent] = score

        # Return agent with highest score
        return max(agent_scores, key=agent_scores.get) if agent_scores else "task-executor"

    def _extract_skills_from_step(self, step: str) -> List[str]:
        """Extract required skills from thinking step"""
        skills = []
        step_lower = step.lower()
        
        if any(word in step_lower for word in ["implement", "code", "develop"]):
            skills.append("implementation")
        if any(word in step_lower for word in ["test", "verify", "validate"]):
            skills.append("testing")
        if any(word in step_lower for word in ["design", "architect", "plan"]):
            skills.append("design")
        if any(word in step_lower for word in ["check", "review", "audit"]):
            skills.append("verification")

        return skills or ["implementation"]

    def _create_timeline(self, state: AgentState) -> Dict[str, str]:
        """Create project timeline"""
        tasks = self._decompose_tasks(state)
        return {
            "total_duration": f"{len(tasks) * 3} hours",
            "start_date": datetime.utcnow().isoformat(),
            "estimated_completion": datetime.utcnow().isoformat()
        }

    def _assess_risks(self, state: AgentState) -> List[str]:
        """Assess project risks"""
        return [
            "Task dependencies may cause delays",
            "Agent availability constraints",
            "Technical complexity exceeding estimates",
            "Integration challenges between components"
        ]

    def _define_success_criteria(self, state: AgentState) -> List[str]:
        """Define success criteria"""
        return [
            "All tasks completed successfully",
            "Code quality meets standards",
            "All tests pass",
            "Documentation is complete",
            "Performance requirements met"
        ]

    def _detect_resource_conflicts(self) -> bool:
        """Detect resource conflicts between agents"""
        # Simplified conflict detection
        return False

    def _needs_workload_rebalancing(self, progress: Dict[str, int]) -> bool:
        """Check if workload rebalancing is needed"""
        return progress.get("blocked", 0) > 2

    async def _execute_coordination_action(self, action: str, state: AgentState):
        """Execute specific coordination action"""
        logger.debug(f"Executing coordination action: {action}")
        # Implementation would handle specific coordination actions

    # Voice command handlers
    async def _handle_plan_task_voice(self, command: str, state: AgentState):
        """Handle voice command for task planning"""
        await self.speak("Starting task planning based on your voice request.")
        # Extract task from voice command and initiate planning

    async def _handle_delegate_task_voice(self, command: str, state: AgentState):
        """Handle voice command for task delegation"""
        await self.speak("Delegating task to appropriate agent.")
        # Handle voice-based task delegation

    async def _handle_status_check_voice(self, command: str, state: AgentState):
        """Handle voice command for status check"""
        progress = state.context.get("progress", {})
        completed = progress.get("completed", 0)
        total = progress.get("total_tasks", 0)
        await self.speak(f"Current status: {completed} of {total} tasks completed.")

    async def _handle_emergency_stop_voice(self, command: str, state: AgentState):
        """Handle voice command for emergency stop"""
        await self.speak("Emergency stop initiated. Halting all agent activities.")
        # Implement emergency stop logic


# Factory function
def create_task_orchestrator(
    agent_id: str = "task-orchestrator",
    state_manager: Optional[StateManager] = None,
    livekit_config: Optional[LiveKitConfig] = None
) -> TaskOrchestratorAgent:
    """Factory function to create task orchestrator agent"""
    return TaskOrchestratorAgent(agent_id, state_manager, livekit_config)