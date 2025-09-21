"""
Sequential Thinking MCP tool adapter for LangGraph agents.
Provides systematic problem decomposition and step-by-step analysis.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ThinkingStage(Enum):
    """Stages of sequential thinking process"""
    PROBLEM_DEFINITION = "Problem Definition"
    INFORMATION_GATHERING = "Information Gathering"
    RESEARCH = "Research"
    ANALYSIS = "Analysis"
    SYNTHESIS = "Synthesis"
    CONCLUSION = "Conclusion"
    CRITICAL_QUESTIONING = "Critical Questioning"
    PLANNING = "Planning"


@dataclass
class ThinkingStep:
    """Individual thinking step"""
    thought_number: int
    total_thoughts: int
    thought: str
    stage: ThinkingStage
    next_thought_needed: bool
    is_revision: bool = False
    revises_thought: Optional[int] = None
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more_thoughts: bool = False


@dataclass
class ThinkingResult:
    """Result of sequential thinking process"""
    steps: List[ThinkingStep]
    final_conclusion: str
    total_steps: int
    stages_covered: List[ThinkingStage]
    success: bool
    error_message: Optional[str] = None


class SequentialThinkingAdapter:
    """Adapter for Sequential Thinking MCP tool"""

    def __init__(self, tool_executor=None):
        self.tool_executor = tool_executor
        self._thinking_sessions = {}

    async def start_thinking_process(
        self,
        problem_statement: str,
        initial_thoughts: int = 5,
        session_id: Optional[str] = None
    ) -> str:
        """Start a new sequential thinking process"""
        try:
            session_id = session_id or f"thinking_{asyncio.get_event_loop().time()}"

            # Initialize thinking session
            self._thinking_sessions[session_id] = {
                "problem": problem_statement,
                "steps": [],
                "current_step": 1,
                "total_thoughts": initial_thoughts,
                "completed": False
            }

            logger.info(f"Started thinking process: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to start thinking process: {e}")
            raise

    async def think_step(
        self,
        session_id: str,
        thought: str,
        stage: ThinkingStage = ThinkingStage.ANALYSIS,
        is_revision: bool = False,
        revises_thought: Optional[int] = None
    ) -> ThinkingStep:
        """Execute a single thinking step"""
        try:
            if session_id not in self._thinking_sessions:
                raise ValueError(f"Thinking session not found: {session_id}")

            session = self._thinking_sessions[session_id]

            # Create thinking step
            step = ThinkingStep(
                thought_number=session["current_step"],
                total_thoughts=session["total_thoughts"],
                thought=thought,
                stage=stage,
                next_thought_needed=session["current_step"] < session["total_thoughts"],
                is_revision=is_revision,
                revises_thought=revises_thought
            )

            # Use MCP tool for actual thinking
            if self.tool_executor:
                mcp_result = await self._call_mcp_tool(step)
                step = self._parse_mcp_result(mcp_result, step)

            # Store step in session
            session["steps"].append(step)
            session["current_step"] += 1

            # Check if more thoughts are needed
            if not step.next_thought_needed:
                session["completed"] = True

            logger.debug(f"Thinking step {step.thought_number} completed")
            return step

        except Exception as e:
            logger.error(f"Thinking step failed: {e}")
            raise

    async def _call_mcp_tool(self, step: ThinkingStep) -> Dict[str, Any]:
        """Call the actual MCP Sequential Thinking tool"""
        try:
            # This would call the actual MCP tool
            # For now, we'll simulate the call
            result = {
                "thoughtNumber": step.thought_number,
                "totalThoughts": step.total_thoughts,
                "nextThoughtNeeded": step.next_thought_needed,
                "branches": [],
                "thoughtHistoryLength": step.thought_number
            }

            return result

        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            raise

    def _parse_mcp_result(self, mcp_result: Dict[str, Any], step: ThinkingStep) -> ThinkingStep:
        """Parse MCP tool result and update step"""
        try:
            # Update step with MCP result
            step.next_thought_needed = mcp_result.get("nextThoughtNeeded", False)

            # Handle branching if present
            branches = mcp_result.get("branches", [])
            if branches:
                step.branch_id = branches[0].get("id") if branches else None

            return step

        except Exception as e:
            logger.error(f"Failed to parse MCP result: {e}")
            return step

    async def complete_thinking_process(self, session_id: str) -> ThinkingResult:
        """Complete thinking process and return results"""
        try:
            if session_id not in self._thinking_sessions:
                raise ValueError(f"Thinking session not found: {session_id}")

            session = self._thinking_sessions[session_id]

            # Generate final conclusion
            final_conclusion = await self._generate_conclusion(session["steps"])

            # Create result
            result = ThinkingResult(
                steps=session["steps"],
                final_conclusion=final_conclusion,
                total_steps=len(session["steps"]),
                stages_covered=list(set(step.stage for step in session["steps"])),
                success=True
            )

            # Clean up session
            del self._thinking_sessions[session_id]

            logger.info(f"Thinking process completed: {session_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to complete thinking process: {e}")
            return ThinkingResult(
                steps=[],
                final_conclusion="",
                total_steps=0,
                stages_covered=[],
                success=False,
                error_message=str(e)
            )

    async def _generate_conclusion(self, steps: List[ThinkingStep]) -> str:
        """Generate final conclusion from thinking steps"""
        try:
            # Combine insights from all thinking steps
            insights = []

            for step in steps:
                if step.stage in [ThinkingStage.CONCLUSION, ThinkingStage.SYNTHESIS]:
                    insights.append(step.thought)

            if not insights:
                # Fallback to last few steps
                insights = [step.thought for step in steps[-2:]]

            conclusion = " ".join(insights)
            return conclusion if conclusion else "Analysis completed with systematic thinking approach."

        except Exception as e:
            logger.error(f"Failed to generate conclusion: {e}")
            return "Thinking process completed but conclusion generation failed."

    async def quick_analysis(
        self,
        problem: str,
        thinking_steps: int = 3
    ) -> ThinkingResult:
        """Perform quick analysis with automatic step progression"""
        try:
            session_id = await self.start_thinking_process(problem, thinking_steps)

            # Predefined analysis stages
            stages = [
                (ThinkingStage.PROBLEM_DEFINITION, f"Analyzing the problem: {problem}"),
                (ThinkingStage.ANALYSIS, f"Breaking down the key components and requirements"),
                (ThinkingStage.CONCLUSION, f"Synthesizing findings and recommendations")
            ]

            # Execute thinking steps
            for i, (stage, thought) in enumerate(stages[:thinking_steps]):
                await self.think_step(session_id, thought, stage)

            return await self.complete_thinking_process(session_id)

        except Exception as e:
            logger.error(f"Quick analysis failed: {e}")
            return ThinkingResult(
                steps=[],
                final_conclusion="",
                total_steps=0,
                stages_covered=[],
                success=False,
                error_message=str(e)
            )

    async def plan_implementation(
        self,
        feature_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ThinkingResult:
        """Plan implementation using systematic thinking"""
        try:
            session_id = await self.start_thinking_process(
                f"Plan implementation of: {feature_description}",
                thinking_steps=6
            )

            # Implementation planning stages
            planning_steps = [
                (ThinkingStage.PROBLEM_DEFINITION,
                 f"Understanding the feature requirements: {feature_description}"),

                (ThinkingStage.INFORMATION_GATHERING,
                 f"Gathering context and constraints: {json.dumps(context or {})}"),

                (ThinkingStage.ANALYSIS,
                 "Breaking down implementation into components and dependencies"),

                (ThinkingStage.PLANNING,
                 "Creating step-by-step implementation plan"),

                (ThinkingStage.CRITICAL_QUESTIONING,
                 "Identifying potential issues and edge cases"),

                (ThinkingStage.CONCLUSION,
                 "Finalizing implementation strategy and next steps")
            ]

            # Execute planning steps
            for stage, thought in planning_steps:
                await self.think_step(session_id, thought, stage)

            return await self.complete_thinking_process(session_id)

        except Exception as e:
            logger.error(f"Implementation planning failed: {e}")
            return ThinkingResult(
                steps=[],
                final_conclusion="",
                total_steps=0,
                stages_covered=[],
                success=False,
                error_message=str(e)
            )

    async def analyze_architecture(
        self,
        system_description: str,
        requirements: List[str]
    ) -> ThinkingResult:
        """Analyze system architecture using structured thinking"""
        try:
            session_id = await self.start_thinking_process(
                f"Architecture analysis for: {system_description}",
                thinking_steps=5
            )

            # Architecture analysis steps
            analysis_steps = [
                (ThinkingStage.PROBLEM_DEFINITION,
                 f"Understanding system requirements: {system_description}"),

                (ThinkingStage.ANALYSIS,
                 f"Analyzing requirements: {', '.join(requirements)}"),

                (ThinkingStage.PLANNING,
                 "Designing system architecture and component interactions"),

                (ThinkingStage.CRITICAL_QUESTIONING,
                 "Evaluating scalability, security, and maintainability concerns"),

                (ThinkingStage.CONCLUSION,
                 "Recommending final architecture and implementation approach")
            ]

            # Execute analysis steps
            for stage, thought in analysis_steps:
                await self.think_step(session_id, thought, stage)

            return await self.complete_thinking_process(session_id)

        except Exception as e:
            logger.error(f"Architecture analysis failed: {e}")
            return ThinkingResult(
                steps=[],
                final_conclusion="",
                total_steps=0,
                stages_covered=[],
                success=False,
                error_message=str(e)
            )

    def get_active_sessions(self) -> List[str]:
        """Get list of active thinking sessions"""
        return list(self._thinking_sessions.keys())

    async def abandon_session(self, session_id: str) -> bool:
        """Abandon a thinking session"""
        try:
            if session_id in self._thinking_sessions:
                del self._thinking_sessions[session_id]
                logger.info(f"Abandoned thinking session: {session_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to abandon session: {e}")
            return False


# Utility functions for integration with agents

async def think_about_task(
    adapter: SequentialThinkingAdapter,
    task_description: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """High-level function for task analysis"""
    try:
        result = await adapter.plan_implementation(task_description, context)

        return {
            "success": result.success,
            "analysis": result.final_conclusion,
            "steps": [step.thought for step in result.steps],
            "stages": [stage.value for stage in result.stages_covered],
            "recommendations": result.final_conclusion
        }

    except Exception as e:
        logger.error(f"Task thinking failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "analysis": "",
            "steps": [],
            "stages": [],
            "recommendations": ""
        }


async def analyze_problem(
    adapter: SequentialThinkingAdapter,
    problem: str,
    depth: int = 3
) -> Dict[str, Any]:
    """Analyze a problem systematically"""
    try:
        result = await adapter.quick_analysis(problem, depth)

        return {
            "success": result.success,
            "problem": problem,
            "analysis": result.final_conclusion,
            "thought_process": [
                {
                    "step": step.thought_number,
                    "stage": step.stage.value,
                    "insight": step.thought
                }
                for step in result.steps
            ],
            "total_steps": result.total_steps
        }

    except Exception as e:
        logger.error(f"Problem analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "problem": problem,
            "analysis": "",
            "thought_process": [],
            "total_steps": 0
        }


# Create default adapter instance
default_sequential_thinking = SequentialThinkingAdapter()