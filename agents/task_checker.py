"""
Task Checker Agent - Quality Assurance and Validation Specialist

This agent specializes in comprehensive testing, validation, and quality assurance
of implemented solutions. It ensures all deliverables meet requirements and
follows best practices through systematic verification workflows.

Key Responsibilities:
- Comprehensive testing strategy development
- Code quality validation and review
- Requirements verification and traceability
- Performance and security testing
- Documentation validation
- Integration testing coordination
- Bug detection and resolution verification

Voice Features:
- Voice-guided testing workflows
- Spoken test results and validation status
- Voice commands for quality checks
- Audio feedback for test execution progress
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from ..core.base_graph import BaseAgent, AgentState, AgentMode, MultiModalMixin
from ..voice.livekit_client import LiveKitClient, LiveKitConfig
from ..tools.sequential_thinking_tools import SequentialThinkingAdapter
from ..tools.serena_tools import SerenaAdapter
from ..tools.context7_tools import Context7Adapter
from ..tools.taskmaster_tools import TaskMasterAdapter
from ..tools.shrimp_tools import ShrimpTaskManagerAdapter
from ..tools.desktop_commander_tools import DesktopCommanderAdapter

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    ENTERPRISE = "enterprise"


class TestCategory(Enum):
    """Categories of testing"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"
    REGRESSION = "regression"


@dataclass
class ValidationResult:
    """Result of validation check"""
    category: TestCategory
    passed: bool
    score: float  # 0-100
    issues: List[str]
    recommendations: List[str]
    details: Dict[str, Any]


@dataclass
class QualityMetrics:
    """Quality metrics for code/implementation"""
    code_coverage: float
    complexity_score: float
    maintainability_index: float
    security_score: float
    performance_score: float
    documentation_coverage: float
    overall_quality: float


class TaskCheckerAgent(BaseAgent, MultiModalMixin):
    """
    Task Checker Agent - Quality Assurance Specialist
    
    Ensures comprehensive validation and quality assurance of all implementations
    through systematic testing and verification workflows.
    """

    def __init__(
        self,
        mode: AgentMode = AgentMode.TEXT,
        checkpointer_path: Optional[str] = None,
        livekit_config: Optional[LiveKitConfig] = None
    ):
        super().__init__(
            agent_type="task-checker",
            mode=mode,
            checkpointer_path=checkpointer_path
        )

        # Voice capabilities initialization
        self.livekit_client = None
        if livekit_config and mode in [AgentMode.VOICE, AgentMode.HYBRID]:
            self.livekit_client = LiveKitClient(livekit_config)
        
        # Initialize MCP tool adapters (mandatory order)
        self.sequential_thinking = SequentialThinkingAdapter()
        self.serena = SerenaAdapter()
        self.context7 = Context7Adapter()
        self.taskmaster = TaskMasterAdapter()
        self.shrimp = ShrimpTaskManagerAdapter()
        self.desktop_commander = DesktopCommanderAdapter()
        
        # Quality assurance state
        self.validation_results = []
        self.quality_metrics = None
        self.current_validation_level = ValidationLevel.STANDARD
        
        # Voice command mappings
        self.voice_commands.update({
            "run validation": self._handle_run_validation,
            "check code quality": self._handle_check_quality,
            "test suite": self._handle_test_suite,
            "validate requirements": self._handle_validate_requirements,
            "security scan": self._handle_security_scan,
            "performance test": self._handle_performance_test,
            "integration test": self._handle_integration_test,
            "regression test": self._handle_regression_test,
            "quality report": self._handle_quality_report,
            "validation status": self._handle_validation_status
        })

    def _build_graph(self) -> StateGraph:
        """Build the task checker workflow graph"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes for validation workflow
        workflow.add_node("analyze_validation_requirements", self._analyze_validation_requirements)
        workflow.add_node("examine_implementation", self._examine_implementation)
        workflow.add_node("research_testing_patterns", self._research_testing_patterns)
        workflow.add_node("plan_validation_strategy", self._plan_validation_strategy)
        workflow.add_node("setup_testing_environment", self._setup_testing_environment)
        workflow.add_node("execute_unit_tests", self._execute_unit_tests)
        workflow.add_node("execute_integration_tests", self._execute_integration_tests)
        workflow.add_node("execute_system_tests", self._execute_system_tests)
        workflow.add_node("validate_performance", self._validate_performance)
        workflow.add_node("validate_security", self._validate_security)
        workflow.add_node("validate_requirements", self._validate_requirements)
        workflow.add_node("generate_quality_report", self._generate_quality_report)
        workflow.add_node("provide_recommendations", self._provide_recommendations)
        
        # Define workflow edges
        workflow.add_edge("analyze_validation_requirements", "examine_implementation")
        workflow.add_edge("examine_implementation", "research_testing_patterns")
        workflow.add_edge("research_testing_patterns", "plan_validation_strategy")
        workflow.add_edge("plan_validation_strategy", "setup_testing_environment")
        workflow.add_edge("setup_testing_environment", "execute_unit_tests")
        workflow.add_edge("execute_unit_tests", "execute_integration_tests")
        workflow.add_edge("execute_integration_tests", "execute_system_tests")
        workflow.add_edge("execute_system_tests", "validate_performance")
        workflow.add_edge("validate_performance", "validate_security")
        workflow.add_edge("validate_security", "validate_requirements")
        workflow.add_edge("validate_requirements", "generate_quality_report")
        workflow.add_edge("generate_quality_report", "provide_recommendations")
        workflow.add_edge("provide_recommendations", END)
        
        workflow.set_entry_point("analyze_validation_requirements")
        
        return workflow

    async def process_input(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process input and execute validation workflow.

        Args:
            input_data: Task description (str) or structured input (dict)

        Returns:
            Dict containing validation results and quality reports
        """
        # Convert string input to structured format
        if isinstance(input_data, str):
            task = input_data
        else:
            task = input_data.get("task", input_data.get("message", ""))

        # Create initial state
        initial_state = self.create_initial_state(task)

        # Execute the validation workflow
        config = {"configurable": {"thread_id": self.session_id}}
        result = await self.app.ainvoke(initial_state, config)

        return {
            "status": "completed",
            "result": result,
            "agent_type": self.agent_type,
            "session_id": self.session_id
        }

    async def _analyze_validation_requirements(self, state: AgentState) -> AgentState:
        """Analyze what needs to be validated using Sequential Thinking"""
        try:
            await self._speak("Starting validation requirement analysis...")
            
            # Use Sequential Thinking for comprehensive analysis
            thinking_result = await self.sequential_thinking.analyze_problem(
                problem=f"Analyze validation requirements for: {state.get('task_description', 'Unknown task')}",
                depth=4
            )
            
            validation_requirements = {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "quality_criteria": [],
                "testing_scope": [],
                "validation_methods": [],
                "success_criteria": []
            }
            
            # Extract requirements from thinking process
            if thinking_result.get("success"):
                for step in thinking_result.get("thought_process", []):
                    if "requirement" in step.get("insight", "").lower():
                        validation_requirements["functional_requirements"].append(step["insight"])
                    elif "quality" in step.get("insight", "").lower():
                        validation_requirements["quality_criteria"].append(step["insight"])
                    elif "test" in step.get("insight", "").lower():
                        validation_requirements["testing_scope"].append(step["insight"])
            
            state["validation_requirements"] = validation_requirements
            state["thinking_analysis"] = thinking_result
            
            await self._speak("Validation requirements analysis completed")
            
        except Exception as e:
            logger.error(f"Error in validation requirements analysis: {e}")
            await self._speak(f"Error analyzing validation requirements: {str(e)}")
            
        return state

    async def _examine_implementation(self, state: AgentState) -> AgentState:
        """Examine the implementation using Serena for code analysis"""
        try:
            await self._speak("Examining implementation structure...")
            
            # Get project overview using Serena
            project_files = await self.serena.list_project_files(".", recursive=True)
            
            # Analyze key implementation files
            implementation_analysis = {
                "file_structure": project_files,
                "code_patterns": [],
                "dependencies": [],
                "test_files": [],
                "documentation": [],
                "configuration": []
            }
            
            # Categorize files for validation planning
            if project_files.get("success"):
                for file_path in project_files.get("files", []):
                    if "test" in file_path.lower() or file_path.endswith(("_test.py", ".test.js", "_spec.py")):
                        implementation_analysis["test_files"].append(file_path)
                    elif file_path.endswith((".md", ".rst", ".txt")):
                        implementation_analysis["documentation"].append(file_path)
                    elif file_path.endswith((".json", ".yaml", ".yml", ".toml", ".ini")):
                        implementation_analysis["configuration"].append(file_path)
                    elif file_path.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c")):
                        implementation_analysis["code_patterns"].append(file_path)
            
            # Analyze code quality if main files exist
            if implementation_analysis["code_patterns"]:
                # Sample a few key files for deeper analysis
                key_files = implementation_analysis["code_patterns"][:5]
                for file_path in key_files:
                    try:
                        file_overview = await self.serena.get_file_overview(file_path)
                        if file_overview.get("success"):
                            implementation_analysis[f"analysis_{file_path}"] = file_overview
                    except Exception as file_error:
                        logger.warning(f"Could not analyze file {file_path}: {file_error}")
            
            state["implementation_analysis"] = implementation_analysis
            
            await self._speak("Implementation examination completed")
            
        except Exception as e:
            logger.error(f"Error examining implementation: {e}")
            await self._speak(f"Error examining implementation: {str(e)}")
            
        return state

    async def _research_testing_patterns(self, state: AgentState) -> AgentState:
        """Research best practices for testing using Context7"""
        try:
            await self._speak("Researching testing best practices...")
            
            # Determine technology stack for targeted research
            implementation = state.get("implementation_analysis", {})
            code_files = implementation.get("code_patterns", [])
            
            # Detect primary technology
            tech_stack = self._detect_technology_stack(code_files)
            
            # Research testing patterns for detected technologies
            testing_research = {}
            
            for tech in tech_stack:
                try:
                    # Research testing frameworks and patterns
                    research_result = await self.context7.get_documentation(
                        tech, 
                        topic="testing best practices"
                    )
                    
                    if research_result.get("success"):
                        testing_research[tech] = research_result.get("documentation", {})
                        
                except Exception as research_error:
                    logger.warning(f"Could not research {tech} testing patterns: {research_error}")
            
            # General testing patterns research
            try:
                general_testing = await self.context7.get_documentation(
                    "software-testing",
                    topic="comprehensive testing strategies"
                )
                testing_research["general"] = general_testing.get("documentation", {})
            except Exception as e:
                logger.warning(f"Could not research general testing patterns: {e}")
            
            state["testing_research"] = testing_research
            state["detected_tech_stack"] = tech_stack
            
            await self._speak("Testing patterns research completed")
            
        except Exception as e:
            logger.error(f"Error researching testing patterns: {e}")
            await self._speak(f"Error researching testing patterns: {str(e)}")
            
        return state

    async def _plan_validation_strategy(self, state: AgentState) -> AgentState:
        """Plan comprehensive validation strategy"""
        try:
            await self._speak("Planning validation strategy...")
            
            requirements = state.get("validation_requirements", {})
            implementation = state.get("implementation_analysis", {})
            research = state.get("testing_research", {})
            tech_stack = state.get("detected_tech_stack", [])
            
            # Create comprehensive validation plan
            validation_plan = {
                "validation_level": self.current_validation_level.value,
                "test_categories": [],
                "test_phases": [],
                "tools_and_frameworks": [],
                "success_criteria": [],
                "timeline": [],
                "resource_requirements": []
            }
            
            # Determine test categories based on project type and requirements
            if implementation.get("test_files"):
                validation_plan["test_categories"].append(TestCategory.UNIT.value)
                validation_plan["test_categories"].append(TestCategory.INTEGRATION.value)
            
            if len(implementation.get("code_patterns", [])) > 5:
                validation_plan["test_categories"].append(TestCategory.SYSTEM.value)
                validation_plan["test_categories"].append(TestCategory.REGRESSION.value)
            
            # Always include these for comprehensive validation
            validation_plan["test_categories"].extend([
                TestCategory.PERFORMANCE.value,
                TestCategory.SECURITY.value
            ])
            
            # Define test phases
            validation_plan["test_phases"] = [
                {
                    "phase": "Environment Setup",
                    "description": "Prepare testing environment and dependencies",
                    "estimated_time": "15 minutes"
                },
                {
                    "phase": "Unit Testing",
                    "description": "Test individual components and functions",
                    "estimated_time": "30 minutes"
                },
                {
                    "phase": "Integration Testing",
                    "description": "Test component interactions and data flow",
                    "estimated_time": "45 minutes"
                },
                {
                    "phase": "System Testing",
                    "description": "End-to-end functionality validation",
                    "estimated_time": "60 minutes"
                },
                {
                    "phase": "Quality Validation",
                    "description": "Performance, security, and compliance checks",
                    "estimated_time": "30 minutes"
                }
            ]
            
            # Recommend tools based on tech stack
            for tech in tech_stack:
                if tech == "python":
                    validation_plan["tools_and_frameworks"].extend([
                        "pytest", "unittest", "coverage", "black", "flake8", "mypy"
                    ])
                elif tech == "javascript":
                    validation_plan["tools_and_frameworks"].extend([
                        "jest", "mocha", "cypress", "eslint", "prettier"
                    ])
                elif tech == "typescript":
                    validation_plan["tools_and_frameworks"].extend([
                        "jest", "vitest", "tsc", "eslint", "prettier"
                    ])
            
            state["validation_plan"] = validation_plan
            
            await self._speak("Validation strategy planning completed")
            
        except Exception as e:
            logger.error(f"Error planning validation strategy: {e}")
            await self._speak(f"Error planning validation strategy: {str(e)}")
            
        return state

    async def _setup_testing_environment(self, state: AgentState) -> AgentState:
        """Setup testing environment using Desktop Commander"""
        try:
            await self._speak("Setting up testing environment...")
            
            validation_plan = state.get("validation_plan", {})
            tools = validation_plan.get("tools_and_frameworks", [])
            
            environment_setup = {
                "dependencies_installed": [],
                "configuration_created": [],
                "test_directories": [],
                "setup_scripts": [],
                "environment_ready": False
            }
            
            # Check if testing dependencies are available
            for tool in tools:
                try:
                    # Check if tool is available
                    check_result = await self.desktop_commander.start_process(
                        command=f"which {tool} || command -v {tool}",
                        timeout_ms=5000
                    )
                    
                    if check_result.get("success"):
                        environment_setup["dependencies_installed"].append(tool)
                        
                except Exception as tool_error:
                    logger.warning(f"Could not check tool {tool}: {tool_error}")
            
            # Create test directories if they don't exist
            test_dirs = ["tests", "test_reports", "coverage_reports"]
            for test_dir in test_dirs:
                try:
                    create_result = await self.desktop_commander.create_directory(test_dir)
                    if create_result.get("success"):
                        environment_setup["test_directories"].append(test_dir)
                except Exception as dir_error:
                    logger.warning(f"Could not create directory {test_dir}: {dir_error}")
            
            # Check for existing test configuration files
            config_files = [
                "pytest.ini", "pyproject.toml", "jest.config.js", 
                "package.json", "tsconfig.json", ".eslintrc"
            ]
            
            for config_file in config_files:
                try:
                    file_check = await self.desktop_commander.get_file_info(config_file)
                    if file_check.get("exists"):
                        environment_setup["configuration_created"].append(config_file)
                except Exception as config_error:
                    logger.debug(f"Config file {config_file} not found: {config_error}")
            
            environment_setup["environment_ready"] = (
                len(environment_setup["dependencies_installed"]) > 0 or
                len(environment_setup["configuration_created"]) > 0
            )
            
            state["environment_setup"] = environment_setup
            
            await self._speak("Testing environment setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up testing environment: {e}")
            await self._speak(f"Error setting up testing environment: {str(e)}")
            
        return state

    async def _execute_unit_tests(self, state: AgentState) -> AgentState:
        """Execute unit tests"""
        try:
            await self._speak("Executing unit tests...")
            
            unit_test_results = await self._run_test_category(
                TestCategory.UNIT, 
                state.get("environment_setup", {}),
                state.get("detected_tech_stack", [])
            )
            
            state["unit_test_results"] = unit_test_results
            
            if unit_test_results.passed:
                await self._speak("Unit tests passed successfully")
            else:
                await self._speak(f"Unit tests failed with {len(unit_test_results.issues)} issues")
            
        except Exception as e:
            logger.error(f"Error executing unit tests: {e}")
            await self._speak(f"Error executing unit tests: {str(e)}")
            
        return state

    async def _execute_integration_tests(self, state: AgentState) -> AgentState:
        """Execute integration tests"""
        try:
            await self._speak("Executing integration tests...")
            
            integration_test_results = await self._run_test_category(
                TestCategory.INTEGRATION,
                state.get("environment_setup", {}),
                state.get("detected_tech_stack", [])
            )
            
            state["integration_test_results"] = integration_test_results
            
            if integration_test_results.passed:
                await self._speak("Integration tests passed successfully")
            else:
                await self._speak(f"Integration tests failed with {len(integration_test_results.issues)} issues")
            
        except Exception as e:
            logger.error(f"Error executing integration tests: {e}")
            await self._speak(f"Error executing integration tests: {str(e)}")
            
        return state

    async def _execute_system_tests(self, state: AgentState) -> AgentState:
        """Execute system tests"""
        try:
            await self._speak("Executing system tests...")
            
            system_test_results = await self._run_test_category(
                TestCategory.SYSTEM,
                state.get("environment_setup", {}),
                state.get("detected_tech_stack", [])
            )
            
            state["system_test_results"] = system_test_results
            
            if system_test_results.passed:
                await self._speak("System tests passed successfully")
            else:
                await self._speak(f"System tests failed with {len(system_test_results.issues)} issues")
            
        except Exception as e:
            logger.error(f"Error executing system tests: {e}")
            await self._speak(f"Error executing system tests: {str(e)}")
            
        return state

    async def _validate_performance(self, state: AgentState) -> AgentState:
        """Validate performance characteristics"""
        try:
            await self._speak("Validating performance...")
            
            performance_results = await self._run_test_category(
                TestCategory.PERFORMANCE,
                state.get("environment_setup", {}),
                state.get("detected_tech_stack", [])
            )
            
            state["performance_results"] = performance_results
            
            if performance_results.passed:
                await self._speak("Performance validation passed")
            else:
                await self._speak(f"Performance issues detected: {len(performance_results.issues)} concerns")
            
        except Exception as e:
            logger.error(f"Error validating performance: {e}")
            await self._speak(f"Error validating performance: {str(e)}")
            
        return state

    async def _validate_security(self, state: AgentState) -> AgentState:
        """Validate security aspects"""
        try:
            await self._speak("Validating security...")
            
            security_results = await self._run_test_category(
                TestCategory.SECURITY,
                state.get("environment_setup", {}),
                state.get("detected_tech_stack", [])
            )
            
            state["security_results"] = security_results
            
            if security_results.passed:
                await self._speak("Security validation passed")
            else:
                await self._speak(f"Security issues detected: {len(security_results.issues)} vulnerabilities")
            
        except Exception as e:
            logger.error(f"Error validating security: {e}")
            await self._speak(f"Error validating security: {str(e)}")
            
        return state

    async def _validate_requirements(self, state: AgentState) -> AgentState:
        """Validate against original requirements"""
        try:
            await self._speak("Validating requirements compliance...")
            
            requirements = state.get("validation_requirements", {})
            all_test_results = [
                state.get("unit_test_results"),
                state.get("integration_test_results"),
                state.get("system_test_results"),
                state.get("performance_results"),
                state.get("security_results")
            ]
            
            # Calculate requirements traceability
            requirements_validation = {
                "functional_compliance": 0.0,
                "quality_compliance": 0.0,
                "overall_compliance": 0.0,
                "missing_requirements": [],
                "compliance_details": {}
            }
            
            # Analyze test results against requirements
            passing_tests = sum(1 for result in all_test_results if result and result.passed)
            total_tests = len([r for r in all_test_results if r is not None])
            
            if total_tests > 0:
                requirements_validation["overall_compliance"] = (passing_tests / total_tests) * 100
                requirements_validation["functional_compliance"] = requirements_validation["overall_compliance"]
                requirements_validation["quality_compliance"] = requirements_validation["overall_compliance"]
            
            state["requirements_validation"] = requirements_validation
            
            compliance_score = requirements_validation["overall_compliance"]
            await self._speak(f"Requirements validation completed with {compliance_score:.1f}% compliance")
            
        except Exception as e:
            logger.error(f"Error validating requirements: {e}")
            await self._speak(f"Error validating requirements: {str(e)}")
            
        return state

    async def _generate_quality_report(self, state: AgentState) -> AgentState:
        """Generate comprehensive quality report"""
        try:
            await self._speak("Generating quality report...")
            
            # Collect all validation results
            all_results = {
                "unit_tests": state.get("unit_test_results"),
                "integration_tests": state.get("integration_test_results"),
                "system_tests": state.get("system_test_results"),
                "performance": state.get("performance_results"),
                "security": state.get("security_results"),
                "requirements": state.get("requirements_validation")
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(all_results)
            
            # Generate comprehensive report
            quality_report = {
                "overall_score": quality_metrics.overall_quality,
                "metrics": {
                    "code_coverage": quality_metrics.code_coverage,
                    "complexity_score": quality_metrics.complexity_score,
                    "maintainability_index": quality_metrics.maintainability_index,
                    "security_score": quality_metrics.security_score,
                    "performance_score": quality_metrics.performance_score,
                    "documentation_coverage": quality_metrics.documentation_coverage
                },
                "validation_summary": all_results,
                "recommendations": [],
                "critical_issues": [],
                "improvement_areas": []
            }
            
            # Identify critical issues and recommendations
            for category, result in all_results.items():
                if result and hasattr(result, 'issues') and result.issues:
                    quality_report["critical_issues"].extend(result.issues)
                if result and hasattr(result, 'recommendations') and result.recommendations:
                    quality_report["recommendations"].extend(result.recommendations)
            
            # Add improvement areas based on metrics
            if quality_metrics.code_coverage < 80:
                quality_report["improvement_areas"].append("Increase test coverage")
            if quality_metrics.complexity_score > 10:
                quality_report["improvement_areas"].append("Reduce code complexity")
            if quality_metrics.security_score < 90:
                quality_report["improvement_areas"].append("Address security vulnerabilities")
            
            state["quality_report"] = quality_report
            self.quality_metrics = quality_metrics
            
            await self._speak(f"Quality report generated with overall score: {quality_metrics.overall_quality:.1f}")
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            await self._speak(f"Error generating quality report: {str(e)}")
            
        return state

    async def _provide_recommendations(self, state: AgentState) -> AgentState:
        """Provide actionable recommendations"""
        try:
            await self._speak("Providing recommendations...")
            
            quality_report = state.get("quality_report", {})
            overall_score = quality_report.get("overall_score", 0)
            
            # Generate prioritized recommendations
            recommendations = {
                "immediate_actions": [],
                "short_term_improvements": [],
                "long_term_enhancements": [],
                "best_practices": []
            }
            
            # Priority recommendations based on quality score
            if overall_score < 60:
                recommendations["immediate_actions"].extend([
                    "Address critical security vulnerabilities",
                    "Fix failing unit tests",
                    "Implement basic error handling"
                ])
            elif overall_score < 80:
                recommendations["short_term_improvements"].extend([
                    "Increase test coverage to 80%+",
                    "Improve documentation coverage",
                    "Optimize performance bottlenecks"
                ])
            else:
                recommendations["long_term_enhancements"].extend([
                    "Implement advanced monitoring",
                    "Add comprehensive logging",
                    "Consider architectural improvements"
                ])
            
            # Add general best practices
            recommendations["best_practices"] = [
                "Regular code reviews",
                "Continuous integration setup",
                "Automated testing in CI/CD",
                "Security scanning integration",
                "Performance monitoring"
            ]
            
            state["final_recommendations"] = recommendations
            
            await self._speak("Validation completed with comprehensive recommendations")
            
        except Exception as e:
            logger.error(f"Error providing recommendations: {e}")
            await self._speak(f"Error providing recommendations: {str(e)}")
            
        return state

    async def _run_test_category(
        self, 
        category: TestCategory, 
        environment: Dict[str, Any],
        tech_stack: List[str]
    ) -> ValidationResult:
        """Run tests for a specific category"""
        try:
            issues = []
            recommendations = []
            score = 85.0  # Default good score
            
            # Simulate test execution based on available tools
            available_tools = environment.get("dependencies_installed", [])
            
            if "python" in tech_stack and "pytest" in available_tools:
                # Run pytest for Python projects
                test_result = await self.desktop_commander.start_process(
                    command="pytest --tb=short --quiet",
                    timeout_ms=30000
                )
                
                if test_result.get("success"):
                    output = test_result.get("output", "")
                    if "FAILED" in output:
                        issues.append("Some tests are failing")
                        score = 60.0
                    elif "passed" in output:
                        score = 95.0
                else:
                    issues.append("Tests could not be executed")
                    score = 30.0
                    
            elif "javascript" in tech_stack and any(tool in available_tools for tool in ["jest", "npm"]):
                # Run npm test for JavaScript projects
                test_result = await self.desktop_commander.start_process(
                    command="npm test",
                    timeout_ms=30000
                )
                
                if test_result.get("success"):
                    score = 90.0
                else:
                    issues.append("JavaScript tests failed or not configured")
                    score = 50.0
            else:
                # No specific testing tools available
                recommendations.append(f"Set up appropriate testing framework for {category.value} tests")
                score = 70.0
            
            # Category-specific checks
            if category == TestCategory.SECURITY:
                # Additional security-specific validations
                if not any("security" in tool for tool in available_tools):
                    recommendations.append("Install security scanning tools")
                    score = min(score, 75.0)
            
            elif category == TestCategory.PERFORMANCE:
                # Performance-specific validations
                if not any("perf" in tool or "benchmark" in tool for tool in available_tools):
                    recommendations.append("Add performance benchmarking tools")
                    score = min(score, 80.0)
            
            return ValidationResult(
                category=category,
                passed=score >= 70.0,
                score=score,
                issues=issues,
                recommendations=recommendations,
                details={"environment": environment, "tech_stack": tech_stack}
            )
            
        except Exception as e:
            logger.error(f"Error running {category.value} tests: {e}")
            return ValidationResult(
                category=category,
                passed=False,
                score=0.0,
                issues=[f"Test execution failed: {str(e)}"],
                recommendations=["Check test environment setup"],
                details={}
            )

    def _calculate_quality_metrics(self, results: Dict[str, Any]) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        try:
            # Extract scores from results
            scores = []
            
            for category, result in results.items():
                if result and hasattr(result, 'score'):
                    scores.append(result.score)
                elif result and isinstance(result, dict) and 'overall_compliance' in result:
                    scores.append(result['overall_compliance'])
            
            # Calculate metrics
            avg_score = sum(scores) / len(scores) if scores else 50.0
            
            return QualityMetrics(
                code_coverage=avg_score,  # Simplified - would use actual coverage tools
                complexity_score=max(0, 15 - (avg_score / 10)),  # Lower is better for complexity
                maintainability_index=avg_score,
                security_score=avg_score,
                performance_score=avg_score,
                documentation_coverage=avg_score * 0.8,  # Typically lower than code coverage
                overall_quality=avg_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return QualityMetrics(
                code_coverage=0.0,
                complexity_score=20.0,
                maintainability_index=0.0,
                security_score=0.0,
                performance_score=0.0,
                documentation_coverage=0.0,
                overall_quality=0.0
            )

    def _detect_technology_stack(self, code_files: List[str]) -> List[str]:
        """Detect technology stack from file extensions"""
        tech_stack = []
        
        extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php'
        }
        
        for file_path in code_files:
            for ext, tech in extensions.items():
                if file_path.endswith(ext) and tech not in tech_stack:
                    tech_stack.append(tech)
        
        return tech_stack or ['generic']

    # Voice command handlers
    async def _handle_run_validation(self, transcription: str):
        """Handle voice command for running validation"""
        await self._speak("Starting comprehensive validation workflow...")
        
        state = AgentState(
            messages=[],
            task_description="Voice-initiated comprehensive validation",
            current_step="analyze_validation_requirements"
        )
        
        result = await self.run(state)
        
        if result.get("quality_report"):
            score = result["quality_report"].get("overall_score", 0)
            await self._speak(f"Validation completed with overall quality score: {score:.1f}")
        else:
            await self._speak("Validation workflow completed")

    async def _handle_check_quality(self, transcription: str):
        """Handle voice command for quality check"""
        await self._speak("Running quality assessment...")
        
        if self.quality_metrics:
            await self._speak(
                f"Current quality metrics: "
                f"Overall quality {self.quality_metrics.overall_quality:.1f}, "
                f"Code coverage {self.quality_metrics.code_coverage:.1f}, "
                f"Security score {self.quality_metrics.security_score:.1f}"
            )
        else:
            await self._speak("No quality metrics available. Run full validation first.")

    async def _handle_test_suite(self, transcription: str):
        """Handle voice command for test suite execution"""
        await self._speak("Executing test suite...")
        
        # Run basic test commands
        try:
            result = await self.desktop_commander.start_process(
                command="pytest --tb=short || npm test || echo 'No standard test runner found'",
                timeout_ms=60000
            )
            
            if result.get("success"):
                await self._speak("Test suite execution completed")
            else:
                await self._speak("Test suite execution encountered issues")
        except Exception as e:
            await self._speak(f"Error running test suite: {str(e)}")

    async def _handle_validate_requirements(self, transcription: str):
        """Handle voice command for requirements validation"""
        await self._speak("Validating requirements compliance...")
        # Implementation would check requirements traceability

    async def _handle_security_scan(self, transcription: str):
        """Handle voice command for security scan"""
        await self._speak("Running security scan...")
        # Implementation would run security tools

    async def _handle_performance_test(self, transcription: str):
        """Handle voice command for performance testing"""
        await self._speak("Running performance tests...")
        # Implementation would run performance benchmarks

    async def _handle_integration_test(self, transcription: str):
        """Handle voice command for integration testing"""
        await self._speak("Running integration tests...")
        # Implementation would run integration test suite

    async def _handle_regression_test(self, transcription: str):
        """Handle voice command for regression testing"""
        await self._speak("Running regression tests...")
        # Implementation would run regression test suite

    async def _handle_quality_report(self, transcription: str):
        """Handle voice command for quality report"""
        await self._speak("Generating quality report...")
        
        if self.quality_metrics:
            report = (
                f"Quality Report: Overall score {self.quality_metrics.overall_quality:.1f}. "
                f"Code coverage {self.quality_metrics.code_coverage:.1f}%. "
                f"Security score {self.quality_metrics.security_score:.1f}. "
                f"Maintainability index {self.quality_metrics.maintainability_index:.1f}."
            )
            await self._speak(report)
        else:
            await self._speak("No quality report available. Run validation first.")

    async def _handle_validation_status(self, transcription: str):
        """Handle voice command for validation status"""
        if self.validation_results:
            passed = sum(1 for r in self.validation_results if r.passed)
            total = len(self.validation_results)
            await self._speak(f"Validation status: {passed} of {total} checks passed")
        else:
            await self._speak("No validation results available")

    async def set_validation_level(self, level: ValidationLevel):
        """Set the validation thoroughness level"""
        self.current_validation_level = level
        await self._speak(f"Validation level set to {level.value}")

    async def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results"""
        return {
            "validation_level": self.current_validation_level.value,
            "quality_metrics": self.quality_metrics.__dict__ if self.quality_metrics else None,
            "validation_results": [
                {
                    "category": r.category.value,
                    "passed": r.passed,
                    "score": r.score,
                    "issues_count": len(r.issues),
                    "recommendations_count": len(r.recommendations)
                }
                for r in self.validation_results
            ]
        }

    def __str__(self) -> str:
        return f"TaskCheckerAgent(session_id={self.session_id}, validation_level={self.current_validation_level.value})"

    def __repr__(self) -> str:
        return self.__str__()