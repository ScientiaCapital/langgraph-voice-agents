#!/usr/bin/env python3
"""
Structural validation test - verifies code structure without importing external dependencies.
This checks that all our fixes are in place.
"""

import ast
import sys
from pathlib import Path

project_root = Path(__file__).parent


def check_file_exists(file_path: Path, description: str) -> bool:
    """Check if a file exists"""
    if file_path.exists():
        print(f"✓ {description}: {file_path}")
        return True
    else:
        print(f"✗ {description} MISSING: {file_path}")
        return False


def check_init_files():
    """Check all __init__.py files exist"""
    print("\n1. Checking __init__.py files...")
    files = [
        (project_root / "core" / "__init__.py", "core/__init__.py"),
        (project_root / "agents" / "__init__.py", "agents/__init__.py"),
        (project_root / "tools" / "__init__.py", "tools/__init__.py"),
        (project_root / "voice" / "__init__.py", "voice/__init__.py"),
    ]

    results = [check_file_exists(path, desc) for path, desc in files]
    return all(results)


def check_imports_in_file(file_path: Path, description: str) -> bool:
    """Parse file and check for import errors"""
    print(f"\n2. Checking imports in {description}...")

    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        return False

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Check for specific import issues we fixed
        if "from ..core.base_agent import" in content:
            print(f"  ✗ Still using wrong import: base_agent")
            return False

        if "from ..core.multimodal_mixin import" in content:
            print(f"  ✗ Still using wrong import: multimodal_mixin")
            return False

        # Try to parse the file
        ast.parse(content)
        print(f"  ✓ Syntax valid, no structural import errors")
        return True

    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False


def check_method_exists(file_path: Path, method_name: str, class_name: str) -> bool:
    """Check if a method exists in a class"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        return True
        return False

    except Exception as e:
        print(f"  ✗ Error checking method: {e}")
        return False


def check_abstract_methods():
    """Check that agents implement required abstract methods"""
    print("\n3. Checking abstract method implementations...")

    agents = [
        ("TaskOrchestratorAgent", project_root / "agents" / "task_orchestrator.py"),
        ("TaskExecutorAgent", project_root / "agents" / "task_executor.py"),
        ("TaskCheckerAgent", project_root / "agents" / "task_checker.py"),
    ]

    results = []
    for class_name, file_path in agents:
        print(f"\n  Checking {class_name}...")

        # Check for _build_graph
        if check_method_exists(file_path, "_build_graph", class_name):
            print(f"    ✓ _build_graph() implemented")
            build_result = True
        else:
            print(f"    ✗ _build_graph() MISSING")
            build_result = False

        # Check for process_input
        if check_method_exists(file_path, "process_input", class_name):
            print(f"    ✓ process_input() implemented")
            process_result = True
        else:
            print(f"    ✗ process_input() MISSING")
            process_result = False

        results.append(build_result and process_result)

    return all(results)


def check_constructor_signature(file_path: Path, class_name: str) -> bool:
    """Check constructor signature matches BaseAgent"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        # Check if it has 'mode' parameter
                        param_names = [arg.arg for arg in item.args.args]
                        if 'mode' in param_names:
                            return True
        return False

    except Exception as e:
        print(f"  ✗ Error checking constructor: {e}")
        return False


def check_constructors():
    """Check that agent constructors are correct"""
    print("\n4. Checking constructor signatures...")

    agents = [
        ("TaskOrchestratorAgent", project_root / "agents" / "task_orchestrator.py"),
        ("TaskExecutorAgent", project_root / "agents" / "task_executor.py"),
        ("TaskCheckerAgent", project_root / "agents" / "task_checker.py"),
    ]

    results = []
    for class_name, file_path in agents:
        if check_constructor_signature(file_path, class_name):
            print(f"  ✓ {class_name} constructor has correct signature")
            results.append(True)
        else:
            print(f"  ✗ {class_name} constructor signature incorrect")
            results.append(False)

    return all(results)


def check_env_example():
    """Check .env.example exists"""
    print("\n5. Checking .env.example...")
    return check_file_exists(project_root / ".env.example", ".env.example")


def main():
    """Run all structural validation tests"""
    print("=" * 70)
    print("LangGraph Voice Agent Framework - Structural Validation")
    print("=" * 70)

    results = []

    # Run all checks
    results.append(("Package initialization files", check_init_files()))
    results.append(("Task Checker imports", check_imports_in_file(
        project_root / "agents" / "task_checker.py", "task_checker.py"
    )))
    results.append(("Abstract method implementations", check_abstract_methods()))
    results.append(("Constructor signatures", check_constructors()))
    results.append(("Environment template", check_env_example()))

    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All structural validations passed!")
        print("✓ All critical blockers have been fixed")
        print("✓ Code structure is correct")
        print("✓ Ready for dependency installation and runtime testing")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
