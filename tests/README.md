# Test Suite - LangGraph Voice Agent Framework

Comprehensive test suite for the voice-enabled agent framework.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
│   ├── test_core.py   # Core module tests
│   └── ...
├── integration/       # Integration tests for workflows
│   ├── test_agents.py # Agent instantiation and workflow tests
│   └── ...
├── voice/            # Voice/LiveKit integration tests
│   └── ...
├── conftest.py       # Pytest configuration and fixtures
└── README.md         # This file
```

## Running Tests

### All Tests
```bash
make test
# or
pytest tests/ -v --cov=.
```

### Unit Tests Only
```bash
make test-unit
# or
pytest tests/ -v -m "unit"
```

### Integration Tests Only
```bash
make test-integration
# or
pytest tests/ -v -m "integration"
```

### Voice Tests Only (requires LiveKit)
```bash
make test-voice
# or
pytest tests/ -v -m "voice"
```

### Fast Tests (no coverage)
```bash
make test-fast
# or
pytest tests/ -v
```

## Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.voice` - Voice/LiveKit tests (requires setup)
- `@pytest.mark.slow` - Slow tests (can be skipped)

### Running Specific Markers

```bash
# Run only unit tests
pytest -m "unit"

# Run everything except voice tests
pytest -m "not voice"

# Run fast tests only
pytest -m "not slow"
```

## Test Fixtures

Common fixtures available from `conftest.py`:

- `event_loop` - Async event loop for async tests
- `agent_mode_text` - TEXT mode for agent creation
- `sample_task` - Simple test task
- `sample_complex_task` - Complex test task for orchestration

## Writing New Tests

### Unit Test Template

```python
import pytest
from your_module import YourClass

class TestYourClass:
    """Tests for YourClass"""

    def test_creation(self):
        """Test object creation"""
        obj = YourClass()
        assert obj is not None

    def test_method(self):
        """Test a specific method"""
        obj = YourClass()
        result = obj.method()
        assert result == expected
```

### Integration Test Template

```python
import pytest
from agents import TaskOrchestrator
from core import AgentMode

@pytest.mark.integration
class TestAgentWorkflow:
    """Integration tests for agent workflow"""

    def test_workflow(self, agent_mode_text):
        """Test complete workflow"""
        agent = TaskOrchestrator(mode=agent_mode_text)
        # Test implementation
        assert True
```

### Async Test Template

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    """Test async operation"""
    result = await async_function()
    assert result is not None
```

## Coverage

Tests include coverage reporting:

```bash
# Run with coverage
pytest --cov=. --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -e ".[dev]"
    pytest tests/ -v --cov=. --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Test Requirements

### Minimal (Unit Tests)
- Python 3.9+
- Core dependencies from requirements.txt

### Integration Tests
- All unit test requirements
- Installed package (`pip install -e .`)

### Voice Tests
- All integration test requirements
- LiveKit server running
- Environment variables configured:
  - `LIVEKIT_URL`
  - `LIVEKIT_API_KEY`
  - `LIVEKIT_API_SECRET`
  - `OPENAI_API_KEY`

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'core'
```
**Solution:** Install package in development mode: `pip install -e .`

### Async Test Failures
```
RuntimeError: Event loop is closed
```
**Solution:** Ensure `pytest-asyncio` is installed and fixtures are properly configured.

### Voice Test Failures
```
ConnectionError: Failed to connect to LiveKit
```
**Solution:**
1. Ensure LiveKit server is running
2. Check environment variables
3. Verify network connectivity

## Contributing

When adding new features:

1. Write unit tests for individual components
2. Write integration tests for workflows
3. Add voice tests if feature involves LiveKit
4. Ensure tests pass: `make test`
5. Check coverage: aim for >80%

---

For more information, see the main [README.md](../README.md)
