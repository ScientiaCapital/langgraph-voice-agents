# Error Handling Examples

This directory contains comprehensive examples of error handling patterns for the LangGraph Voice Agent Framework.

## Examples

### error_handling_demo.py

Demonstrates 10 common error handling patterns:

1. **Basic Error Handling** - Try/except fundamentals
2. **Invalid Input Handling** - Graceful handling of malformed input
3. **Initialization Errors** - Dealing with configuration issues
4. **Timeout Handling** - Managing long-running operations
5. **Retry Logic** - Implementing exponential backoff
6. **Graceful Degradation** - Falling back when features unavailable
7. **Context Manager Pattern** - Resource cleanup with async context managers
8. **Error State Tracking** - Using agent state to track errors
9. **Multi-Agent Coordination** - Error handling across agent chains
10. **Logging and Debugging** - Enhanced debugging techniques

## Usage

```bash
python examples/error_handling/error_handling_demo.py
```

## Best Practices

### 1. Always Use Try/Except for Agent Operations

```python
try:
    result = await agent.process_input(task)
except Exception as e:
    logger.error(f"Error: {e}")
    # Handle appropriately
```

### 2. Implement Timeouts for Long Operations

```python
result = await asyncio.wait_for(
    agent.process_input(task),
    timeout=30.0
)
```

### 3. Use Retry Logic for Transient Errors

```python
for attempt in range(max_retries):
    try:
        result = await agent.process_input(task)
        break
    except ConnectionError:
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
        else:
            raise
```

### 4. Implement Graceful Degradation

```python
try:
    # Try voice-enabled mode
    agent = TaskOrchestrator(mode=AgentMode.VOICE, livekit_config=config)
except Exception:
    # Fall back to text mode
    agent = TaskOrchestrator(mode=AgentMode.TEXT)
```

### 5. Log Errors Comprehensively

```python
logger.error(f"Error type: {type(e).__name__}")
logger.error(f"Error message: {str(e)}")
logger.exception("Full traceback:")
```

## Common Error Types

### ValueError
- Invalid input format
- Empty task strings
- Missing required fields

### TypeError
- Wrong parameter types
- None where object expected

### ConnectionError
- LiveKit connection issues
- MCP tool server unavailable
- Network timeouts

### TimeoutError
- Long-running operations
- Blocked workflows
- Resource exhaustion

## Error Recovery Strategies

### Strategy 1: Fail Fast
```python
if not task:
    raise ValueError("Task cannot be empty")
```

### Strategy 2: Retry with Backoff
```python
delay = initial_delay
for attempt in range(max_retries):
    try:
        return await operation()
    except TransientError:
        await asyncio.sleep(delay)
        delay *= 2
```

### Strategy 3: Fallback to Defaults
```python
try:
    result = await preferred_method()
except Exception:
    result = await fallback_method()
```

### Strategy 4: Partial Success
```python
results = []
errors = []

for task in tasks:
    try:
        results.append(await process(task))
    except Exception as e:
        errors.append((task, e))

return results, errors
```

## Testing Error Conditions

```python
import pytest

@pytest.mark.asyncio
async def test_handles_empty_input():
    agent = TaskOrchestrator(mode=AgentMode.TEXT)

    with pytest.raises(ValueError):
        await agent.process_input("")

@pytest.mark.asyncio
async def test_timeout_handling():
    agent = TaskOrchestrator(mode=AgentMode.TEXT)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            agent.process_input(complex_task),
            timeout=0.001  # Intentionally short
        )
```

## See Also

- [API Reference](../../docs/API_REFERENCE.md) - Complete API documentation
- [Basic Demo](../basic_demo.py) - Basic usage examples
- [Tests](../../tests/) - Test suite with error scenarios

---

*Error handling examples v0.1.0 - Last updated: 2025-01-21*
