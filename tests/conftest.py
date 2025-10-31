"""
Pytest configuration and fixtures for the test suite
"""

import pytest
import asyncio
from typing import Generator
from core import AgentMode


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def agent_mode_text():
    """Provide TEXT agent mode for testing"""
    return AgentMode.TEXT


@pytest.fixture
def sample_task():
    """Provide a sample task for testing"""
    return "Build a REST API for user authentication"


@pytest.fixture
def sample_complex_task():
    """Provide a complex task for testing orchestration"""
    return "Design and implement a microservices architecture with API gateway, service discovery, and distributed tracing"
