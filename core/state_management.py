"""
Advanced state management for LangGraph agents.
Handles persistence, synchronization, and state transitions.
"""

from typing import Dict, Any, Optional, List, Type, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import json
import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum

import redis
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class StateStatus(Enum):
    """State lifecycle status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class StatePersistenceModel(Base):
    """SQLAlchemy model for state persistence"""
    __tablename__ = "agent_states"

    id = Column(String, primary_key=True)
    agent_type = Column(String, nullable=False)
    session_id = Column(String, nullable=False)
    state_data = Column(Text, nullable=False)  # JSON serialized state
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, default=1)


@dataclass
class StateMetadata:
    """Metadata for state management"""
    state_id: str
    agent_type: str
    session_id: str
    status: StateStatus
    created_at: datetime
    updated_at: datetime
    version: int = 1
    tags: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)


class StateManager(ABC):
    """Abstract base class for state management"""

    @abstractmethod
    async def save_state(
        self,
        state_id: str,
        state_data: Dict[str, Any],
        metadata: StateMetadata
    ) -> bool:
        """Save state with metadata"""
        pass

    @abstractmethod
    async def load_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Load state by ID"""
        pass

    @abstractmethod
    async def delete_state(self, state_id: str) -> bool:
        """Delete state"""
        pass

    @abstractmethod
    async def list_states(
        self,
        agent_type: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[StateStatus] = None
    ) -> List[StateMetadata]:
        """List states with optional filters"""
        pass


class DatabaseStateManager(StateManager):
    """SQLAlchemy-based state manager for persistent storage"""

    def __init__(self, database_url: str = "sqlite:///agent_states.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    async def save_state(
        self,
        state_id: str,
        state_data: Dict[str, Any],
        metadata: StateMetadata
    ) -> bool:
        """Save state to database"""
        try:
            session = self.SessionLocal()

            # Check if state exists
            existing = session.query(StatePersistenceModel).filter_by(id=state_id).first()

            if existing:
                # Update existing state
                existing.state_data = json.dumps(state_data)
                existing.status = metadata.status.value
                existing.updated_at = datetime.utcnow()
                existing.version += 1
            else:
                # Create new state
                new_state = StatePersistenceModel(
                    id=state_id,
                    agent_type=metadata.agent_type,
                    session_id=metadata.session_id,
                    state_data=json.dumps(state_data),
                    status=metadata.status.value,
                    created_at=metadata.created_at,
                    updated_at=metadata.updated_at
                )
                session.add(new_state)

            session.commit()
            session.close()

            logger.debug(f"State saved: {state_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state {state_id}: {e}")
            return False

    async def load_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Load state from database"""
        try:
            session = self.SessionLocal()
            state_model = session.query(StatePersistenceModel).filter_by(id=state_id).first()
            session.close()

            if state_model:
                return json.loads(state_model.state_data)
            return None

        except Exception as e:
            logger.error(f"Failed to load state {state_id}: {e}")
            return None

    async def delete_state(self, state_id: str) -> bool:
        """Delete state from database"""
        try:
            session = self.SessionLocal()
            state_model = session.query(StatePersistenceModel).filter_by(id=state_id).first()

            if state_model:
                session.delete(state_model)
                session.commit()

            session.close()
            logger.debug(f"State deleted: {state_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete state {state_id}: {e}")
            return False

    async def list_states(
        self,
        agent_type: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[StateStatus] = None
    ) -> List[StateMetadata]:
        """List states with filters"""
        try:
            session = self.SessionLocal()
            query = session.query(StatePersistenceModel)

            if agent_type:
                query = query.filter_by(agent_type=agent_type)
            if session_id:
                query = query.filter_by(session_id=session_id)
            if status:
                query = query.filter_by(status=status.value)

            states = query.all()
            session.close()

            return [
                StateMetadata(
                    state_id=state.id,
                    agent_type=state.agent_type,
                    session_id=state.session_id,
                    status=StateStatus(state.status),
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    version=state.version
                )
                for state in states
            ]

        except Exception as e:
            logger.error(f"Failed to list states: {e}")
            return []


class RedisStateManager(StateManager):
    """Redis-based state manager for fast in-memory access"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)

    async def save_state(
        self,
        state_id: str,
        state_data: Dict[str, Any],
        metadata: StateMetadata
    ) -> bool:
        """Save state to Redis"""
        try:
            # Save state data
            state_key = f"state:{state_id}"
            metadata_key = f"metadata:{state_id}"

            self.redis_client.hset(state_key, mapping=state_data)
            self.redis_client.hset(metadata_key, mapping=asdict(metadata))

            # Set expiration (24 hours for active states)
            expiry = 86400 if metadata.status == StateStatus.ACTIVE else 604800  # 7 days for others
            self.redis_client.expire(state_key, expiry)
            self.redis_client.expire(metadata_key, expiry)

            logger.debug(f"State saved to Redis: {state_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state to Redis {state_id}: {e}")
            return False

    async def load_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Load state from Redis"""
        try:
            state_key = f"state:{state_id}"
            state_data = self.redis_client.hgetall(state_key)

            return dict(state_data) if state_data else None

        except Exception as e:
            logger.error(f"Failed to load state from Redis {state_id}: {e}")
            return None

    async def delete_state(self, state_id: str) -> bool:
        """Delete state from Redis"""
        try:
            state_key = f"state:{state_id}"
            metadata_key = f"metadata:{state_id}"

            self.redis_client.delete(state_key, metadata_key)
            logger.debug(f"State deleted from Redis: {state_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete state from Redis {state_id}: {e}")
            return False

    async def list_states(
        self,
        agent_type: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[StateStatus] = None
    ) -> List[StateMetadata]:
        """List states from Redis"""
        try:
            metadata_keys = self.redis_client.keys("metadata:*")
            states = []

            for key in metadata_keys:
                metadata_dict = self.redis_client.hgetall(key)
                if metadata_dict:
                    metadata = StateMetadata(**metadata_dict)

                    # Apply filters
                    if agent_type and metadata.agent_type != agent_type:
                        continue
                    if session_id and metadata.session_id != session_id:
                        continue
                    if status and metadata.status != status:
                        continue

                    states.append(metadata)

            return states

        except Exception as e:
            logger.error(f"Failed to list states from Redis: {e}")
            return []


class HybridStateManager(StateManager):
    """Hybrid manager using Redis for active states and DB for persistence"""

    def __init__(
        self,
        database_url: str = "sqlite:///agent_states.db",
        redis_url: str = "redis://localhost:6379/0"
    ):
        self.db_manager = DatabaseStateManager(database_url)
        self.redis_manager = RedisStateManager(redis_url)

    async def save_state(
        self,
        state_id: str,
        state_data: Dict[str, Any],
        metadata: StateMetadata
    ) -> bool:
        """Save to both Redis and DB"""
        db_success = await self.db_manager.save_state(state_id, state_data, metadata)
        redis_success = await self.redis_manager.save_state(state_id, state_data, metadata)

        return db_success and redis_success

    async def load_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Load from Redis first, fallback to DB"""
        state = await self.redis_manager.load_state(state_id)

        if state is None:
            state = await self.db_manager.load_state(state_id)

            # Warm Redis cache if found in DB
            if state:
                metadata = StateMetadata(
                    state_id=state_id,
                    agent_type=state.get("agent_type", "unknown"),
                    session_id=state.get("session_id", "unknown"),
                    status=StateStatus.ACTIVE,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                await self.redis_manager.save_state(state_id, state, metadata)

        return state

    async def delete_state(self, state_id: str) -> bool:
        """Delete from both storages"""
        db_success = await self.db_manager.delete_state(state_id)
        redis_success = await self.redis_manager.delete_state(state_id)

        return db_success or redis_success  # Success if deleted from either

    async def list_states(
        self,
        agent_type: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[StateStatus] = None
    ) -> List[StateMetadata]:
        """List states from DB (authoritative source)"""
        return await self.db_manager.list_states(agent_type, session_id, status)


class StateTransitionManager:
    """Manages state transitions and validation"""

    VALID_TRANSITIONS = {
        StateStatus.ACTIVE: [StateStatus.PAUSED, StateStatus.COMPLETED, StateStatus.FAILED],
        StateStatus.PAUSED: [StateStatus.ACTIVE, StateStatus.COMPLETED, StateStatus.FAILED],
        StateStatus.COMPLETED: [StateStatus.ARCHIVED],
        StateStatus.FAILED: [StateStatus.ACTIVE, StateStatus.ARCHIVED],
        StateStatus.ARCHIVED: []  # Terminal state
    }

    @classmethod
    def can_transition(cls, from_status: StateStatus, to_status: StateStatus) -> bool:
        """Check if transition is valid"""
        return to_status in cls.VALID_TRANSITIONS.get(from_status, [])

    @classmethod
    def validate_transition(cls, from_status: StateStatus, to_status: StateStatus) -> None:
        """Validate transition or raise exception"""
        if not cls.can_transition(from_status, to_status):
            raise ValueError(f"Invalid transition from {from_status.value} to {to_status.value}")


class StateSynchronizer:
    """Synchronizes state across multiple agents"""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self._locks = {}  # Session-level locks

    async def acquire_lock(self, session_id: str, timeout: float = 30.0) -> bool:
        """Acquire lock for session"""
        if session_id in self._locks:
            return False

        self._locks[session_id] = asyncio.Lock()

        try:
            await asyncio.wait_for(self._locks[session_id].acquire(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            del self._locks[session_id]
            return False

    async def release_lock(self, session_id: str) -> None:
        """Release lock for session"""
        if session_id in self._locks:
            self._locks[session_id].release()
            del self._locks[session_id]

    async def synchronized_update(
        self,
        state_id: str,
        update_func,
        *args,
        **kwargs
    ) -> Any:
        """Execute update function with session lock"""
        state = await self.state_manager.load_state(state_id)
        if not state:
            raise ValueError(f"State not found: {state_id}")

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("State missing session_id")

        locked = await self.acquire_lock(session_id)
        if not locked:
            raise RuntimeError(f"Could not acquire lock for session: {session_id}")

        try:
            result = await update_func(state, *args, **kwargs)
            return result
        finally:
            await self.release_lock(session_id)


# Factory function for creating state managers

def create_state_manager(
    backend: str = "hybrid",
    database_url: str = "sqlite:///agent_states.db",
    redis_url: str = "redis://localhost:6379/0"
) -> StateManager:
    """Factory function to create appropriate state manager"""

    if backend == "database":
        return DatabaseStateManager(database_url)
    elif backend == "redis":
        return RedisStateManager(redis_url)
    elif backend == "hybrid":
        return HybridStateManager(database_url, redis_url)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Context manager for state operations

class StateContext:
    """Context manager for state operations"""

    def __init__(self, state_manager: StateManager, state_id: str):
        self.state_manager = state_manager
        self.state_id = state_id
        self.state = None

    async def __aenter__(self):
        self.state = await self.state_manager.load_state(self.state_id)
        return self.state

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.state and exc_type is None:
            # Save state if no exception occurred
            metadata = StateMetadata(
                state_id=self.state_id,
                agent_type=self.state.get("agent_type", "unknown"),
                session_id=self.state.get("session_id", "unknown"),
                status=StateStatus.ACTIVE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            await self.state_manager.save_state(self.state_id, self.state, metadata)