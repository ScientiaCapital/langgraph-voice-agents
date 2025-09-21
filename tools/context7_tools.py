"""
Context7 MCP tool adapter for LangGraph agents.
Provides up-to-date documentation and library reference capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


class DocumentationType(Enum):
    """Types of documentation available"""
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    GUIDE = "guide"
    EXAMPLE = "example"
    CHANGELOG = "changelog"
    MIGRATION = "migration"


@dataclass
class LibraryInfo:
    """Information about a library"""
    id: str
    name: str
    description: str
    version: Optional[str] = None
    trust_score: Optional[int] = None
    code_snippets: int = 0
    documentation_coverage: str = "unknown"


@dataclass
class DocumentationResult:
    """Documentation retrieval result"""
    library_id: str
    content: str
    doc_type: DocumentationType
    tokens_used: int
    topic_focus: Optional[str] = None
    relevance_score: float = 1.0


@dataclass
class CodeExample:
    """Code example from documentation"""
    title: str
    description: str
    code: str
    language: str
    library: str
    tags: List[str] = None


class Context7Adapter:
    """Adapter for Context7 MCP documentation tool"""

    def __init__(self, tool_executor=None):
        self.tool_executor = tool_executor
        self._library_cache = {}
        self._doc_cache = {}

    async def resolve_library_id(self, library_name: str) -> List[LibraryInfo]:
        """Resolve library name to Context7-compatible library ID"""
        try:
            # Check cache first
            if library_name in self._library_cache:
                return self._library_cache[library_name]

            # This would call the actual MCP tool
            # For now, simulate library resolution
            libraries = await self._simulate_library_resolution(library_name)

            # Cache results
            self._library_cache[library_name] = libraries

            logger.debug(f"Resolved {len(libraries)} libraries for: {library_name}")
            return libraries

        except Exception as e:
            logger.error(f"Failed to resolve library {library_name}: {e}")
            return []

    async def _simulate_library_resolution(self, library_name: str) -> List[LibraryInfo]:
        """Simulate library resolution (replace with actual MCP call)"""
        # Common library mappings
        library_mappings = {
            "react": [
                LibraryInfo("facebook/react", "React", "JavaScript library for building user interfaces", "18.2.0", 10, 5000),
                LibraryInfo("facebook/react-dom", "ReactDOM", "React DOM renderer", "18.2.0", 10, 2000)
            ],
            "nextjs": [
                LibraryInfo("vercel/next.js", "Next.js", "React framework for production", "13.4.0", 9, 3000)
            ],
            "express": [
                LibraryInfo("expressjs/express", "Express", "Fast, unopinionated web framework for Node.js", "4.18.0", 9, 2500)
            ],
            "fastapi": [
                LibraryInfo("tiangolo/fastapi", "FastAPI", "Modern, fast web framework for building APIs with Python", "0.95.0", 9, 1800)
            ],
            "langchain": [
                LibraryInfo("langchain-ai/langchain", "LangChain", "Framework for developing applications powered by language models", "0.1.0", 8, 1200)
            ],
            "langgraph": [
                LibraryInfo("langchain-ai/langgraph", "LangGraph", "Library for building stateful, multi-actor applications with LLMs", "0.0.40", 8, 800)
            ],
            "livekit": [
                LibraryInfo("livekit/livekit", "LiveKit", "Open source WebRTC infrastructure", "1.4.0", 8, 600)
            ]
        }

        # Find matching libraries
        matches = []
        library_lower = library_name.lower()

        for key, libs in library_mappings.items():
            if library_lower in key or key in library_lower:
                matches.extend(libs)

        # If no exact matches, create a generic entry
        if not matches:
            matches.append(LibraryInfo(
                f"generic/{library_name}",
                library_name,
                f"Documentation for {library_name}",
                trust_score=5,
                code_snippets=100
            ))

        return matches

    async def get_library_docs(
        self,
        library_id: str,
        topic: Optional[str] = None,
        tokens: int = 5000
    ) -> DocumentationResult:
        """Get documentation for a specific library"""
        try:
            # Check cache
            cache_key = f"{library_id}:{topic}:{tokens}"
            if cache_key in self._doc_cache:
                return self._doc_cache[cache_key]

            # This would call the actual MCP tool
            # For now, simulate documentation retrieval
            doc_result = await self._simulate_doc_retrieval(library_id, topic, tokens)

            # Cache result
            self._doc_cache[cache_key] = doc_result

            logger.debug(f"Retrieved docs for {library_id}, topic: {topic}")
            return doc_result

        except Exception as e:
            logger.error(f"Failed to get docs for {library_id}: {e}")
            return DocumentationResult(
                library_id=library_id,
                content=f"Error retrieving documentation: {str(e)}",
                doc_type=DocumentationType.API_REFERENCE,
                tokens_used=0
            )

    async def _simulate_doc_retrieval(
        self,
        library_id: str,
        topic: Optional[str],
        tokens: int
    ) -> DocumentationResult:
        """Simulate documentation retrieval (replace with actual MCP call)"""

        # Template documentation content based on library
        doc_templates = {
            "facebook/react": self._get_react_docs(topic),
            "vercel/next.js": self._get_nextjs_docs(topic),
            "langchain-ai/langgraph": self._get_langgraph_docs(topic),
            "livekit/livekit": self._get_livekit_docs(topic),
            "tiangolo/fastapi": self._get_fastapi_docs(topic)
        }

        content = doc_templates.get(library_id, f"Generic documentation for {library_id}")

        if topic:
            content = f"# {topic}\n\n{content}"

        return DocumentationResult(
            library_id=library_id,
            content=content[:tokens * 4],  # Approximate token limit
            doc_type=DocumentationType.API_REFERENCE,
            tokens_used=min(len(content) // 4, tokens),
            topic_focus=topic,
            relevance_score=0.9 if topic else 0.7
        )

    def _get_react_docs(self, topic: Optional[str]) -> str:
        """Get React documentation content"""
        if topic and "hooks" in topic.lower():
            return """
# React Hooks

React Hooks let you use state and other React features without writing a class.

## useState
```jsx
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
    </div>
  );
}
```

## useEffect
```jsx
import { useEffect, useState } from 'react';

function UserProfile({ userId }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetchUser(userId).then(setUser);
  }, [userId]);

  return user ? <div>{user.name}</div> : <div>Loading...</div>;
}
```
"""
        elif topic and "component" in topic.lower():
            return """
# React Components

Components let you split the UI into independent, reusable pieces.

## Function Components
```jsx
function Welcome(props) {
  return <h1>Hello, {props.name}</h1>;
}
```

## Class Components
```jsx
class Welcome extends React.Component {
  render() {
    return <h1>Hello, {this.props.name}</h1>;
  }
}
```
"""
        else:
            return """
# React Documentation

React is a JavaScript library for building user interfaces. It lets you compose complex UIs from small and isolated pieces of code called "components".

## Key Concepts:
- Components: Independent, reusable pieces of UI
- JSX: Syntax extension for JavaScript
- Props: How data flows down to components
- State: Component's internal data
- Hooks: Functions that let you "hook into" React features
"""

    def _get_nextjs_docs(self, topic: Optional[str]) -> str:
        """Get Next.js documentation content"""
        if topic and "routing" in topic.lower():
            return """
# Next.js App Router

Next.js 13+ uses file-system based routing in the `app/` directory.

## Basic Routing
```
app/
├── page.tsx          # /
├── about/
│   └── page.tsx      # /about
└── blog/
    ├── page.tsx      # /blog
    └── [slug]/
        └── page.tsx  # /blog/[slug]
```

## Dynamic Routes
```tsx
// app/blog/[slug]/page.tsx
export default function BlogPost({ params }: { params: { slug: string } }) {
  return <h1>Post: {params.slug}</h1>
}
```

## Route Groups
```
app/
├── (marketing)/
│   ├── about/
│   └── contact/
└── (dashboard)/
    ├── dashboard/
    └── settings/
```
"""
        else:
            return """
# Next.js Documentation

Next.js is a React framework for building full-stack web applications.

## Key Features:
- App Router: File-system based routing
- Server Components: React components that render on the server
- API Routes: Build APIs with serverless functions
- Built-in CSS Support: Import CSS files directly
- Image Optimization: Automatic image optimization
- TypeScript Support: Built-in TypeScript support
"""

    def _get_langgraph_docs(self, topic: Optional[str]) -> str:
        """Get LangGraph documentation content"""
        if topic and "state" in topic.lower():
            return """
# LangGraph State Management

LangGraph uses typed state that flows through the graph.

## Defining State
```python
from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: list[str]
    current_task: str
    completed: bool

# Create graph with state
workflow = StateGraph(AgentState)
```

## State Updates
```python
def process_task(state: AgentState) -> AgentState:
    # Update state
    return {
        "messages": state["messages"] + ["Task processed"],
        "completed": True
    }
```
"""
        elif topic and "graph" in topic.lower():
            return """
# Building LangGraph Workflows

Create multi-step workflows with conditional logic.

## Basic Graph
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("process", process_task)
workflow.add_node("validate", validate_result)

# Add edges
workflow.add_edge("process", "validate")
workflow.add_edge("validate", END)

# Set entry point
workflow.set_entry_point("process")

# Compile
app = workflow.compile()
```

## Conditional Edges
```python
def decide_next_step(state: AgentState) -> str:
    if state["completed"]:
        return "finish"
    else:
        return "retry"

workflow.add_conditional_edges(
    "validate",
    decide_next_step,
    {
        "finish": END,
        "retry": "process"
    }
)
```
"""
        else:
            return """
# LangGraph Documentation

LangGraph is a library for building stateful, multi-actor applications with LLMs.

## Core Concepts:
- StateGraph: Main abstraction for building workflows
- Nodes: Individual steps in the workflow
- Edges: Connections between nodes
- State: Shared data structure that flows through the graph
- Checkpointing: Save and restore workflow state
"""

    def _get_livekit_docs(self, topic: Optional[str]) -> str:
        """Get LiveKit documentation content"""
        if topic and "audio" in topic.lower():
            return """
# LiveKit Audio Features

Handle real-time audio in LiveKit applications.

## Audio Tracks
```python
from livekit import rtc

# Create audio source
audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)

# Create track
audio_track = rtc.LocalAudioTrack.create_audio_track(
    "microphone", audio_source
)

# Publish track
await room.local_participant.publish_track(
    audio_track,
    rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
)
```

## Audio Processing
```python
# Subscribe to audio
@room.on("track_subscribed")
def on_track_subscribed(track, publication, participant):
    if isinstance(track, rtc.AudioTrack):
        audio_stream = rtc.AudioStream(track)

        async def process_audio():
            async for frame in audio_stream:
                # Process audio frame
                audio_data = frame.data.tobytes()
                # Your processing logic here
```
"""
        else:
            return """
# LiveKit Documentation

LiveKit is an open source WebRTC infrastructure for real-time communication.

## Key Components:
- Room: Virtual space for participants to connect
- Participant: User in a room
- Track: Audio/video stream from a participant
- Publication: Metadata about a published track
- Subscription: Connection to receive a published track

## Basic Usage:
```python
from livekit import rtc

room = rtc.Room()

@room.on("participant_connected")
def on_participant_connected(participant):
    print(f"{participant.identity} joined")

await room.connect(url, token)
```
"""

    def _get_fastapi_docs(self, topic: Optional[str]) -> str:
        """Get FastAPI documentation content"""
        if topic and "routing" in topic.lower():
            return """
# FastAPI Routing

Define API endpoints with path operations.

## Basic Routes
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

## Route Parameters
```python
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model_name": model_name}
```
"""
        else:
            return """
# FastAPI Documentation

FastAPI is a modern, fast web framework for building APIs with Python.

## Key Features:
- Automatic API documentation with Swagger UI
- Built-in data validation with Pydantic
- High performance (on par with NodeJS and Go)
- Easy to use and learn
- Production-ready code with automatic validation

## Quick Start:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}
```
"""

    async def search_documentation(
        self,
        query: str,
        libraries: Optional[List[str]] = None,
        doc_types: Optional[List[DocumentationType]] = None
    ) -> List[DocumentationResult]:
        """Search documentation across libraries"""
        try:
            results = []

            # If no libraries specified, search common ones
            if not libraries:
                libraries = ["react", "nextjs", "fastapi", "langgraph", "livekit"]

            for library_name in libraries:
                # Resolve library
                lib_infos = await self.resolve_library_id(library_name)

                for lib_info in lib_infos[:1]:  # Take first match
                    # Get docs with query as topic
                    doc_result = await self.get_library_docs(
                        lib_info.id,
                        topic=query,
                        tokens=2000
                    )

                    # Calculate relevance score based on query
                    relevance = self._calculate_relevance(doc_result.content, query)
                    doc_result.relevance_score = relevance

                    if relevance > 0.3:  # Only include relevant results
                        results.append(doc_result)

            # Sort by relevance
            results.sort(key=lambda x: x.relevance_score, reverse=True)

            logger.debug(f"Found {len(results)} documentation results for: {query}")
            return results

        except Exception as e:
            logger.error(f"Failed to search documentation: {e}")
            return []

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query"""
        try:
            content_lower = content.lower()
            query_lower = query.lower()

            # Simple keyword matching
            query_words = query_lower.split()
            matches = sum(1 for word in query_words if word in content_lower)

            if not query_words:
                return 0.0

            relevance = matches / len(query_words)

            # Boost score if query appears as exact phrase
            if query_lower in content_lower:
                relevance += 0.3

            return min(relevance, 1.0)

        except Exception:
            return 0.5  # Default relevance

    async def get_code_examples(
        self,
        library_name: str,
        concept: str,
        max_examples: int = 5
    ) -> List[CodeExample]:
        """Get code examples for a specific concept"""
        try:
            # Resolve library
            libraries = await self.resolve_library_id(library_name)
            if not libraries:
                return []

            # Get documentation
            doc_result = await self.get_library_docs(
                libraries[0].id,
                topic=concept,
                tokens=3000
            )

            # Extract code examples from documentation
            examples = self._extract_code_examples(doc_result.content, library_name, concept)

            return examples[:max_examples]

        except Exception as e:
            logger.error(f"Failed to get code examples: {e}")
            return []

    def _extract_code_examples(
        self,
        content: str,
        library: str,
        concept: str
    ) -> List[CodeExample]:
        """Extract code examples from documentation content"""
        try:
            examples = []

            # Find code blocks
            code_pattern = r'```(\w+)?\n(.*?)\n```'
            matches = re.findall(code_pattern, content, re.DOTALL)

            for i, (language, code) in enumerate(matches):
                if not language:
                    language = "text"

                # Create example
                example = CodeExample(
                    title=f"{concept} Example {i+1}",
                    description=f"Code example for {concept} in {library}",
                    code=code.strip(),
                    language=language,
                    library=library,
                    tags=[concept, library]
                )
                examples.append(example)

            return examples

        except Exception as e:
            logger.error(f"Failed to extract code examples: {e}")
            return []

    async def get_best_practices(
        self,
        library_name: str,
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get best practices for a library"""
        try:
            # Get comprehensive documentation
            libraries = await self.resolve_library_id(library_name)
            if not libraries:
                return {"error": "Library not found"}

            doc_result = await self.get_library_docs(
                libraries[0].id,
                topic=f"best practices {topic}" if topic else "best practices",
                tokens=4000
            )

            # Extract best practices
            practices = self._extract_best_practices(doc_result.content)

            return {
                "library": library_name,
                "topic": topic,
                "practices": practices,
                "source": libraries[0].id
            }

        except Exception as e:
            logger.error(f"Failed to get best practices: {e}")
            return {"error": str(e)}

    def _extract_best_practices(self, content: str) -> List[str]:
        """Extract best practices from documentation"""
        practices = []

        # Look for common best practice indicators
        indicators = ["best practice", "recommended", "should", "avoid", "don't", "do"]

        lines = content.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if any(indicator in line_lower for indicator in indicators):
                if len(line.strip()) > 20:  # Avoid very short lines
                    practices.append(line.strip())

        return practices[:10]  # Limit to top 10


# Utility functions for integration with agents

async def get_documentation_for_task(
    adapter: Context7Adapter,
    task_description: str,
    preferred_libraries: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Get relevant documentation for a specific task"""
    try:
        # Search documentation
        results = await adapter.search_documentation(
            task_description,
            libraries=preferred_libraries
        )

        # Get code examples
        examples = []
        if preferred_libraries:
            for lib in preferred_libraries[:2]:  # Limit to first 2 libraries
                lib_examples = await adapter.get_code_examples(lib, task_description)
                examples.extend(lib_examples)

        return {
            "task": task_description,
            "documentation": results,
            "code_examples": examples,
            "total_results": len(results)
        }

    except Exception as e:
        logger.error(f"Failed to get documentation for task: {e}")
        return {"error": str(e)}


async def lookup_api_reference(
    adapter: Context7Adapter,
    library_name: str,
    api_method: str
) -> Dict[str, Any]:
    """Look up specific API reference"""
    try:
        # Get targeted documentation
        doc_result = await adapter.get_library_docs(
            f"generic/{library_name}",  # Fallback if not found
            topic=api_method,
            tokens=2000
        )

        # Get related examples
        examples = await adapter.get_code_examples(library_name, api_method)

        return {
            "library": library_name,
            "method": api_method,
            "documentation": doc_result.content,
            "examples": examples,
            "relevance": doc_result.relevance_score
        }

    except Exception as e:
        logger.error(f"Failed to lookup API reference: {e}")
        return {"error": str(e)}


# Create default adapter instance
default_context7 = Context7Adapter()