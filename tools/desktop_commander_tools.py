"""
Desktop Commander MCP tool adapter for LangGraph agents.
Provides file system operations, process management, and search capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class FileType(Enum):
    """File type classification"""
    TEXT = "text"
    BINARY = "binary"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ARCHIVE = "archive"
    EXECUTABLE = "executable"


class SearchType(Enum):
    """Search operation types"""
    FILES = "files"
    CONTENT = "content"


class ProcessStatus(Enum):
    """Process execution status"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class FileInfo:
    """File information structure"""
    path: str
    name: str
    size: int
    created: datetime
    modified: datetime
    file_type: FileType
    permissions: str
    is_directory: bool = False


@dataclass
class SearchResult:
    """Search result structure"""
    path: str
    matches: List[str]
    line_numbers: List[int]
    context: Optional[str] = None


@dataclass
class ProcessResult:
    """Process execution result"""
    pid: Optional[int]
    command: str
    stdout: str
    stderr: str
    return_code: int
    status: ProcessStatus
    execution_time: float


class DesktopCommanderAdapter:
    """Adapter for Desktop Commander MCP tool"""

    def __init__(self, tool_executor=None):
        self.tool_executor = tool_executor
        self._active_processes = {}
        self._search_sessions = {}
        self._base_path = os.getcwd()

    async def read_file(
        self,
        path: str,
        offset: int = 0,
        length: int = 1000,
        is_url: bool = False
    ) -> Dict[str, Any]:
        """Read file contents with optional offset and length"""
        try:
            if is_url:
                # Simulated URL reading (would use actual HTTP client)
                return {
                    "success": True,
                    "content": f"URL content from {path}",
                    "is_url": True,
                    "size": len(f"URL content from {path}")
                }

            # Ensure absolute path
            abs_path = os.path.abspath(path)
            
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }

            # Check if it's a directory
            if os.path.isdir(abs_path):
                return {
                    "success": False,
                    "error": f"Path is a directory: {path}"
                }

            # Read file content
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as file:
                if offset > 0:
                    # Skip to offset
                    for _ in range(offset):
                        if not file.readline():
                            break

                # Read specified number of lines
                lines = []
                for _ in range(length):
                    line = file.readline()
                    if not line:
                        break
                    lines.append(line.rstrip('\n\r'))

                content = '\n'.join(lines)

            logger.debug(f"Read file: {path} (offset: {offset}, length: {length})")
            return {
                "success": True,
                "content": content,
                "path": abs_path,
                "lines_read": len(lines),
                "offset": offset
            }

        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def write_file(
        self,
        path: str,
        content: str,
        mode: str = "rewrite"
    ) -> Dict[str, Any]:
        """Write content to file"""
        try:
            abs_path = os.path.abspath(path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            # Determine write mode
            write_mode = "w" if mode == "rewrite" else "a"

            # Write content
            with open(abs_path, write_mode, encoding='utf-8') as file:
                file.write(content)

            # Get file info
            file_stat = os.stat(abs_path)
            
            logger.debug(f"Written to file: {path} (mode: {mode})")
            return {
                "success": True,
                "path": abs_path,
                "mode": mode,
                "size": file_stat.st_size,
                "content_length": len(content)
            }

        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def list_directory(
        self,
        path: str = "."
    ) -> Dict[str, Any]:
        """List directory contents"""
        try:
            abs_path = os.path.abspath(path)
            
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": f"Directory not found: {path}"
                }

            if not os.path.isdir(abs_path):
                return {
                    "success": False,
                    "error": f"Path is not a directory: {path}"
                }

            # List directory contents
            items = []
            for item_name in os.listdir(abs_path):
                item_path = os.path.join(abs_path, item_name)
                
                try:
                    stat = os.stat(item_path)
                    is_dir = os.path.isdir(item_path)
                    
                    items.append({
                        "name": item_name,
                        "path": item_path,
                        "type": "[DIR]" if is_dir else "[FILE]",
                        "size": stat.st_size if not is_dir else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_directory": is_dir
                    })
                except OSError:
                    # Skip items that can't be accessed
                    continue

            # Sort directories first, then files
            items.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))

            logger.debug(f"Listed directory: {path} ({len(items)} items)")
            return {
                "success": True,
                "path": abs_path,
                "items": items,
                "total_count": len(items)
            }

        except Exception as e:
            logger.error(f"Failed to list directory {path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def create_directory(
        self,
        path: str
    ) -> Dict[str, Any]:
        """Create directory and any necessary parent directories"""
        try:
            abs_path = os.path.abspath(path)
            
            # Create directory with parents
            os.makedirs(abs_path, exist_ok=True)

            logger.debug(f"Created directory: {path}")
            return {
                "success": True,
                "path": abs_path,
                "message": f"Directory created: {path}"
            }

        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def move_file(
        self,
        source: str,
        destination: str
    ) -> Dict[str, Any]:
        """Move or rename file/directory"""
        try:
            abs_source = os.path.abspath(source)
            abs_dest = os.path.abspath(destination)

            if not os.path.exists(abs_source):
                return {
                    "success": False,
                    "error": f"Source not found: {source}"
                }

            # Ensure destination directory exists
            dest_dir = os.path.dirname(abs_dest)
            os.makedirs(dest_dir, exist_ok=True)

            # Move the file/directory
            os.rename(abs_source, abs_dest)

            logger.debug(f"Moved: {source} -> {destination}")
            return {
                "success": True,
                "source": abs_source,
                "destination": abs_dest,
                "message": f"Moved {source} to {destination}"
            }

        except Exception as e:
            logger.error(f"Failed to move {source} to {destination}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def start_search(
        self,
        path: str,
        pattern: str,
        search_type: SearchType = SearchType.FILES,
        file_pattern: Optional[str] = None,
        max_results: Optional[int] = None,
        literal_search: bool = False,
        ignore_case: bool = True
    ) -> Dict[str, Any]:
        """Start a background search operation"""
        try:
            import uuid
            session_id = str(uuid.uuid4())
            
            # Store search session
            self._search_sessions[session_id] = {
                "path": os.path.abspath(path),
                "pattern": pattern,
                "search_type": search_type,
                "file_pattern": file_pattern,
                "max_results": max_results or 100,
                "literal_search": literal_search,
                "ignore_case": ignore_case,
                "results": [],
                "status": "running",
                "created_at": datetime.utcnow()
            }

            # Simulate search execution (in real implementation, this would be async)
            await self._execute_search(session_id)

            logger.info(f"Started search session: {session_id}")
            return {
                "success": True,
                "session_id": session_id,
                "search_type": search_type.value,
                "pattern": pattern,
                "status": "running"
            }

        except Exception as e:
            logger.error(f"Failed to start search: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_search(self, session_id: str):
        """Execute search operation (simulated)"""
        try:
            session = self._search_sessions[session_id]
            search_path = session["path"]
            pattern = session["pattern"]
            search_type = session["search_type"]
            
            results = []
            
            if search_type == SearchType.FILES:
                # Search for files by name
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if pattern.lower() in file.lower() if session["ignore_case"] else pattern in file:
                            file_path = os.path.join(root, file)
                            results.append({
                                "path": file_path,
                                "name": file,
                                "type": "file"
                            })
                            
                            if len(results) >= session["max_results"]:
                                break
                    
                    if len(results) >= session["max_results"]:
                        break
            
            else:  # CONTENT search
                # Search for content within files
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith(('.txt', '.py', '.js', '.ts', '.md', '.json', '.yaml', '.yml')):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    for line_num, line in enumerate(f, 1):
                                        if pattern.lower() in line.lower() if session["ignore_case"] else pattern in line:
                                            results.append({
                                                "path": file_path,
                                                "line_number": line_num,
                                                "content": line.strip(),
                                                "type": "content"
                                            })
                                            
                                            if len(results) >= session["max_results"]:
                                                break
                                    
                                    if len(results) >= session["max_results"]:
                                        break
                            except:
                                continue
                    
                    if len(results) >= session["max_results"]:
                        break

            # Update session with results
            session["results"] = results
            session["status"] = "completed"
            session["completed_at"] = datetime.utcnow()

        except Exception as e:
            session["status"] = "failed"
            session["error"] = str(e)
            logger.error(f"Search execution failed: {e}")

    async def get_search_results(
        self,
        session_id: str,
        offset: int = 0,
        length: int = 100
    ) -> Dict[str, Any]:
        """Get search results with pagination"""
        try:
            if session_id not in self._search_sessions:
                return {
                    "success": False,
                    "error": f"Search session not found: {session_id}"
                }

            session = self._search_sessions[session_id]
            results = session["results"]
            
            # Apply offset and length
            paginated_results = results[offset:offset + length]

            return {
                "success": True,
                "session_id": session_id,
                "status": session["status"],
                "results": paginated_results,
                "total_results": len(results),
                "offset": offset,
                "length": len(paginated_results)
            }

        except Exception as e:
            logger.error(f"Failed to get search results: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def stop_search(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Stop an active search"""
        try:
            if session_id not in self._search_sessions:
                return {
                    "success": False,
                    "error": f"Search session not found: {session_id}"
                }

            session = self._search_sessions[session_id]
            session["status"] = "stopped"
            session["stopped_at"] = datetime.utcnow()

            logger.debug(f"Stopped search session: {session_id}")
            return {
                "success": True,
                "session_id": session_id,
                "status": "stopped"
            }

        except Exception as e:
            logger.error(f"Failed to stop search: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def start_process(
        self,
        command: str,
        timeout_ms: int = 30000,
        shell: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start a new process"""
        try:
            import uuid
            pid = str(uuid.uuid4())  # Simulated PID
            
            # Store process info
            self._active_processes[pid] = {
                "command": command,
                "status": ProcessStatus.RUNNING,
                "started_at": datetime.utcnow(),
                "timeout_ms": timeout_ms,
                "shell": shell or "bash"
            }

            # Simulate process execution
            if command.startswith("python"):
                output = "Python process started successfully"
                return_code = 0
            elif command.startswith("node"):
                output = "Node.js process started successfully"
                return_code = 0
            else:
                output = f"Command executed: {command}"
                return_code = 0

            # Update process result
            self._active_processes[pid].update({
                "status": ProcessStatus.COMPLETED,
                "stdout": output,
                "stderr": "",
                "return_code": return_code,
                "completed_at": datetime.utcnow()
            })

            logger.debug(f"Started process: {command}")
            return {
                "success": True,
                "pid": pid,
                "command": command,
                "status": "completed",
                "stdout": output,
                "return_code": return_code
            }

        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def interact_with_process(
        self,
        pid: str,
        input_text: str,
        timeout_ms: int = 8000,
        wait_for_prompt: bool = True
    ) -> Dict[str, Any]:
        """Send input to a running process"""
        try:
            if pid not in self._active_processes:
                return {
                    "success": False,
                    "error": f"Process not found: {pid}"
                }

            process = self._active_processes[pid]
            
            # Simulate process interaction
            if "python" in process["command"]:
                if "import" in input_text:
                    output = f"Imported: {input_text.replace('import ', '')}"
                elif "print" in input_text:
                    output = input_text.replace("print(", "").replace(")", "").strip('\'"')
                else:
                    output = f">>> {input_text}\nExecuted successfully"
            else:
                output = f"Process response to: {input_text}"

            logger.debug(f"Process interaction: {pid} <- {input_text}")
            return {
                "success": True,
                "pid": pid,
                "input": input_text,
                "output": output,
                "status": "waiting_for_input"
            }

        except Exception as e:
            logger.error(f"Process interaction failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def list_sessions(self) -> Dict[str, Any]:
        """List all active process sessions"""
        try:
            sessions = []
            for pid, process in self._active_processes.items():
                sessions.append({
                    "pid": pid,
                    "command": process["command"],
                    "status": process["status"].value,
                    "started_at": process["started_at"].isoformat(),
                    "runtime": (datetime.utcnow() - process["started_at"]).total_seconds()
                })

            return {
                "success": True,
                "sessions": sessions,
                "total_count": len(sessions)
            }

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def force_terminate(
        self,
        pid: str
    ) -> Dict[str, Any]:
        """Terminate a running process"""
        try:
            if pid not in self._active_processes:
                return {
                    "success": False,
                    "error": f"Process not found: {pid}"
                }

            process = self._active_processes[pid]
            process["status"] = ProcessStatus.TERMINATED
            process["terminated_at"] = datetime.utcnow()

            logger.debug(f"Terminated process: {pid}")
            return {
                "success": True,
                "pid": pid,
                "status": "terminated"
            }

        except Exception as e:
            logger.error(f"Failed to terminate process: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_file_info(
        self,
        path: str
    ) -> Dict[str, Any]:
        """Get detailed file metadata"""
        try:
            abs_path = os.path.abspath(path)
            
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }

            stat = os.stat(abs_path)
            is_dir = os.path.isdir(abs_path)

            file_info = {
                "path": abs_path,
                "name": os.path.basename(abs_path),
                "size": stat.st_size,
                "is_directory": is_dir,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
                "type": "directory" if is_dir else "file"
            }

            # Add line count for text files
            if not is_dir and abs_path.endswith(('.txt', '.py', '.js', '.ts', '.md', '.json')):
                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                        line_count = sum(1 for _ in f)
                    file_info["line_count"] = line_count
                except:
                    pass

            return {
                "success": True,
                "file_info": file_info
            }

        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Utility functions for agent integration

async def setup_project_workspace(
    adapter: DesktopCommanderAdapter,
    project_name: str,
    base_path: str = "."
) -> Dict[str, Any]:
    """Set up a complete project workspace"""
    try:
        project_path = os.path.join(base_path, project_name)
        
        # Create main project directory
        await adapter.create_directory(project_path)
        
        # Create standard subdirectories
        subdirs = ["src", "tests", "docs", "scripts", "config"]
        for subdir in subdirs:
            await adapter.create_directory(os.path.join(project_path, subdir))

        # Create basic files
        files_to_create = {
            "README.md": f"# {project_name}\n\nProject description goes here.",
            ".gitignore": "*.pyc\n__pycache__/\n.env\nnode_modules/\n",
            "requirements.txt": "# Python dependencies\n",
            "package.json": f'{{\n  "name": "{project_name}",\n  "version": "1.0.0"\n}}'
        }

        for filename, content in files_to_create.items():
            await adapter.write_file(
                os.path.join(project_path, filename),
                content
            )

        return {
            "success": True,
            "project_path": project_path,
            "directories_created": len(subdirs) + 1,
            "files_created": len(files_to_create)
        }

    except Exception as e:
        logger.error(f"Failed to setup workspace: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Create default adapter instance
default_desktop_commander = DesktopCommanderAdapter()