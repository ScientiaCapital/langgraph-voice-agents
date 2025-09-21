"""
Serena MCP tool adapter for LangGraph agents.
Provides code intelligence, navigation, and semantic code analysis.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class SymbolKind(Enum):
    """LSP symbol kinds"""
    FILE = 1
    MODULE = 2
    NAMESPACE = 3
    PACKAGE = 4
    CLASS = 5
    METHOD = 6
    PROPERTY = 7
    FIELD = 8
    CONSTRUCTOR = 9
    ENUM = 10
    INTERFACE = 11
    FUNCTION = 12
    VARIABLE = 13
    CONSTANT = 14


@dataclass
class CodeSymbol:
    """Represents a code symbol"""
    name: str
    kind: SymbolKind
    name_path: str
    relative_path: str
    line_start: int
    line_end: int
    body: Optional[str] = None
    description: Optional[str] = None


@dataclass
class FileInfo:
    """File information from Serena"""
    path: str
    exists: bool
    size: Optional[int] = None
    symbols: List[CodeSymbol] = None
    content: Optional[str] = None


@dataclass
class SearchResult:
    """Search result from Serena"""
    file_path: str
    line_number: int
    content: str
    context_before: List[str] = None
    context_after: List[str] = None


class SerenaAdapter:
    """Adapter for Serena MCP code intelligence tool"""

    def __init__(self, tool_executor=None, project_root: Optional[str] = None):
        self.tool_executor = tool_executor
        self.project_root = project_root or os.getcwd()
        self._active_project = None

    async def activate_project(self, project_path: str) -> bool:
        """Activate a project in Serena"""
        try:
            # This would call the actual MCP tool
            # For now, simulate project activation
            if os.path.exists(project_path):
                self._active_project = project_path
                logger.info(f"Activated Serena project: {project_path}")
                return True
            else:
                logger.error(f"Project path does not exist: {project_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to activate project: {e}")
            return False

    async def read_file(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        max_chars: int = -1
    ) -> FileInfo:
        """Read file content"""
        try:
            full_path = os.path.join(self.project_root, file_path) if not os.path.isabs(file_path) else file_path

            if not os.path.exists(full_path):
                return FileInfo(path=file_path, exists=False)

            # Read file content
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Apply line range if specified
            if start_line is not None or end_line is not None:
                start_idx = (start_line or 1) - 1  # Convert to 0-based
                end_idx = end_line if end_line else len(lines)
                lines = lines[start_idx:end_idx]

            content = ''.join(lines)

            # Apply character limit
            if max_chars > 0 and len(content) > max_chars:
                content = content[:max_chars] + "... (truncated)"

            return FileInfo(
                path=file_path,
                exists=True,
                size=len(content),
                content=content
            )

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return FileInfo(path=file_path, exists=False)

    async def list_directory(
        self,
        dir_path: str = ".",
        recursive: bool = False,
        max_chars: int = -1
    ) -> Dict[str, Any]:
        """List directory contents"""
        try:
            full_path = os.path.join(self.project_root, dir_path) if not os.path.isabs(dir_path) else dir_path

            if not os.path.exists(full_path):
                return {"error": f"Directory does not exist: {dir_path}"}

            result = {"directories": [], "files": []}

            if recursive:
                for root, dirs, files in os.walk(full_path):
                    rel_root = os.path.relpath(root, full_path)
                    for d in dirs:
                        result["directories"].append(os.path.join(rel_root, d) if rel_root != "." else d)
                    for f in files:
                        result["files"].append(os.path.join(rel_root, f) if rel_root != "." else f)
            else:
                for item in os.listdir(full_path):
                    item_path = os.path.join(full_path, item)
                    if os.path.isdir(item_path):
                        result["directories"].append(item)
                    else:
                        result["files"].append(item)

            return result

        except Exception as e:
            logger.error(f"Failed to list directory {dir_path}: {e}")
            return {"error": str(e)}

    async def find_symbol(
        self,
        name_path: str,
        relative_path: str = "",
        include_body: bool = False,
        depth: int = 0,
        substring_matching: bool = False
    ) -> List[CodeSymbol]:
        """Find symbols in codebase"""
        try:
            # This would call the actual MCP tool
            # For now, simulate symbol finding with basic file scanning
            symbols = []

            if relative_path:
                files_to_search = [relative_path]
            else:
                # Find relevant files
                result = await self.list_directory(".", recursive=True)
                files_to_search = [f for f in result.get("files", []) if f.endswith(('.py', '.js', '.ts', '.jsx', '.tsx'))]

            for file_path in files_to_search:
                file_symbols = await self._extract_symbols_from_file(file_path, name_path, include_body, substring_matching)
                symbols.extend(file_symbols)

            logger.debug(f"Found {len(symbols)} symbols for {name_path}")
            return symbols

        except Exception as e:
            logger.error(f"Failed to find symbol {name_path}: {e}")
            return []

    async def _extract_symbols_from_file(
        self,
        file_path: str,
        name_pattern: str,
        include_body: bool,
        substring_matching: bool
    ) -> List[CodeSymbol]:
        """Extract symbols from a single file"""
        try:
            file_info = await self.read_file(file_path)
            if not file_info.exists or not file_info.content:
                return []

            symbols = []
            lines = file_info.content.split('\n')

            # Simple pattern matching for common languages
            for i, line in enumerate(lines):
                line_stripped = line.strip()

                # Python class/function detection
                if line_stripped.startswith('class ') or line_stripped.startswith('def '):
                    symbol_name = self._extract_symbol_name(line_stripped)
                    if symbol_name and self._matches_pattern(symbol_name, name_pattern, substring_matching):
                        symbol = CodeSymbol(
                            name=symbol_name,
                            kind=SymbolKind.CLASS if line_stripped.startswith('class') else SymbolKind.FUNCTION,
                            name_path=symbol_name,
                            relative_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,  # Would need better parsing for accurate end
                            body=line if include_body else None
                        )
                        symbols.append(symbol)

                # JavaScript/TypeScript function/class detection
                elif any(keyword in line_stripped for keyword in ['function ', 'class ', 'const ', 'let ', 'var ']):
                    symbol_name = self._extract_js_symbol_name(line_stripped)
                    if symbol_name and self._matches_pattern(symbol_name, name_pattern, substring_matching):
                        symbol = CodeSymbol(
                            name=symbol_name,
                            kind=self._determine_js_symbol_kind(line_stripped),
                            name_path=symbol_name,
                            relative_path=file_path,
                            line_start=i + 1,
                            line_end=i + 1,
                            body=line if include_body else None
                        )
                        symbols.append(symbol)

            return symbols

        except Exception as e:
            logger.error(f"Failed to extract symbols from {file_path}: {e}")
            return []

    def _extract_symbol_name(self, line: str) -> Optional[str]:
        """Extract symbol name from Python code line"""
        try:
            if line.startswith('class '):
                # Extract class name
                parts = line.split('(')[0].split(':')[0].split()
                return parts[1] if len(parts) > 1 else None
            elif line.startswith('def '):
                # Extract function name
                parts = line.split('(')[0].split()
                return parts[1] if len(parts) > 1 else None
            return None
        except:
            return None

    def _extract_js_symbol_name(self, line: str) -> Optional[str]:
        """Extract symbol name from JavaScript/TypeScript code line"""
        try:
            if 'function ' in line:
                # Extract function name
                parts = line.split('function ')[1].split('(')[0].strip()
                return parts if parts else None
            elif 'class ' in line:
                # Extract class name
                parts = line.split('class ')[1].split(' ')[0].split('{')[0].strip()
                return parts if parts else None
            elif any(keyword in line for keyword in ['const ', 'let ', 'var ']):
                # Extract variable name
                for keyword in ['const ', 'let ', 'var ']:
                    if keyword in line:
                        parts = line.split(keyword)[1].split('=')[0].strip()
                        return parts if parts else None
            return None
        except:
            return None

    def _determine_js_symbol_kind(self, line: str) -> SymbolKind:
        """Determine symbol kind for JavaScript/TypeScript"""
        if 'function ' in line:
            return SymbolKind.FUNCTION
        elif 'class ' in line:
            return SymbolKind.CLASS
        elif any(keyword in line for keyword in ['const ', 'let ', 'var ']):
            return SymbolKind.VARIABLE
        return SymbolKind.VARIABLE

    def _matches_pattern(self, symbol_name: str, pattern: str, substring_matching: bool) -> bool:
        """Check if symbol name matches pattern"""
        if substring_matching:
            return pattern.lower() in symbol_name.lower()
        else:
            return pattern.lower() == symbol_name.lower()

    async def search_for_pattern(
        self,
        pattern: str,
        relative_path: str = "",
        context_lines_before: int = 0,
        context_lines_after: int = 0,
        max_results: int = 100
    ) -> List[SearchResult]:
        """Search for pattern in codebase"""
        try:
            results = []

            if relative_path:
                files_to_search = [relative_path]
            else:
                # Search all text files
                dir_result = await self.list_directory(".", recursive=True)
                files_to_search = [
                    f for f in dir_result.get("files", [])
                    if any(f.endswith(ext) for ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.txt', '.json', '.yaml', '.yml'])
                ]

            for file_path in files_to_search[:50]:  # Limit files to search
                file_results = await self._search_in_file(
                    file_path, pattern, context_lines_before, context_lines_after
                )
                results.extend(file_results)

                if len(results) >= max_results:
                    break

            logger.debug(f"Found {len(results)} matches for pattern: {pattern}")
            return results[:max_results]

        except Exception as e:
            logger.error(f"Failed to search for pattern {pattern}: {e}")
            return []

    async def _search_in_file(
        self,
        file_path: str,
        pattern: str,
        context_before: int,
        context_after: int
    ) -> List[SearchResult]:
        """Search for pattern in a single file"""
        try:
            file_info = await self.read_file(file_path)
            if not file_info.exists or not file_info.content:
                return []

            results = []
            lines = file_info.content.split('\n')

            for i, line in enumerate(lines):
                if pattern.lower() in line.lower():
                    # Get context lines
                    context_before_lines = lines[max(0, i - context_before):i] if context_before > 0 else []
                    context_after_lines = lines[i + 1:i + 1 + context_after] if context_after > 0 else []

                    result = SearchResult(
                        file_path=file_path,
                        line_number=i + 1,
                        content=line,
                        context_before=context_before_lines,
                        context_after=context_after_lines
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to search in file {file_path}: {e}")
            return []

    async def get_symbols_overview(
        self,
        file_path: str,
        max_chars: int = -1
    ) -> Dict[str, Any]:
        """Get overview of symbols in a file"""
        try:
            symbols = await self.find_symbol("", file_path, include_body=False)

            overview = {
                "file_path": file_path,
                "total_symbols": len(symbols),
                "by_kind": {},
                "symbols": []
            }

            # Group by kind
            for symbol in symbols:
                kind_name = symbol.kind.name.lower()
                if kind_name not in overview["by_kind"]:
                    overview["by_kind"][kind_name] = 0
                overview["by_kind"][kind_name] += 1

                overview["symbols"].append({
                    "name": symbol.name,
                    "kind": kind_name,
                    "line": symbol.line_start,
                    "name_path": symbol.name_path
                })

            return overview

        except Exception as e:
            logger.error(f"Failed to get symbols overview for {file_path}: {e}")
            return {"error": str(e)}

    async def find_referencing_symbols(
        self,
        name_path: str,
        relative_path: str
    ) -> List[SearchResult]:
        """Find symbols that reference the given symbol"""
        try:
            # Extract the symbol name from name_path
            symbol_name = name_path.split('/')[-1]

            # Search for references
            results = await self.search_for_pattern(
                symbol_name,
                context_lines_before=2,
                context_lines_after=2
            )

            # Filter out the definition itself
            filtered_results = [
                result for result in results
                if result.file_path != relative_path or
                not any(keyword in result.content for keyword in ['def ', 'class ', 'function '])
            ]

            logger.debug(f"Found {len(filtered_results)} references to {name_path}")
            return filtered_results

        except Exception as e:
            logger.error(f"Failed to find references for {name_path}: {e}")
            return []

    async def analyze_codebase_structure(self) -> Dict[str, Any]:
        """Analyze overall codebase structure"""
        try:
            structure = {
                "total_files": 0,
                "by_language": {},
                "top_level_directories": [],
                "key_files": []
            }

            # Get directory listing
            dir_result = await self.list_directory(".", recursive=True)

            # Analyze files
            files = dir_result.get("files", [])
            structure["total_files"] = len(files)

            # Group by language
            for file_path in files:
                ext = os.path.splitext(file_path)[1]
                if ext not in structure["by_language"]:
                    structure["by_language"][ext] = 0
                structure["by_language"][ext] += 1

            # Get top-level directories
            structure["top_level_directories"] = dir_result.get("directories", [])

            # Identify key files
            key_patterns = ['package.json', 'requirements.txt', 'Cargo.toml', 'go.mod', 'setup.py', 'main.py', 'index.js', 'app.py']
            structure["key_files"] = [f for f in files if any(pattern in f for pattern in key_patterns)]

            return structure

        except Exception as e:
            logger.error(f"Failed to analyze codebase structure: {e}")
            return {"error": str(e)}


# Utility functions for integration with agents

async def analyze_code_context(
    adapter: SerenaAdapter,
    file_path: str,
    symbol_name: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze code context for a file or symbol"""
    try:
        context = {
            "file_info": await adapter.read_file(file_path, max_chars=5000),
            "symbols_overview": await adapter.get_symbols_overview(file_path),
            "directory_context": await adapter.list_directory(os.path.dirname(file_path) or ".")
        }

        if symbol_name:
            context["symbol_references"] = await adapter.find_referencing_symbols(
                symbol_name, file_path
            )

        return context

    except Exception as e:
        logger.error(f"Failed to analyze code context: {e}")
        return {"error": str(e)}


async def find_implementation_location(
    adapter: SerenaAdapter,
    feature_description: str
) -> Dict[str, Any]:
    """Find where to implement a new feature"""
    try:
        # Search for related patterns
        keywords = feature_description.lower().split()
        search_results = []

        for keyword in keywords[:3]:  # Limit to first 3 keywords
            results = await adapter.search_for_pattern(keyword, max_results=10)
            search_results.extend(results)

        # Analyze codebase structure
        structure = await adapter.analyze_codebase_structure()

        return {
            "search_results": search_results,
            "codebase_structure": structure,
            "recommendations": _generate_implementation_recommendations(search_results, structure)
        }

    except Exception as e:
        logger.error(f"Failed to find implementation location: {e}")
        return {"error": str(e)}


def _generate_implementation_recommendations(
    search_results: List[SearchResult],
    structure: Dict[str, Any]
) -> List[str]:
    """Generate recommendations for implementation location"""
    recommendations = []

    # Analyze file patterns
    file_patterns = {}
    for result in search_results:
        dir_name = os.path.dirname(result.file_path)
        if dir_name not in file_patterns:
            file_patterns[dir_name] = 0
        file_patterns[dir_name] += 1

    # Most common directories
    if file_patterns:
        most_common = max(file_patterns.items(), key=lambda x: x[1])
        recommendations.append(f"Consider implementing in {most_common[0]}/ directory (most related code)")

    # Language recommendations
    languages = structure.get("by_language", {})
    if languages:
        main_lang = max(languages.items(), key=lambda x: x[1])
        recommendations.append(f"Primary language appears to be {main_lang[0]} files")

    # Key files
    key_files = structure.get("key_files", [])
    if key_files:
        recommendations.append(f"Key configuration files: {', '.join(key_files)}")

    return recommendations


# Create default adapter instance
default_serena = SerenaAdapter()