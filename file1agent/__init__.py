"""
File1Agent: A Python package for file analysis, summarization, and relationship visualization.
"""

from .config import File1AgentConfig, ModelConfig, LLMConfig, RerankConfig
from .file_manager import FileManager
from .utils.file_summary import FileSummary

import sys

__all__ = [
    "File1Agent",
    "File1AgentConfig",
    "ModelConfig",
    "LLMConfig",
    "RerankConfig",
    "FileManager",
    "FileSummary",
]
