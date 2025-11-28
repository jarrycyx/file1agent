"""
File1Agent: A Python package for file analysis, summarization, and relationship visualization.
"""

from .config import File1AgentConfig, ModelConfig, LLMConfig, RerankConfig
from .file_manager import FileManager
from .file_summary import FileSummary

import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYYMMDDHHmmss}</green>|<level>{level}</level>|{message}|<yellow>{file}:{line}</yellow>|"
    + f"<cyan>file1.agent</cyan>",
    colorize=True,
    level="INFO",
)

__all__ = [
    "File1Agent",
    "File1AgentConfig",
    "ModelConfig",
    "LLMConfig",
    "RerankConfig",
    "FileManager",
    "FileSummary",
]
