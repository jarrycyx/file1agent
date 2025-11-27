"""
File1: A Python package for file analysis, summarization, and relationship visualization.
"""

from .config import File1Config, ModelConfig, LLMConfig, RerankConfig
from .file_manager import FileManager
from .file_summary import FileSummary

import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYYMMDDHHmmss}</green>|<level>{level}</level>|{message}|<yellow>{file}:{line}</yellow>|"
    + f"<cyan>file1</cyan>",
    colorize=True,
    level="INFO",
)

__all__ = [
    "File1",
    "File1Config",
    "ModelConfig",
    "LLMConfig",
    "RerankConfig",
    "FileManager",
    "FileSummary",
]
