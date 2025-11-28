import os
import sys
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union
from loguru import logger

from ..config import File1AgentConfig


class File1AgentBase:
    """
    File management tool for detecting duplicate files and simulated data files.
    Uses file summaries to identify potential duplicates and LLM to verify.
    """

    def __init__(self, config: Union[File1AgentConfig, str, dict, None] = None, log_level: str = "WARNING", **kwargs):
        """
        Initialize the file management tool

        Args:
            config: Configuration object or path to TOML file or TOML string
        """

        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:YYYYMMDDHHmmss}</green>|<level>{level}</level>|{message}|<yellow>{file}:{line}</yellow>|"
            + f"<cyan>file1.agent</cyan>",
            colorize=True,
            level=log_level,
        )

        if isinstance(config, str):
            if os.path.isfile(config):
                self.config = File1AgentConfig.from_toml(config)
            else:
                self.config = File1AgentConfig.from_toml_str(config)
        elif isinstance(config, dict):
            self.config = File1AgentConfig.from_dict(config)
        elif config is None:
            self.config = File1AgentConfig.from_args(**kwargs)
        else:
            self.config = config
