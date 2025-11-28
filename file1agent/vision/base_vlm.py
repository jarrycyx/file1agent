"""
Base class for Visual Language Model (VLM) implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json
from datetime import datetime
from loguru import logger

from ..config import File1AgentConfig, ModelConfig


class BaseVLM(ABC):
    """
    Abstract base class for Visual Language Model implementations.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the VLM with configuration.
        
        Args:
            config: File1AgentConfiguration object
        """
        self.config = config
        self.model = config.model
        self.base_url = config.base_url
        self.api_key = config.api_key
    
    
    @abstractmethod
    def call_vlm(self, image_base64: str, prompt: str, max_retries: int = 3) -> str:
        """
        Generic VLM calling function for handling image-related requests.
        
        Args:
            image_base64: Base64 encoded image data
            prompt: Prompt text
            max_retries: Maximum number of retries on failure
            
        Returns:
            VLM response content
        """
        pass