"""
Vision module for handling visual language model (VLM) requests.
"""

from .base_vlm import BaseVLM
from .openai_vlm import OpenAIVLM

__all__ = ["BaseVLM", "OpenAIVLM"]
