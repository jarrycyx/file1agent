"""
OpenAI implementation of Visual Language Model (VLM).
"""

from typing import Any, Dict, List
from loguru import logger
import openai

from .base_vlm import BaseVLM


class OpenAIVLM(BaseVLM):
    """
    OpenAI implementation of Visual Language Model.
    """
    
    def _initialize_model(self):
        """
        Initialize the OpenAI client.
        """
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _format_message(self, prompt: str, image_base64: str) -> List[Dict[str, Any]]:
        """
        Format the message for OpenAI's API.
        
        Args:
            prompt: Text prompt
            image_base64: Base64 encoded image
            
        Returns:
            Formatted message for OpenAI's API
        """
        # Try different message formats for compatibility with different models
        formatters = [
            # Formatter A: Standard OpenAI format
            lambda p, img: [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": p},
                        {"type": "image_url", "image_url": {"url": img}}
                    ]
                }
            ],
            # Formatter B: Alternative format for some models
            lambda p, img: [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": p},
                        {
                            "type": "image",
                            "source_type": "base64",
                            "data": img,
                            "mime_type": "image/jpeg"
                        }
                    ]
                }
            ]
        ]
        
        # Return the first formatter by default
        # The call_vlm method will try different formatters if needed
        return formatters[0](prompt, image_base64)
    
    def _call_model(self, message: List[Dict[str, Any]]) -> str:
        """
        Call the OpenAI model with the formatted message.
        
        Args:
            message: Formatted message for OpenAI's API
            
        Returns:
            Model response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            max_tokens=1024,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def call_vlm(self, image_base64: str, prompt: str, max_retries: int = 3) -> str:
        """
        Override the base method to try different message formats.
        
        Args:
            image_base64: Base64 encoded image data
            prompt: Prompt text
            max_retries: Maximum number of retries on failure
            
        Returns:
            VLM response content
        """
        self._initialize_model()
        
        # Define different formatters to try
        formatters = [
            # Formatter A: Standard OpenAI format
            lambda p, img: [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": p},
                        {"type": "image_url", "image_url": {"url": img}}
                    ]
                }
            ],
            # Formatter B: Alternative format for some models
            lambda p, img: [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": p},
                        {
                            "type": "image",
                            "source_type": "base64",
                            "data": img,
                            "mime_type": "image/jpeg"
                        }
                    ]
                }
            ]
        ]
        
        error_msg = ""
        for attempt in range(max_retries):
            for formatter in formatters:
                try:
                    # Format the message for the specific VLM API
                    message = formatter(prompt, image_base64)
                    
                    # Call the VLM model
                    response = self._call_model(message)
                    
                    # Save the conversation for debugging
                    self._save_llm_call([message, response])
                    
                    logger.info(f"Vision response: {response.replace('\n', ' ')}")
                    return response
                except Exception as e:
                    error_msg = f"Error when calling VLM: {e}"
                    logger.warning(error_msg)
                    logger.warning("Retrying with different formatter...")
                    import time
                    time.sleep(5)
        
        return error_msg
