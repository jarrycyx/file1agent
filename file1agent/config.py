import os
from typing import Optional
from pydantic import BaseModel, Field
import toml


class ModelConfig(BaseModel):
    """Chat model configuration"""

    model: str = Field(default="", description="Chat model name")
    base_url: Optional[str] = Field(default=None, description="Base URL for API")
    api_key: Optional[str] = Field(default=None, description="API key for chat service")


class LLMConfig(BaseModel):
    """Language model configuration"""

    language: str = Field(default="chs", description="Language for the model")
    chat: ModelConfig = Field(default_factory=ModelConfig, description="Chat model configuration")
    condenser: ModelConfig = Field(default_factory=ModelConfig, description="Condenser model configuration")
    vision: ModelConfig = Field(default_factory=ModelConfig, description="Vision model configuration")


class RerankConfig(BaseModel):
    """Rerank model configuration"""

    rerank_api_key: Optional[str] = Field(default=None, description="API key for rerank service")
    rerank_base_url: Optional[str] = Field(default=None, description="Base URL for rerank service")
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3", description="Rerank model name")
    file_duplicate_threshold: float = Field(
        default=0.8, description="Threshold for considering files as duplicates based on reranking score"
    )


class File1AgentConfig(BaseModel):
    """
    Configuration model for File1Agent agent file management.
    This model represents the configuration structure used for file management operations.
    """

    # Configuration sections
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    rerank: RerankConfig = Field(default_factory=RerankConfig, description="Rerank configuration")

    @classmethod
    def from_args(
        cls,
        llm_chat_model: str,
        llm_chat_base_url: str,
        llm_chat_api_key: str,
        llm_vision_model: str,
        llm_vision_base_url: str,
        llm_vision_api_key: str,
        rerank_model: str,
        rerank_api_key: str,
        rerank_base_url: str,
    ) -> "File1AgentConfig":
        """Load configuration from command line arguments"""
        config_dict = {
            "llm": {
                "chat": {
                    "model": llm_chat_model,
                    "base_url": llm_chat_base_url,
                    "api_key": llm_chat_api_key,
                },
                "vision": {
                    "model": llm_vision_model,
                    "base_url": llm_vision_base_url,
                    "api_key": llm_vision_api_key,
                },
            },
            "rerank": {
                "rerank_model": rerank_model,
                "rerank_api_key": rerank_api_key,
                "rerank_base_url": rerank_base_url,
            },
        }
        return cls.from_dict(config_dict)

    @classmethod
    def from_toml_str(cls, toml_str: str) -> "File1AgentConfig":
        """Load configuration from a TOML file"""
        config_data = toml.loads(toml_str)

        # Create config instance with default sub-configs
        config = cls(**config_data)

        return config

    @classmethod
    def from_dict(cls, config_dict: dict) -> "File1AgentConfig":
        """Load configuration from a dictionary"""
        # Create config instance with default sub-configs
        config = cls(**config_dict)

        return config

    @classmethod
    def from_toml(cls, toml_path: str) -> "File1AgentConfig":
        """Load configuration from a TOML file"""
        with open(toml_path, "r") as file:
            config_data = toml.load(file)

        # Create config instance with default sub-configs
        config = cls(**config_data)

        return config

    def save_toml(self, toml_path: str):
        """Save configuration to a TOML file"""
        # Convert the config to a dictionary
        config_dict = self.model_dump()

        with open(toml_path, "w") as file:
            toml.dump(config_dict, file)


def get_lang_prompt(lang: str) -> str:
    """
    Get language-specific prompt suffix based on the language parameter.

    Args:
        lang: Language code ('eng' or 'chs')

    Returns:
        Language-specific prompt suffix string
    """
    if lang == "chs":
        return "\n\n确保使用中文书写所有的论文文字、程序注释、思考过程、执行报告、文献综述，但不要强行翻译专有名词和引用文献的标题、人名、期刊名（例如LSTM，RCT，ICU，Lucas，Schmidgall）"
    elif lang == "eng":
        return ""
    else:
        # Default to English for unsupported languages
        return ""
