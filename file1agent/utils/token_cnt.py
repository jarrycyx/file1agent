import math
from typing import Iterable, Union

# 定义消息类型
class HumanMessage:
    def __init__(self, content: str):
        self.content = content
        self.name = None

class AIMessage:
    def __init__(self, content: str, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.name = None

class ToolMessage:
    def __init__(self, content: str, tool_call_id: str):
        self.content = content
        self.tool_call_id = tool_call_id
        self.name = None

def _get_message_openai_role(message):
    """获取消息的OpenAI角色"""
    if isinstance(message, HumanMessage):
        return "user"
    elif isinstance(message, AIMessage):
        return "assistant"
    elif isinstance(message, ToolMessage):
        return "tool"
    else:
        return "unknown"

def convert_to_messages(messages):
    """将消息转换为标准消息格式"""
    # 简化实现，直接返回原始消息
    return messages

def count_tokens_approximately(
    messages: Iterable,
    *,
    chars_per_token: float = 4.0,
    extra_tokens_per_message: float = 3.0,
    count_name: bool = True,
) -> int:
    """Approximate the total number of tokens in messages.

    The token count includes stringified message content, role, and (optionally) name.
    - For AI messages, the token count also includes stringified tool calls.
    - For tool messages, the token count also includes the tool call ID.
    - For strings, the token count is based on the string length.

    Args:
        messages: List of messages or strings to count tokens for.
        chars_per_token: Number of characters per token to use for the approximation.
            Default is 4 (one token corresponds to ~4 chars for common English text).
            You can also specify float values for more fine-grained control.
            See more here: https://platform.openai.com/tokenizer
        extra_tokens_per_message: Number of extra tokens to add per message.
            Default is 3 (special tokens, including beginning/end of message).
            You can also specify float values for more fine-grained control.
            See more here: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        count_name: Whether to include message names in the count.
            Enabled by default.

    Returns:
        Approximate number of tokens in the messages.

    Note:
        This is a simple approximation that may not match the exact token count used by
        specific models. For accurate counts, use model-specific tokenizers.

    Warning:
        This function does not currently support counting image tokens.
    """
    token_count = 0.0
    for item in messages:
        # 如果是字符串，直接计算token数量
        if isinstance(item, str):
            message_chars = len(item)
            # 添加角色字符数（默认为user）
            message_chars += len("user")
        # 如果是消息对象，按照原来的逻辑处理
        else:
            message = item
            message_chars = 0
            
            # 处理字符串内容
            if isinstance(message.content, str):
                message_chars += len(message.content)
            # 处理非字符串内容
            else:
                content = repr(message.content)
                message_chars += len(content)

            # 处理AI消息的工具调用
            if (
                isinstance(message, AIMessage)
                # 排除Anthropic格式，因为工具调用已包含在内容中
                and not isinstance(message.content, list)
                and message.tool_calls
            ):
                tool_calls_content = repr(message.tool_calls)
                message_chars += len(tool_calls_content)

            # 处理工具消息的工具调用ID
            if isinstance(message, ToolMessage):
                message_chars += len(message.tool_call_id)

            # 添加角色字符数
            role = _get_message_openai_role(message)
            message_chars += len(role)

            # 添加名称字符数（如果存在且需要计数）
            if hasattr(message, 'name') and message.name and count_name:
                message_chars += len(message.name)

        # 注意：我们向上取整每条消息，以确保
        # 单个消息token计数加起来等于消息列表的总计数
        token_count += math.ceil(message_chars / chars_per_token)

        # 每条消息添加额外token
        token_count += extra_tokens_per_message

    # 如果extra_tokens_per_message是浮点数，再次向上取整
    return math.ceil(token_count)