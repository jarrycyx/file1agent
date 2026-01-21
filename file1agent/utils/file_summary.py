import os
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import NoneType
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import tqdm
import functools

from ..config import File1AgentConfig, ModelConfig
from .token_cnt import HumanMessage, count_tokens_approximately
from ..vision import OpenAIVLM
from .image_converter import ImageConverter
from .base import File1AgentBase
from .file_inclusion import FileInclusion


class FileSummary:
    """
    File inspection and summarization tool for checking file modification times and generating file trees with one-sentence summaries for each file.
    """

    def __init__(
        self,
        config: File1AgentConfig,
        analyze_dir: Optional[str] = None,
        summary_cache_path: Optional[str] = None,
        worker_num: int = 1,
        max_file_num: int = 200,
        **kwargs,
    ):
        """
        Initialize the file inspection and summarization tool

        Args:
            analyze_dir: Directory to analyze
            config: Configuration object
            summary_cache_path: Path to summary cache JSON file, default to "file_summary_cache.json" in analyze_dir
        """
        self.analyze_dir = analyze_dir
        self.config = config
        self.worker_num = worker_num
        self.max_file_num = max_file_num

        self.summary_cache_path = summary_cache_path or os.path.join(
            self.analyze_dir, ".f1a_cache", "file_summary_cache.json"
        )
        self.file_inclusion = FileInclusion(self.config.inclusion)

        self._load_cache()

        # Initialize the large language model using OpenAI Python SDK
        self.concluder_model = self.config.llm.chat.model

    def _load_cache(self) -> Dict:
        """
        Load file cache

        Returns:
            File cache dictionary
        """
        if os.path.exists(self.summary_cache_path):
            try:
                with open(self.summary_cache_path, "r", encoding="utf-8") as f:
                    file_summary_cache = json.load(f)
                    logger.debug(
                        f"File summary cache loaded with {len(file_summary_cache)} entries: {str(file_summary_cache)}"
                    )
                    self.file_cache = file_summary_cache

                    # Delete files that are not in file_tree
                    file_tree = self._generate_file_tree_recusive(self.analyze_dir)
                    file_tree = [fp for fp, _ in file_tree]
                    new_cache = {}
                    for fp, v in self.file_cache.items():
                        if fp in file_tree:
                            new_cache[fp] = v
                    self.file_cache = new_cache

                    return self.file_cache
            except Exception as e:
                logger.warning(f"Failed to load file summary cache: {e}")
        self.file_cache = {}
        return self.file_cache

    def _save_cache(self):
        """
        Save file cache with thread safety
        """

        # Delete files that does not exists
        for file_path in list(self.file_cache.keys()):
            if not os.path.exists(file_path):
                logger.debug(f"File {file_path} does not exist, remove from cache")
                del self.file_cache[file_path]

        try:
            cache_dir = os.path.dirname(self.summary_cache_path)
            if cache_dir:  # Ensure directory path is not empty
                os.makedirs(cache_dir, exist_ok=True)
            with open(self.summary_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.file_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save file cache: {e}")

    def _get_file_mtime(self, file_path: str) -> float:
        """
        Get file modification time

        Args:
            file_path: File path

        Returns:
            File modification timestamp
        """
        try:
            return os.path.getmtime(file_path)
        except Exception as e:
            logger.warning(f"Failed to get file modification time {file_path}: {e}")
            return 0

    def _is_file_updated(self, file_path: str) -> bool:
        """
        Check if file has been updated

        Args:
            file_path: File path

        Returns:
            True if file is new or has been updated, otherwise False
        """
        abs_path = os.path.abspath(file_path)
        current_mtime = self._get_file_mtime(abs_path)

        # Check if file modification time has been updated
        cached_mtime = self.file_cache.get(abs_path, {}).get("mtime", 0)
        logger.debug(f"Check file {abs_path}, current mtime: {current_mtime}, cached mtime: {cached_mtime}")
        return current_mtime > cached_mtime

    def _read_file_content(self, file_path: str, max_size: int = 5000) -> str:
        """
        Read file content

        Args:
            file_path: File path
            max_size: Maximum number of bytes to read

        Returns:
            File content as string
        """
        try:
            # Check file extension to handle image and PDF files
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in [".png", ".jpg", ".jpeg", ".pdf"]:
                if not self.config.llm.vision:
                    return ""

                if file_path in self.file_cache:
                    return self.file_cache[file_path]["summary"]
                for try_i in range(3):
                    try:
                        # 获取base64编码
                        base64_list = ImageConverter.get_fig_base64([file_path], merge_pdf=True)
                        if base64_list and len(base64_list) > 0:
                            # 获取第一个base64编码
                            base64_content = base64_list[0][1]
                            # 使用视觉模型进行分类和总结
                            prompt = "Please summarize in one sentence (no more than 500 characters) the main function and content of the following image:"
                            logger.debug(f"Summarize {file_path} with prompt: {prompt}")
                            vlm = OpenAIVLM(self.config.llm.vision)
                            summary = vlm.call_vlm(base64_content, prompt)
                            if summary:
                                return summary
                    except Exception as e:
                        logger.warning(f"Failed to process {file_ext} file {file_path}: {e}")
                return f"Failed to process {file_ext} file"

            # Try to read as text file
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(max_size)
                    return content
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try other encodings
                try:
                    with open(file_path, "r", encoding="gbk", errors="ignore") as f:
                        content = f.read(max_size)
                        return content
                except UnicodeDecodeError:
                    # If still fails, it's a binary file
                    logger.warning(f"{file_path} is a binary file, skipping")
                    return ""
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return ""

    def _summarize_file(self, file_path: str) -> str:
        """
        Use large language model to generate a one-sentence summary of the file

        Args:
            file_path: File path

        Returns:
            One-sentence summary of the file (no more than 200 characters)
        """
        for try_i in range(3):
            try:
                # Read file content, ensuring correct handling
                content = self._read_file_content(file_path)
                if not content:
                    return "Cannot read file content"

                # Check if this is a summary for an image or PDF file (already processed by vision model)
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in [".png", ".jpg", ".jpeg", ".pdf"]:
                    # For image and PDF files, _read_file_content already returns the vision model summary
                    return content.replace("\n", " ")

                # Limit content length to avoid excessive size
                if len(content) > 2000:
                    content = content[:2000] + "...[Content truncated]"

                file_name = os.path.basename(file_path)

                # Build a more detailed prompt including file path information
                prompt = f"Please summarize in one sentence (no more than 500 characters) the main function and content of the following file '{file_name}':\n\nFile Content:\n{content}"

                logger.debug(f"Summarizing file {file_path}...")

                concluder_llm = OpenAI(api_key=self.config.llm.chat.api_key, base_url=self.config.llm.chat.base_url)
                # Use OpenAI Python SDK to call the model
                response = concluder_llm.chat.completions.create(
                    model=self.config.llm.chat.model, messages=[{"role": "user", "content": prompt}]
                )
                summary = response.choices[0].message.content.strip()

                logger.debug(f"Summary for {file_path}: {summary}")

                # Ensure summary does not exceed 500 characters
                if len(summary) > 500:
                    summary = summary[:500] + "..."

                return summary.replace("\n", " ")
            except Exception as e:
                logger.warning(f"Failed to summarize file {file_path}: {e}")
        return f"Summary failed after {try_i} attempts"

    def get_file_summary(self, file_path: str) -> str:
        """
        Get file summary from cache or generate if not exists

        Args:
            file_path: File path

        Returns:
            File summary
        """
        abs_path = os.path.abspath(file_path)
        this_file_summary = self.file_cache.get(abs_path, {}).get("summary", "")
        if os.path.isfile(abs_path):
            if self._is_file_updated(abs_path) or len(this_file_summary) > 2000:
                summary = self._summarize_file(abs_path)
                self._update_cache(abs_path, summary)
            else:
                # Get summary from cache
                summary = this_file_summary
            return summary
        else:
            # TODO: Handle directory summary
            return ""

    def get_all_summaries(self) -> dict[str, str]:
        """
        Get all file summaries using multithreading

        Returns:
            Dictionary of file path to file summary
        """
        all_summaries = {p: self.file_cache.get(p, {}).get("summary", "") for p in self.file_cache}
        all_file_tree = self._generate_file_tree_recusive(self.analyze_dir)
        all_paths = [path for path, _ in all_file_tree]
        all_paths = all_paths[: self.max_file_num]

        # If worker_num is 1, use the original single-thread approach
        if self.worker_num <= 1:
            for file_path in all_paths:
                if os.path.isfile(file_path):
                    all_summaries[file_path] = self.get_file_summary(file_path)
        else:
            # Prepare task list
            task_list = []
            for path in all_paths:
                if os.path.isfile(path) and self._is_file_updated(path):
                    task_list.append(path)

            # Use ThreadPoolExecutor for multithreading
            with ThreadPoolExecutor(max_workers=self.worker_num) as executor:
                # Submit all tasks
                future_to_path = {executor.submit(self._summarize_file, path): path for path in task_list}

                # Process results as they complete
                for future in tqdm.tqdm(as_completed(future_to_path), total=len(task_list)):
                    file_path = future_to_path[future]
                    try:
                        summary = future.result()
                        self._update_cache(file_path, summary)
                        all_summaries[file_path] = summary
                    except Exception as e:
                        logger.warning(f"Error processing file {file_path}: {e}")
                        all_summaries[file_path] = f"Error: {str(e)}"

        self._save_cache()
        return all_summaries

    def _update_cache(self, file_path: str, summary: str):
        """
        Update file cache

        Args:
            file_path: File path
            summary: File summary
        """
        abs_path = os.path.abspath(file_path)
        self.file_cache[abs_path] = {"mtime": self._get_file_mtime(abs_path), "summary": summary}

    def _generate_file_tree_recusive(self, analyze_path: str, prefix: str = "") -> List[str]:
        """
        Generate file tree and file summaries

        Args:
            analyze_path: Path to analyze
            prefix: Prefix string

        Returns:
            List of file tree tuples (file_path, file_tree_show_line, file_summary)
        """

        file_tree = []

        # Get all items in the directory, sorted by name
        items = sorted(os.listdir(analyze_path))
        item_path_list = [os.path.abspath(os.path.join(analyze_path, item)) for item in items]

        for i, item_path in enumerate(item_path_list):
            if not self.file_inclusion.is_included(item_path):
                continue
            if len(file_tree) >= self.max_file_num:
                break

            is_last_item = i == len(items) - 1

            # Add current item to file tree
            connector = "└── " if is_last_item else "├── "
            file_tree_show_line = f"{prefix}{connector}{os.path.basename(item_path)}"

            # If it's a file, check if summary needs to be updated
            if os.path.isfile(item_path):
                # this_file_summary = self.get_file_summary(item_path)
                file_tree.append((item_path, file_tree_show_line))
            elif os.path.isdir(item_path):
                file_tree.append((item_path, file_tree_show_line))
                extension = "    " if is_last_item else "│   "
                new_tree = self._generate_file_tree_recusive(item_path, prefix + extension)
                file_tree.extend(new_tree)

        return file_tree[: self.max_file_num]

    def generate_file_tree_with_summaries(self) -> List[str]:
        """
        Generate file tree and file summaries

        Args:
            prefix: Prefix string

        Returns:
            List of file tree tuples (file_path, file_tree_show_line, file_summary)
        """

        self.get_all_summaries()
        file_tree = self._generate_file_tree_recusive(self.analyze_dir)
        file_tree_with_summaries = []
        for item in file_tree:
            file_path, file_tree_show_line = item
            file_summary = self.get_file_summary(file_path)
            file_tree_with_summaries.append((file_path, file_tree_show_line, file_summary))

        self._save_cache()
        return file_tree_with_summaries


# Example usage
if __name__ == "__main__":
    from .config import File1AgentConfig
    from ..state import load_state

    config, state, last_subgraph = load_state("../outputs/pred_aki_trend_eicu_demo")

    # Create file inspection and summarization tool
    file_summary = FileSummary(config)

    file_summary.generate_file_tree_with_summaries("../outputs/pred_aki_trend_eicu_demo")
