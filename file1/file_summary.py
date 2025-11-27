import os
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from openai import OpenAI

from .base import File1Base
from .config import File1Config
from .utils.token_cnt import HumanMessage, count_tokens_approximately
from .utils.vision import get_fig_base64
from .vision import OpenAIVLM
# Removed import of perform_rerank as we now use APIReranker directly


class FileSummary(File1Base):
    """
    File inspection and summarization tool for checking file modification times and generating file trees with one-sentence summaries for each file.
    """
    
    def __init__(self, config: Union[File1Config, str, dict, None] = None, analyze_dir: str = None, summary_cache_path: str = None, **kwargs):
        """
        Initialize the file inspection and summarization tool
        
        Args:
            analyze_dir: Directory to analyze
            config: Configuration object
            summary_cache_path: Path to summary cache JSON file, default to "file_summary_cache.json" in analyze_dir
        """
        super().__init__(config, **kwargs)
        self.analyze_dir = analyze_dir
            
        self.summary_cache_path = summary_cache_path or os.path.join(self.analyze_dir, "file_summary_cache.json")
        self.file_cache = self._load_cache()
        
        # Initialize the large language model using OpenAI Python SDK
        self.concluder_llm = OpenAI(
            api_key=self.config.llm.chat.api_key,
            base_url=self.config.llm.chat.base_url
        )
        self.concluder_model = self.config.llm.chat.model
    
    def _load_cache(self) -> Dict:
        """
        Load file cache
        
        Returns:
            File cache dictionary
        """
        if os.path.exists(self.summary_cache_path):
            try:
                with open(self.summary_cache_path, 'r', encoding='utf-8') as f:
                    file_summary_cache = json.load(f)
                    logger.debug(f"File summary cache loaded with {len(file_summary_cache)} entries: {str(file_summary_cache)}")
                    return file_summary_cache
            except Exception as e:
                logger.warning(f"Failed to load file summary cache: {e}")
        return {}
    
    def _save_cache(self):
        """
        Save file cache
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
            with open(self.summary_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.file_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save file cache: {e}")
    
    def get_file_summaries(self, file_path: str):
        try:
            abs_path = os.path.abspath(file_path)
            if self._is_file_updated(abs_path):
                summary = self._summarize_file(abs_path)
                self._update_file_cache(abs_path, summary)
            
            return self.file_cache[abs_path]
        except Exception as e:
            logger.warning(f"Failed to get file summaries {abs_path}: {e}")
            logger.warning(traceback.format_exc())
            return f"Error: Failed to get file summaries {abs_path}: {e}"
    
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
            if file_ext in ['.png', '.jpg', '.jpeg', '.pdf']:
                for try_i in range(3):
                    try:
                        # 获取base64编码
                        base64_list = get_fig_base64([file_path], merge_pdf=True)
                        if base64_list and len(base64_list) > 0:
                            # 获取第一个base64编码
                            base64_content = base64_list[0][1]
                            # 使用视觉模型进行分类和总结
                            prompt = "Please summarize in one sentence (no more than 500 characters) the main function and content of the following image:"
                            logger.info(f"Summarize {file_path} with prompt: {prompt}")
                            vlm = OpenAIVLM(self.config)
                            summary = vlm.call_vlm(base64_content, prompt)
                            if summary:
                                return summary
                    except Exception as e:
                        logger.warning(f"Failed to process {file_ext} file {file_path}: {e}")
                return f"Failed to process {file_ext} file"
            
            # Try to read as text file
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(max_size)
                    return content
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try other encodings
                try:
                    with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                        content = f.read(max_size)
                        return content
                except UnicodeDecodeError:
                    # If still fails, it's a binary file
                    logger.warning(f"{file_path} is a binary file, skipping")
                    return ""
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return f"Cannot read file: {str(e)}"
    
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
                if file_ext in ['.png', '.jpg', '.jpeg', '.pdf']:
                    # For image and PDF files, _read_file_content already returns the vision model summary
                    return content.replace("\n", " ")
                
                # Limit content length to avoid excessive size
                if len(content) > 2000:
                    content = content[:2000] + "...[Content truncated]"
                
                file_name = os.path.basename(file_path)
                
                # Build a more detailed prompt including file path information
                prompt = f"Please summarize in one sentence (no more than 500 characters) the main function and content of the following file '{file_name}':\n\nFile Path: {file_path}\n\nFile Content:\n{content}\n\nSummary:"
                
                logger.debug(f"Summarizing file {file_path}...")
            
                # Use OpenAI Python SDK to call the model
                response = self.concluder_llm.chat.completions.create(
                    model=self.concluder_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.choices[0].message.content.strip()
                    
                logger.info(f"Summary for {file_path}: {summary}")
                
                # Ensure summary does not exceed 500 characters
                if len(summary) > 500:
                    summary = summary[:500] + "..."
                    
                return summary.replace("\n", " ")
            except Exception as e:
                logger.warning(f"Failed to summarize file {file_path}: {e}")
        return f"Summary failed after {try_i} attempts"
    
    def _update_file_cache(self, file_path: str, summary: str):
        """
        Update file cache
        
        Args:
            file_path: File path
            summary: File summary
        """
        abs_path = os.path.abspath(file_path)
        self.file_cache[abs_path] = {
            "mtime": self._get_file_mtime(abs_path),
            "summary": summary
        }
    
    def _generate_file_tree(self, directory: str, prefix: str = "", is_last: bool = True, 
                           file_tree: List[str] = None, summaries: Dict[str, str] = None) -> Tuple[List[str], Dict[str, str]]:
        """
        Generate file tree and file summaries
        
        Args:
            directory: Directory path
            prefix: Prefix string
            is_last: Whether this is the last item
            file_tree: File tree list
            summaries: File summary dictionary
            
        Returns:
            Tuple of file tree list and file summary dictionary
        """
        if file_tree is None:
            file_tree = []
        if summaries is None:
            summaries = {}
            
        # Get all items in the directory, sorted by name
        items = sorted(os.listdir(directory))
        
        # Filter out hidden files and specific directories
        exclude_dirs = ['.git', '__pycache__', '.pytest_cache', 'node_modules', '.vscode', '.idea']
        exclude_files = ['.DS_Store', 'Thumbs.db']
        
        items = [item for item in items if not item.startswith('.') and item not in exclude_dirs + exclude_files]
        
        for i, item in enumerate(items):
            item_path = os.path.join(directory, item)
            is_last_item = i == len(items) - 1
            
            # Add current item to file tree
            connector = "└── " if is_last_item else "├── "
            file_tree.append(f"{prefix}{connector}{item}")
            
            # If it's a file, check if summary needs to be updated
            if os.path.isfile(item_path):
                abs_path = os.path.abspath(item_path)
                this_file_summary = self.file_cache.get(abs_path, {}).get("summary", "")
                if self._is_file_updated(item_path) or len(this_file_summary) > 2000:
                    summary = self._summarize_file(item_path)
                    self._update_file_cache(item_path, summary)
                else:
                    # Get summary from cache
                    summary = this_file_summary
                
                if summary:
                    # Use file tree line as key to ensure uniqueness
                    file_tree_line = f"{prefix}{connector}{item}"
                    summaries[file_tree_line] = summary
            
            # If it's a directory, process recursively
            elif os.path.isdir(item_path):
                extension = "    " if is_last_item else "│   "
                self._generate_file_tree(
                    item_path, 
                    prefix + extension, 
                    is_last_item, 
                    file_tree, 
                    summaries
                )
        
        return file_tree, summaries
    
    def get_file_tree_with_summaries(self, max_token_cnt: int = 1000, question: str = "") -> str:
        """
        Get file tree and one-sentence summary for each file
        
        Args:
            directory: Directory path
        
        Returns:
            String representation of file tree with one-sentence summaries for each file
        """
        
        # Generate file tree and summaries
        file_tree, summaries = self._generate_file_tree(self.analyze_dir)
        
        # Save cache
        self._save_cache()
        
        # Build result string
        result = [f"File tree: {self.analyze_dir}"]
        result.extend(file_tree)
        result.append("\nFile summaries:")
        
        # Add summaries in file tree order
        for line in file_tree:
            # 直接使用文件树行作为键查找总结
            if line in summaries:
                # Extract file name
                file_name = line.split("└── ")[-1].split("├── ")[-1].strip()
                # Add indentation to match file tree structure
                indent = line[:len(line) - len(file_name)]
                result.append(f"{indent}{file_name}: {summaries[line]}")
                
        if count_tokens_approximately(["\n".join(result)]) > max_token_cnt:
            if question:
                logger.info(f"Reranking result with question: {question}")
                from .reranker.api_reranker import APIReranker
                reranker = APIReranker(
                    model=self.config.rerank.rerank_model,
                    api_key=self.config.rerank.rerank_api_key,
                    base_url=self.config.rerank.rerank_base_url
                )
                result = reranker.rerank_to_limit(result, question, max_token_cnt)
            else:
                logger.warning("No question provided for reranking. Directly truncate the result.")
                truncated = []
                current_token = 0
                for line in result:
                    token_cnt = count_tokens_approximately([line])
                    if current_token + token_cnt <= max_token_cnt:
                        truncated.append(line)
                        current_token += token_cnt
                    else:
                        break
                
                result = truncated
        
        return "\n".join(result)


# Example usage
if __name__ == "__main__":
    from .config import File1Config
    from ..state import load_state
    
    config, state, last_subgraph = load_state("../outputs/pred_aki_trend_eicu_demo")
    
    # Create file inspection and summarization tool
    file_summary = FileSummary(config)
    
    # Get file tree and summaries
    result = file_summary.get_file_tree_with_summaries(question="Clustering machine learning")
    print(result)