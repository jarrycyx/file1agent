import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union
from loguru import logger
import tqdm
import numpy as np
import re
from openai import OpenAI
import multiprocessing
from functools import partial

from loguru import logger
logger = logger.bind(module="file1agent_file_manager")

from .utils.base import File1AgentBase
from .config import File1AgentConfig, ModelConfig
from .utils.token_cnt import HumanMessage, count_tokens_approximately
from .utils.visualization import visualize_graph
from .utils.file_summary import FileSummary
from .utils.file_inclusion import FileInclusion
from .reranker.api_reranker import APIReranker


class FileManager(File1AgentBase):
    """
    File management tool for detecting duplicate files and simulated data files.
    Uses file summaries to identify potential duplicates and LLM to verify.
    """

    def __init__(
        self,
        config: Union[File1AgentConfig, str, dict, None] = None,
        analyze_dir: str = None,
        summary_cache_path: str = None,
        file_relationships_save_path: str = None,
        backup_path: str = None,
        log_level: str = "WARNING",
        log_path: str = None,
        worker_num: int = 1,
        max_file_num: int = 200,
        **kwargs,
    ):
        """
        Initialize the file management tool

        Args:
            config: Configuration object or path to TOML file or TOML string
            analyze_dir: Directory to analyze
            summary_cache_path: Path to summary cache JSON file, default to "file_summary_cache.json" in analyze_dir
            backup_path: Path to backup directory for deleted files, default to ".f1a_cache/backup" in analyze_dir
        """
        super().__init__(config, log_level=log_level, log_path=log_path, **kwargs)

        self.worker_num = worker_num
        self.max_file_num = max_file_num

        self.analyze_dir = analyze_dir
        self.backup_path = backup_path or os.path.join(analyze_dir, ".f1a_cache", "backup")
        self.file_relationships_save_path = file_relationships_save_path or os.path.join(
            analyze_dir, ".f1a_cache", "file_relationships.json"
        )
        self.file_summary = FileSummary(
            self.config,
            analyze_dir,
            summary_cache_path=summary_cache_path,
            worker_num=worker_num,
            max_file_num=max_file_num,
            **kwargs,
        )
        self.file_summary.generate_file_tree_with_summaries()
        self.file_inclusion = FileInclusion(self.config.inclusion)

        # Initialize the large language model for detailed comparison using OpenAI Python SDK
        self.comparison_llm = OpenAI(api_key=self.config.llm.chat.api_key, base_url=self.config.llm.chat.base_url)
        self.comparison_model = self.config.llm.chat.model

        # Initialize the large language model for simulated data detection using OpenAI Python SDK
        self.detection_llm = OpenAI(api_key=self.config.llm.chat.api_key, base_url=self.config.llm.chat.base_url)
        self.detection_model = self.config.llm.chat.model

        # Track deleted files
        self.deleted_files = []

        # Track duplicate file pairs
        self.duplicate_pairs = []

    def _compare_files_with_llm(self, file1_path: str, file2_path: str) -> bool:
        """
        Use LLM to compare two files and determine if they are duplicates

        Args:
            file1_path: Path to the first file
            file2_path: Path to the second file

        Returns:
            True if files are duplicates, False otherwise
        """
        try:
            # Read file contents
            content1 = self.file_summary._read_file_content(file1_path)
            content2 = self.file_summary._read_file_content(file2_path)
            
            if not content1 or not content2:
                logger.warning(f"Empty content for files {file1_path} and {file2_path}")
                return False

            file1_name = os.path.basename(file1_path)
            file2_name = os.path.basename(file2_path)

            # Build comparison prompt
            prompt = f"""
Please compare the following two files and determine if they are duplicates or implement different versions of the same functionality (e.g., one is the improved/optimized version of the other).

File 1: {file1_name}
Path: {file1_path}
Content:
{content1}

File 2: {file2_name}
Path: {file2_path}
Content:
{content2}

Please respond with concise reasons (less than 100 words) and then "Yes" if the files are duplicates or implement different versions of the same functionality, or "No" if they are different.

Example:
Reason: File 2 is the improved version of File 1, with additional features and optimizations.
Result: Yes
"""

            # Use OpenAI Python SDK to call the model
            response = self.comparison_llm.chat.completions.create(
                model=self.comparison_model, messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content.strip().upper()

            logger.debug(f"LLM comparison result for {file1_name} and {file2_name}: {result}")

            return ("Yes" in result) or ("YES" in result)
        except Exception as e:
            logger.warning(f"Failed to compare files {file1_path} and {file2_path}: {e}")
            return False

    def _detect_simulated_data(self, file_path: str) -> bool:
        """
        Use LLM to detect if a file contains simulated or fake data

        Args:
            file_path: Path to the file

        Returns:
            True if file contains simulated/fake data, False otherwise
        """
        try:
            # Read file content
            content = self.file_summary._read_file_content(file_path)

            file_name = os.path.basename(file_path)

            # Build detection prompt
            prompt = f"""
Please analyze the following file and determine if it contains simulated, fake, or mock data itself. Please only inspect this file content and do not infer from the file name or other files.

File: {file_name}
Path: {file_path}
Content:
{content}

Please respond with concise reasons (less than 100 words) and then "Yes" if the file contains simulated, fake, mock data, or "No" if it contains real data.

Example:
Reason: The line "data = np.random.rand(100, 100)" generates random data for testing purposes.
Result: Yes
"""

            # Use OpenAI Python SDK to call the model
            response = self.detection_llm.chat.completions.create(
                model=self.detection_model, messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content.strip().upper()

            logger.info(f"Simulated data detection result for {file_name}: {result.replace('\n', ' ')}")

            return ("Yes" in result) or ("YES" in result)
        except Exception as e:
            logger.warning(f"Failed to detect simulated data in {file_path}: {e}")
            return False

    def _delete_file(self, file_path: str, reason: str):
        """
        Move a file to backup directory with timestamp and record the reason

        Args:
            file_path: Path to the file to delete
            reason: Reason for deletion
        """
        try:
            # Create backup directory if it doesn't exist
            backup_dir = os.path.join(self.backup_path, "deleted")
            os.makedirs(backup_dir, exist_ok=True)

            # Get file name and extension
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1]
            base_name = os.path.splitext(file_name)[0]

            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create new file name with timestamp
            new_file_name = f"{timestamp}_{base_name}{file_ext}"
            new_file_path = os.path.join(backup_dir, new_file_name)

            # Move file to backup directory
            os.rename(file_path, new_file_path)

            self.deleted_files.append(
                {
                    "path": file_path,
                    "backup_path": new_file_path,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            logger.info(
                f"Moved file {os.path.relpath(file_path, self.analyze_dir)} to {os.path.relpath(new_file_path, self.analyze_dir)}: {reason}"
            )
        except Exception as e:
            logger.warning(f"Failed to move file {file_path}: {e}")

    def _get_file_summaries(self) -> Dict[str, str]:
        """
        Get file summaries from FileSummary for code files only

        Returns:
            Dictionary mapping file paths to their summaries
        """
        # Get file tree with summaries
        self.file_summary.generate_file_tree_with_summaries()
        result = self.file_summary.file_cache

        summaries = {}
        for path in result:
            summaries[path] = result[path]["summary"]

        return summaries

    def _find_duplicates_with_reranker(self, summaries: Dict[str, str]) -> List[Tuple[str, str]]:
        """
        Use reranker to find potential duplicate files based on summaries
        Only compares files within the same subdirectory

        Args:
            summaries: Dictionary mapping file paths to their summaries

        Returns:
            List of tuples containing paths to potential duplicate files
        """
        file_paths = list(summaries.keys())

        # For each file, compare it with all other files in the same subdirectory using reranker
        for i, file_path in enumerate(file_paths):
            confirmed_duplicates = []

            if not (os.path.abspath(os.path.join(self.analyze_dir)) in os.path.abspath(file_path)):
                continue

            if not os.path.exists(file_path):
                continue

            if not self.file_inclusion.is_included(file_path):
                continue

            # Get the parent directory of the current file
            current_dir = os.path.dirname(file_path)

            # Create a list of all other files in the same subdirectory
            other_files = []
            for path in file_paths:
                if not self.file_inclusion.is_included(path):
                    continue
                if (
                    (os.path.dirname(path) == current_dir)
                    and (path != file_path)
                    and (os.path.splitext(path)[1] == os.path.splitext(file_path)[1])
                ):
                    other_files.append(path)
            other_summaries = [summaries[path] for path in other_files]
            logger.debug(
                f"Comparing {os.path.relpath(file_path, self.analyze_dir)} with {len(other_files)} other files in the same subdirectory"
            )
            logger.debug(
                f"Comparing {os.path.relpath(file_path, self.analyze_dir)} with {[os.path.relpath(path, self.analyze_dir) for path in other_files]}"
            )

            if not other_summaries:
                continue

            # Use APIReranker to get relevance scores
            from .reranker.api_reranker import APIReranker

            query = summaries[file_path]
            if len(query) > 2000:
                logger.warning(f"Query is too long, truncating to 2000 characters: {query[:2000]}")
                query = query[:2000]

            # Create APIReranker instance
            reranker = APIReranker(
                model=self.config.rerank.rerank_model,
                api_key=self.config.rerank.rerank_api_key,
                base_url=self.config.rerank.rerank_base_url,
                worker_num=self.worker_num,
            )

            try:
                # Call rerank_with_scores method to get both reranked documents and scores
                reranked_docs, rerank_scores = reranker.rerank_with_scores(other_summaries, query)

                # Create results structure with original indices and scores
                results = []
                for i, doc in enumerate(reranked_docs):
                    # Find the original index of this document in other_summaries
                    original_index = other_summaries.index(doc)
                    results.append({"index": original_index, "relevance_score": rerank_scores[i]})

                all_rerank_scores = rerank_scores

                max_score = max(all_rerank_scores)
                min_score = min(all_rerank_scores)
                avg_score = sum(all_rerank_scores) / len(all_rerank_scores)
                med_score = np.median(all_rerank_scores)
                above_05 = sum(1 for score in all_rerank_scores if score > 0.5)
                above_08 = sum(1 for score in all_rerank_scores if score > 0.8)
                logger.debug(
                    f"All rerank scores max: {max_score:.3f}, min: {min_score:.3f}, avg: {avg_score:.3f}, median: {med_score:.3f}, num_files: {len(all_rerank_scores)}, #>0.5: {above_05}, #>0.8: {above_08}"
                )

                # Process results and filter by relevance score > threshold
                for result in results:
                    relevance_score = result["relevance_score"]
                    if relevance_score > self.config.rerank.file_duplicate_threshold:
                        # if relevance_score > 0.8:
                        # Get the index of the matching document
                        doc_index = result["index"]
                        other_file_path = other_files[doc_index]
                        logger.debug(
                            f"Found potential duplicate pair: {file_path} and {other_file_path} with score {relevance_score:.3f}, summary1: {summaries[file_path][:200]}, summary2: {summaries[other_file_path][:200]}"
                        )

                        if os.path.exists(other_file_path):
                            # Use LLM to verify if they are actually duplicates
                            if self._compare_files_with_llm(file_path, other_file_path):
                                logger.info(
                                    f"LLM confirmed duplicate pair: {file_path} and {other_file_path} with score {relevance_score:.3f}"
                                )
                                confirmed_duplicates.append((file_path, other_file_path))

            except Exception as e:
                logger.warning(f"Error calling rerank API: {e}")
                continue

            logger.info(f"Found {len(confirmed_duplicates)} confirmed duplicate pairs")

            # Remove older duplicates
            self._remove_older_duplicates(confirmed_duplicates)

    def _remove_older_duplicates(self, duplicates: List[Tuple[str, str]]):
        """
        Remove the older file from each duplicate pair

        Args:
            duplicates: List of tuples containing paths to duplicate files
        """
        for file1, file2 in duplicates:
            try:
                # Get modification times
                mtime1 = os.path.getmtime(file1)
                mtime2 = os.path.getmtime(file2)

                # Remove the older file
                if mtime1 < mtime2:
                    self._delete_file(file1, f"Older duplicate version of {file2}")
                else:
                    self._delete_file(file2, f"Older duplicate version of {file1}")
            except Exception as e:
                logger.warning(f"Failed to compare modification times for {file1} and {file2}: {e}")

    def _find_and_remove_simulated_data_files(self, summaries: Dict[str, str]):
        """
        Find and remove code files containing simulated or fake data

        Args:
            summaries: Dictionary mapping file paths to their summaries
        """
        # Filter for code files only: py, sh, c, cpp, r

        for file_path in summaries.keys():
            # Skip if file has already been deleted
            if not os.path.exists(file_path):
                continue

            # Get file extension
            _, ext = os.path.splitext(file_path)

            # Check if it's a code file
            if not self.file_inclusion.is_code_file(ext):
                continue

            if not self.file_inclusion.is_included(file_path):
                continue

            # Check if the file contains simulated data
            if self._detect_simulated_data(file_path):
                self._delete_file(file_path, "Contains simulated/fake data")

    def clean_repository(self):
        """
        Main method to clean the repository by removing duplicates and simulated data files

        Returns:
            Dictionary with information about deleted files and duplicate pairs
        """
        logger.info("Starting repository cleaning process")

        # Get file summaries
        summaries = self._get_file_summaries()
        logger.info(f"Found {len(summaries)} files to analyze")

        # Find and remove simulated data files
        self._find_and_remove_simulated_data_files(summaries)

        # Find potential duplicates using reranker
        duplicates = self._find_duplicates_with_reranker(summaries)

        logger.info("Repository cleaning process completed")

        return self.deleted_files

    def build_graph(self, visualize: bool = True) -> Dict[str, List[str]]:
        """
        Build a directed graph of file relationships based on file name references

        Returns:
            Dictionary mapping file paths to list of referenced file paths
        """

        logger.info("Building file relationship graph")

        # Get all files from file_summary
        self.file_summary.generate_file_tree_with_summaries()
        all_files = list(self.file_summary.file_cache.keys())

        # Create a dictionary to store file relationships
        file_relationships = {}

        # Determine the number of processes to use
        num_process = max(self.worker_num, os.cpu_count() // 4 - 1)
        logger.info(f"Using {num_process} processes to build file relationship graph")

        # Create a partial function with all_files as a fixed argument
        process_func = partial(
            self._process_file_for_graph,
            analyze_dir=self.analyze_dir,
            all_files=all_files,
            inclusion_config=self.config.inclusion,
        )

        all_files_with_content = [
            (file_path, self.file_summary._read_file_content(file_path)) for file_path in all_files
        ]

        # Use multiprocessing to process files in parallel
        with multiprocessing.Pool(processes=num_process) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(process_func, all_files_with_content),
                    total=len(all_files),
                    desc="Building file relationship graph",
                )
            )

        # Process the results and build the file_relationships dictionary
        for file_path, referenced_files in results:
            if referenced_files:
                file_relationships[file_path] = referenced_files

        # Save the file relationships to JSON
        self._save_relationships_to_json(file_relationships)

        # Create and save the graph visualization
        if visualize:
            if len(file_relationships) > 10:
                logger.warning(f"Only visualize the first 10 file relationships out of {len(file_relationships)}")
            visualize_graph(
                {k: v[:10] for k, v in list(file_relationships.items())[:10]},
                all_files,
                save_fig_path=os.path.splitext(self.file_relationships_save_path)[0],
            )

        logger.info(f"Built file relationship graph with {len(file_relationships)} nodes")
        return file_relationships

    def _save_relationships_to_json(self, file_relationships: Dict[str, List[str]]) -> None:
        """
        Save the file relationships dictionary to a JSON file

        Args:
            file_relationships: Dictionary mapping file paths to list of referenced file paths
        """
        try:
            # Save both the full path version and the readable version
            json_path = self.file_relationships_save_path
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(file_relationships, f, indent=2)

            logger.info(f"File relationships saved to {json_path}")
        except Exception as e:
            logger.error(f"Error saving file relationships to JSON: {e}")

    def _load_relationships_from_json(self) -> Dict[str, List[str]]:
        """
        Load the file relationships dictionary from a JSON file

        Returns:
            Dictionary mapping file paths to list of referenced file paths
        """
        try:
            json_path = self.file_relationships_save_path
            with open(json_path, "r", encoding="utf-8") as f:
                file_relationships = json.load(f)

            logger.info(f"Loaded file relationships from {json_path}")
            return file_relationships
        except FileNotFoundError:
            logger.info(f"File relationships JSON not found at {json_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading file relationships from JSON: {e}")
            return {}

    @staticmethod
    def _process_file_for_graph(
        file_path_content_tuple: Tuple[str, str],
        analyze_dir: str,
        all_files: List[str],
        inclusion_config: FileInclusion,
    ) -> Tuple[str, List[str]]:
        """
        Process a single file to find references to other files

        Args:
            file_path_content_tuple: Tuple of (file_path, file_content)
            all_files: List of all files in the project
            inclusion_config: FileInclusion instance to check file inclusion

        Returns:
            Tuple of (file_path, list of referenced files)
        """
        # Skip non-existent files
        file_inclusion = FileInclusion(inclusion_config)

        file_path, file_content = file_path_content_tuple

        if not os.path.exists(file_path):
            return (os.path.relpath(file_path, analyze_dir), [])

        if not file_inclusion.is_included(file_path):
            return (os.path.relpath(file_path, analyze_dir), [])

        # Get file extension
        _, ext = os.path.splitext(file_path)

        if file_inclusion.is_no_child_file(file_path):
            return (os.path.relpath(file_path, analyze_dir), [])

        logger.debug(f"Finding children files for: {file_path}")

        # Find references to other files
        referenced_files = []

        # For each other file, check if its name is referenced in the current file
        for other_file_path in all_files:
            if other_file_path == file_path:
                continue

            # Get just the filename without path
            other_filename = os.path.basename(other_file_path)
            other_basename = os.path.splitext(other_filename)[0]

            _, ext = os.path.splitext(other_file_path)
            if not file_inclusion.is_included(other_file_path):
                continue

            # Check if the filename or basename is referenced in the content
            # Using word boundaries to avoid partial matches
            if re.search(r"\b" + re.escape(other_filename) + r"\b", file_content):
                referenced_files.append(os.path.relpath(other_file_path, analyze_dir))
            elif re.search(r"\b" + re.escape(other_basename) + r"\b", file_content):
                referenced_files.append(os.path.relpath(other_file_path, analyze_dir))

        logger.debug(f"Found references to: {referenced_files}")
        return (os.path.relpath(file_path, analyze_dir), referenced_files)

    def search_workspace_graph(self, max_token_cnt: int = 1000, question: str = "") -> str:
        """
        Search workspace using file relationship graph to provide structured results

        Args:
            max_token_cnt: Maximum number of tokens to include in the result
            question: Optional question to rerank results based on relevance

        Returns:
            String representation of file relationships with summaries
        """
        # Build or load the file relationship graph
        file_relationships = self.build_graph()
        # TODO: only update the file relationships that are changed

        # Get file summaries
        summaries = self.file_summary.get_all_summaries()

        # Build result string with graph structure
        result = [f"File relationship graph: {self.analyze_dir}"]
        rel_result = []

        this_relation = ""
        # Add each file and its relationships
        for file_path, referenced_files in file_relationships.items():
            # Get file summary if available
            abs_file_path = os.path.abspath(os.path.join(self.analyze_dir, file_path))
            file_summary = summaries.get(abs_file_path, "No summary available")

            this_relation += f"\nFile: {file_path}"
            this_relation += f"\nSummary: {file_summary}"

            # Add referenced files
            if referenced_files:
                this_relation += "\nReferences:"
                for ref_file in referenced_files:
                    ref_path = os.path.abspath(os.path.join(self.analyze_dir, ref_file))
                    ref_summary = summaries.get(ref_path, "No summary available")
                    this_relation += f"\n  - {ref_file}: {ref_summary}"
            else:
                this_relation += "\nReferences: None"

        rel_result.append(this_relation)

        # Check token count and truncate or rerank if necessary
        if count_tokens_approximately(["\n".join(rel_result)]) > max_token_cnt:
            if question:
                logger.info(f"Reranking result with question: {question}")
                from .reranker.api_reranker import APIReranker

                reranker = APIReranker(
                    model=self.config.rerank.rerank_model,
                    api_key=self.config.rerank.rerank_api_key,
                    base_url=self.config.rerank.rerank_base_url,
                    worker_num=self.worker_num,
                )
                rel_result = reranker.rerank_to_limit(rel_result, question, max_token_cnt)
            else:
                logger.warning("No question provided for reranking. Directly truncate the result.")
                truncated = []
                current_token = 0
                for line in rel_result:
                    token_cnt = count_tokens_approximately([line])
                    if current_token + token_cnt <= max_token_cnt:
                        truncated.append(line)
                        current_token += token_cnt
                    else:
                        break

                rel_result = truncated

        result.extend(rel_result)

        return "\n".join(result)

    def search_workspace_direct(self, max_token_cnt: int = 1000, question: str = "") -> str:
        """
        Get file tree and one-sentence summary for each file directly without relationship graph

        Args:
            directory: Directory path

        Returns:
            String representation of file tree with one-sentence summaries for each file
        """

        # Generate file tree and summaries
        file_tree = self.file_summary.generate_file_tree_with_summaries()

        # Build result string
        result = [f"File tree: {self.analyze_dir}"]

        # Add summaries in file tree order
        for path, line, summary in file_tree:
            result.append(f"{line}: {summary}")

        if count_tokens_approximately(["\n".join(result)]) > max_token_cnt:
            if question:
                logger.info(f"Reranking result with question: {question}")

                reranker = APIReranker(
                    model=self.config.rerank.rerank_model,
                    api_key=self.config.rerank.rerank_api_key,
                    base_url=self.config.rerank.rerank_base_url,
                    worker_num=self.worker_num,
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

    def search_workspace(self, max_token_cnt: int = 1000, question: str = "", use_graph: bool = True) -> str:
        """
        Get file tree and one-sentence summary for each file with relationship graph

        Args:
            directory: Directory path

        Returns:
            String representation of file tree with one-sentence summaries for each file
        """
        if use_graph:
            return self.search_workspace_graph(max_token_cnt, question)
        else:
            return self.search_workspace_direct(max_token_cnt, question)


# Example usage
if __name__ == "__main__":

    config = File1AgentConfig.from_toml("outputs/pred_aki_trend_eicu_demo/config.toml")

    # Create file manager
    file_manager = FileManager(
        config,
        analyze_dir="outputs/pred_aki_trend_eicu_demo/workspace",
        summary_cache_path="outputs/pred_aki_trend_eicu_demo/file_summary_cache.json",
    )

    # Clean the repository
    result = file_manager.clean_repository()
    print(f"Deleted {len(result)} files")
    print(f"Found {len(result)} duplicate pairs")

    # Build and visualize the file relationship graph
    file_relationships = file_manager.build_graph()
    print(f"File relationships: {file_relationships}")

    # Print the paths to the saved files
    print(f"\nGraph saved to file_relation.png")
    print(f"JSON file saved to file_relationships.json")
