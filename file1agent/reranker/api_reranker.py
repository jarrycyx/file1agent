import os
import json
from typing import Optional, Type, Dict, Any, Union, List
from typing_extensions import Literal
from loguru import logger

logger = logger.bind(module="file1agent_api_reranker")

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import requests
import traceback
from copy import deepcopy
import numpy as np

from ..utils.token_cnt import HumanMessage, count_tokens_approximately
from .base_reranker import BaseReranker


class APIReranker(BaseReranker):
    """
    A class for reranking documents based on their relevance to a query.

    This class provides functionality to rerank a list of documents using an external API.
    It handles chunking of large document lists, API retries, and token limit management.
    """

    def __init__(self, model: str, api_key: str, base_url: str, max_retries: int = 10, chunk_size: int = 32, worker_num: int = 1, document_max_chars: int = 2000):
        """
        Initialize the Reranker.

        Args:
            model: The name of the rerank model to use
            api_key: API key for authentication
            base_url: Base URL for the rerank API
            max_retries: Maximum number of retries for failed API calls
            chunk_size: Size of document chunks to process at once
        """
        self.worker_num = worker_num
        self.document_max_chars = document_max_chars
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.chunk_size = chunk_size

    def _prepare_query(self, query: str) -> str:
        """
        Prepare the query by truncating if it's too long.

        Args:
            query: The original query string

        Returns:
            The prepared query string
        """
        if len(query) > self.document_max_chars:
            logger.warning(f"Query is too long, truncating to {self.document_max_chars} characters: " + query[:self.document_max_chars])
            return query[:self.document_max_chars]
        return query

    def _chunk_documents(self, documents: List[str]) -> List[List[str]]:
        """
        Split documents into chunks for processing.
        Each chunk will have at most 10 documents, and each document will be
        truncated to document_max_chars if it's longer.

        Args:
            documents: List of document strings

        Returns:
            List of document chunks
        """
        # Truncate documents if they're too long
        processed_docs = []
        for doc in documents:
            if len(doc) > self.document_max_chars:
                logger.warning(f"Document is too long, truncating to {self.document_max_chars} characters")
                processed_docs.append(doc[:self.document_max_chars])
            else:
                processed_docs.append(doc)
        logger.info(f"Total {len(processed_docs)} documents, dividing to {len(processed_docs) // 10} chunks")
        
        # Split into chunks of at most 10 documents each
        chunk_size = self.chunk_size
        return [processed_docs[i : i + chunk_size] for i in range(0, len(processed_docs), chunk_size)]

    def _call_rerank_api(self, query: str, documents: List[str]) -> List[float]:
        """
        Call the rerank API with the given query and documents.

        Args:
            query: The query string
            documents: List of document strings to rerank

        Returns:
            List of relevance scores for each document
        """
        payload = {"model": self.model, "query": query, "documents": documents, "return_raw_scores": True}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        url = self.base_url + "rerank"

        scores = []
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()

                scores = []
                for res in response.json()["results"]:
                    if "relevance_score" in res:
                        scores.append(res["relevance_score"])
                    else:
                        raise ValueError("Invalid response format")

                break
            except Exception as e:
                logger.warning(f"Get rerank result error: {e}")
                logger.warning(traceback.format_exc())
                logger.warning(f"Retrying... {attempt + 1}/{self.max_retries}")
                try:
                    logger.warning(str(response))
                except:
                    pass
                time.sleep(10)
                continue

        return scores
    
    def _process_chunk(self, query: str, chunk: List[str]) -> List[float]:
        """
        Process a single chunk of documents and return relevance scores.
        This method is designed to be used in multiprocessing.
        
        Args:
            query: The query string
            chunk: A chunk of document strings to rerank
            
        Returns:
            List of relevance scores for each document in the chunk
        """
        scores = self._call_rerank_api(query, chunk)
        
        # If rerank failed, assign default scores
        if not scores:
            logger.warning("Rerank failed, setting score to 0.0")
            scores = [0.0] * len(chunk)
            
        return scores

    def _filter_by_token_limit(self, documents: List[str], scores: List[float], token_limit: int) -> List[str]:
        """
        Filter documents based on token limit.

        Args:
            documents: List of document strings
            scores: List of relevance scores
            token_limit: Maximum number of tokens to include

        Returns:
            List of document texts within the token limit
        """
        all_str = []
        current_token_cnt = 0

        for doc, score in zip(documents, scores):
            # 直接使用字符串内容计算token数量
            msg_token_cnt = count_tokens_approximately([doc])
            if current_token_cnt + msg_token_cnt > token_limit:
                break
            all_str.append(doc)
            current_token_cnt += msg_token_cnt

        return all_str

    def rerank_to_limit(self, documents: List[str], query: str, token_limit: int) -> List[str]:
        """
        Rerank documents based on their relevance to the query using threading.

        Args:
            documents: List of document strings to rerank
            query: Query string for relevance scoring
            token_limit: Maximum number of tokens to include in the result

        Returns:
            List of reranked document texts within the token limit
        """
        # Use rerank_with_scores to get sorted documents and scores
        sorted_documents, sorted_scores = self.rerank_with_scores(documents, query)
        
        # Filter by token limit
        return self._filter_by_token_limit(sorted_documents, sorted_scores, token_limit)

    def rerank_with_scores(self, documents: List[str], query: str) -> tuple[list[str], list[float]]:
        """
        Rerank documents based on their relevance to the query using threading.

        Args:
            documents: List of document strings to rerank
            query: Query string for relevance scoring

        Returns:
            Tuple of (reranked document texts, relevance scores) within the token limit
        """
        query = self._prepare_query(query)
        all_documents = []
        all_scores = []

        # Split documents into chunks if needed
        document_chunks = self._chunk_documents(documents)
        
        # Use threading to process chunks in parallel
        if self.worker_num > 1 and len(document_chunks) > 1:
            logger.info(f"Processing {len(document_chunks)} chunks with {self.worker_num} threads")
            
            # Create a thread pool executor
            with ThreadPoolExecutor(max_workers=self.worker_num) as executor:
                # Create a partial function with the query fixed
                process_func = partial(self._process_chunk, query)
                
                # Submit all tasks to the executor
                future_to_chunk = {executor.submit(process_func, chunk): chunk for chunk in document_chunks}
                
                # Process results as they complete with progress bar
                for future in tqdm(as_completed(future_to_chunk), total=len(document_chunks), desc="Processing chunks"):
                    chunk = future_to_chunk[future]
                    try:
                        scores = future.result()
                        all_documents.extend(chunk)
                        all_scores.extend(scores)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        # Add default scores for failed chunks
                        all_documents.extend(chunk)
                        all_scores.extend([0.0] * len(chunk))
        else:
            # Process sequentially if only one worker or only one chunk
            for chunk in tqdm(document_chunks, desc="Processing chunks"):
                scores = self._process_chunk(query, chunk)
                all_documents.extend(chunk)
                all_scores.extend(scores)

        # Sort by relevance score
        sorted_pairs = sorted(zip(all_documents, all_scores), key=lambda x: x[1], reverse=True)
        sorted_documents = [doc for doc, score in sorted_pairs]
        sorted_scores = [score for doc, score in sorted_pairs]

        # Log statistics
        if sorted_scores:
            logger.info(
                f"All rerank scores max: {max(sorted_scores):.3f}, min: {min(sorted_scores):.3f}, avg: {sum(sorted_scores) / len(sorted_scores):.3f}"
            )

        return sorted_documents, sorted_scores


# Backward compatibility function
def perform_rerank(
    all_docs_str: List[str], query: str, token_cnt: int, rerank_model: str, rerank_api_key: str, rerank_base_url: str
) -> List[str]:
    """
    Legacy function for backward compatibility.

    Args:
        all_docs_str: List of document strings to rerank
        query: Query string for relevance scoring
        token_cnt: Maximum number of tokens to include in the result
        rerank_model: The name of the rerank model to use
        rerank_api_key: API key for authentication
        rerank_base_url: Base URL for the rerank API

    Returns:
        List of reranked document texts within the token limit
    """
    reranker = APIReranker(model=rerank_model, api_key=rerank_api_key, base_url=rerank_base_url)
    return reranker.rerank_to_limit(all_docs_str, query, token_cnt)


"""
curl --request POST \
  --url https://cloud.infini-ai.com/maas/v1/rerank \
  --header "Authorization: Bearer $API_KEY" \
  --header "Content-Type: application/json" \
  --data '{
      "model": "bge-reranker-v2-m3",
      "query": "This log documents a cardiac arrest cohort analysis workflow that loads multiple ICU patient datasets (demographics, diagnoses, vital signs, APACHE variables) totaling millions of rows to identify and analyze cardiac arrest patients.",
      "documents": [
          "The file implements a cardiac arrest cohort workflow that creates both cardiac arrest and control patient cohorts from real patient data (with a 3:1 control ratio) or generates simulated patient data when real data is unavailable, outputting results to a specified workspace directory.",
          "The file is a CSV containing admission‑diagnosis records for ICU patients, with columns for diagnosis IDs, patient IDs, time offsets, hierarchical diagnosis paths, and diagnosis names/notes."
      ]
    }'



"""
