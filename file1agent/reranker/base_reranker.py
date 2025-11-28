from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseReranker(ABC):
    """
    Abstract base class for document rerankers.
    
    This class defines the interface that all reranker implementations should follow.
    It provides a common way to rerank documents based on their relevance to a query.
    """
    
    @abstractmethod
    def rerank_to_limit(self, documents: List[str], query: str, token_limit: int) -> List[str]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            documents: List of document strings to rerank
            query: Query string for relevance scoring
            token_limit: Maximum number of tokens to include in the result
            
        Returns:
            List of reranked document texts within the token limit
        """
        pass
    
    @abstractmethod
    def _prepare_query(self, query: str) -> str:
        """
        Prepare the query by truncating or modifying it as needed.
        
        Args:
            query: The original query string
            
        Returns:
            The prepared query string
        """
        pass
    
    @abstractmethod
    def _chunk_documents(self, documents: List[str]) -> List[List[str]]:
        """
        Split documents into chunks for processing.
        
        Args:
            documents: List of document strings
            
        Returns:
            List of document chunks
        """
        pass
    
    @abstractmethod
    def _call_rerank_api(self, query: str, documents: List[str]) -> List[float]:
        """
        Call the rerank API with the query and documents.
        
        Args:
            query: The query string
            documents: List of document strings to rerank
            
        Returns:
            List of relevance scores for each document
        """
        pass
    
    @abstractmethod
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
        pass