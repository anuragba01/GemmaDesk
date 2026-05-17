"""
gateway.py - Intent Gateway for Query Routing

This module provides the IntentGateway class which classifies user queries 
into different intents (e.g., summary requests, expressions of confusion)
using embedding-based cosine similarity.
"""
import logging
import numpy as np

log = logging.getLogger("rag.gateway")


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())

class IntentGateway:
    """
    Gateway to classify user intent based on queries to route them to the appropriate RAG logic.
    Uses embeddings to detect requests for summarization or expressions of confusion.
    """

    def __init__(self, embed_model):
        """
        Initializes the IntentGateway with an embedding model and precomputes embeddings for trigger phrases.
        
        Args:
            embed_model: The embedding model instance used to embed documents and queries.
        """
        self.embed_model = embed_model
        log.info("Initializing Intent Gateway with Nomic...")
        
        # Define trigger phrases for confusion/frustration
        self.confusion_triggers = [
            "I am confused",
            "This makes no sense",
            "I don't understand this",
            "Explain this simply",
            "This is frustrating",
            "What does this mean",
            "Too hard to understand"
        ]
        
        # Define trigger phrases for summary/full document requests
        self.summary_triggers = [
            "Summarize this",
            "Give me a summary",
            "Tell me everything about",
            "Provide a comprehensive overview",
            "What is this document about",
            "Explain the whole thing",
            "Give me a high level overview",
            "Break down this entire material",
            "give me real life use",
            "what do i need to study before study the material",
            "tell me what is the key point of this document",
            "what is prerequisite for this document"
        ]
        self.summary_keywords = (
            "summary",
            "summarize",
            "overview",
            "key point",
            "key points",
            "whole document",
            "whole thing",
            "entire material",
            "what is this document about",
            "tell me everything about",
            "prerequisite",
            "real life use",
        )
        self.confusion_keywords = (
            "confused",
            "makes no sense",
            "don't understand",
            "do not understand",
            "explain simply",
            "too hard",
            "what does this mean",
            "frustrating",
        )
        
        # Pre-compute embeddings for triggers
        try:
            # Confusion embeddings
            self.confusion_embeddings = np.array(self.embed_model.embed_documents(self.confusion_triggers))
            norms_c = np.linalg.norm(self.confusion_embeddings, axis=1, keepdims=True)
            self.confusion_embeddings = self.confusion_embeddings / np.where(norms_c == 0, 1e-10, norms_c)
            
            # Summary embeddings
            self.summary_embeddings = np.array(self.embed_model.embed_documents(self.summary_triggers))
            norms_s = np.linalg.norm(self.summary_embeddings, axis=1, keepdims=True)
            self.summary_embeddings = self.summary_embeddings / np.where(norms_s == 0, 1e-10, norms_s)
            
        except Exception as e:
            log.error(f"Failed to initialize gateway embeddings: {e}")
            self.confusion_embeddings = None
            self.summary_embeddings = None

    def _matches_keywords(self, query: str, keywords: tuple[str, ...]) -> bool:
        normalized = _normalize_query(query)
        return any(keyword in normalized for keyword in keywords)

    def is_summary_request(self, query: str, threshold: float = 0.80) -> bool:
        """
        Checks if the user query is a request for a document summary.
        
        Args:
            query (str): The user's input query.
            threshold (float, optional): The cosine similarity threshold for classification. Defaults to 0.80.
            
        Returns:
            bool: True if the query indicates a summary request, False otherwise.
        """
        if self._matches_keywords(query, self.summary_keywords):
            log.info("Gateway triggered SUMMARY mode via keyword rule.")
            return True
        if self.summary_embeddings is None:
            return False
        
        try:
            q_emb = np.array(self.embed_model.embed_query(query))
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
            
            similarities = np.dot(self.summary_embeddings, q_emb)
            max_sim = np.max(similarities)
            
            if max_sim > threshold:
                log.info(f"Gateway triggered SUMMARY mode (sim: {max_sim:.2f})")
                return True
            return False
        except Exception as e:
            log.warning(f"Gateway summary check failed: {e}")
            return False

    def is_confused(self, query: str, threshold: float = 0.85) -> bool:
        """
        Checks if the user query indicates confusion or frustration.
        
        Args:
            query (str): The user's input query.
            threshold (float, optional): The cosine similarity threshold for classification. Defaults to 0.85.
            
        Returns:
            bool: True if the query indicates confusion, False otherwise.
        """
        if self._matches_keywords(query, self.confusion_keywords):
            log.info("Gateway triggered CONFUSION mode via keyword rule.")
            return True
        if self.confusion_embeddings is None:
            return False
        
        try:
            # Prefix for Nomic query
            from rag.prompts import EMBED_QUERY_PREFIX
            query_embed = self.embed_model.embed_query(EMBED_QUERY_PREFIX + query)
            query_vec = np.array(query_embed)
            
            # Normalize query vector
            query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            
            # Compute similarities with all triggers
            similarities = np.dot(self.confusion_embeddings, query_vec)
            max_sim = np.max(similarities)
            
            log.info(f"Gateway Max Similarity for '{query}': {max_sim:.2f}")
            return max_sim > threshold
        except Exception as e:
            log.warning(f"Gateway classification failed: {e}")
            return False
