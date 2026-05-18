"""
gateway.py - Intent Gateway for Query Routing

Classifies user queries using fast keyword matching to decide if a query
needs full-document retrieval (summary mode) vs. focused semantic search.

"""
import logging

log = logging.getLogger("rag.gateway")


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


class IntentGateway:
    """
    Routes user queries to the appropriate RAG retrieval strategy using
    instant keyword-based matching. Zero latency overhead per query.
    """

    # embed_model is accepted for backward compatibility but not used
    def __init__(self, embed_model=None):
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
        log.info("Intent Gateway ready (keyword-only, zero latency).")

    def is_summary_request(self, query: str, threshold: float = 0.80) -> bool:
        """
        Returns True if the query requests a full document summary.
        Uses pure keyword matching — no embedding inference required.

        Args:
            query:     The user's input query.
            threshold: Unused. Kept for API compatibility only.

        Returns:
            bool: True if the query matches a known summary keyword.
        """
        normalized = _normalize_query(query)
        matched = any(kw in normalized for kw in self.summary_keywords)
        if matched:
            log.info("Gateway: SUMMARY mode triggered by keyword '%s'.",
                     next(kw for kw in self.summary_keywords if kw in normalized))
        return matched

    def is_confused(self, query: str) -> bool:
        """
        Removed. Previously detected user frustration/confusion to inject
        a patient tone modifier. This feature was removed because:
          - Keyword/embedding detection was unreliable and caused false positives
          - Tone modulation is better left to the base system prompt
        Always returns False so existing call sites in rag.py are safe.
        """
        return False
