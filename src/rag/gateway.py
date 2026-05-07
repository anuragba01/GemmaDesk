import logging
import numpy as np

log = logging.getLogger("rag.gateway")

class IntentGateway:
    def __init__(self, embed_model):
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

    def is_summary_request(self, query: str, threshold: float = 0.80) -> bool:
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

    def _cosine_similarity(self, vec_a, vec_b):
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10)

    def is_confused(self, query: str, threshold: float = 0.85) -> bool:
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
