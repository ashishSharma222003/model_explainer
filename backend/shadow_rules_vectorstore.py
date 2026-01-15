"""
Shadow Rules Vector Store Module
Manages FAISS indices for semantic search on shadow rules.
Each session has its own isolated index.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

# Lazy import for sentence-transformers and faiss to avoid startup delays
_embedding_model = None
_faiss = None

logger = logging.getLogger(__name__)

# Storage directory for FAISS indices
VECTOR_STORE_DIR = "shadow_rules_indices"

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2


def get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully")
    return _embedding_model


def get_faiss():
    """Lazy load faiss."""
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


class ShadowRuleMetadata:
    """Metadata for a shadow rule in the vector store."""
    def __init__(
        self,
        rule_id: str,
        rule_text: str,
        source_analysis: str,
        simple_rule: str = "",
        target_decision: str = "",
        predicted_outcome: str = "",
        confidence_level: str = "",
        samples_affected: int = 0,
        created_at: str = None
    ):
        self.rule_id = rule_id
        self.rule_text = rule_text  # The text used for embedding (usually simple_rule)
        self.source_analysis = source_analysis  # 'decision-tree', 'chat-discovered', 'manual'
        self.simple_rule = simple_rule or rule_text
        self.target_decision = target_decision
        self.predicted_outcome = predicted_outcome
        self.confidence_level = confidence_level
        self.samples_affected = samples_affected
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_text": self.rule_text,
            "source_analysis": self.source_analysis,
            "simple_rule": self.simple_rule,
            "target_decision": self.target_decision,
            "predicted_outcome": self.predicted_outcome,
            "confidence_level": self.confidence_level,
            "samples_affected": self.samples_affected,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadowRuleMetadata":
        return cls(
            rule_id=data.get("rule_id", ""),
            rule_text=data.get("rule_text", ""),
            source_analysis=data.get("source_analysis", "manual"),
            simple_rule=data.get("simple_rule", ""),
            target_decision=data.get("target_decision", ""),
            predicted_outcome=data.get("predicted_outcome", ""),
            confidence_level=data.get("confidence_level", ""),
            samples_affected=data.get("samples_affected", 0),
            created_at=data.get("created_at")
        )


class SimilarRuleResult:
    """Result of a similarity search."""
    def __init__(
        self,
        rule_id: str,
        rule_text: str,
        simple_rule: str,
        similarity_score: float,
        source_analysis: str,
        target_decision: str = "",
        predicted_outcome: str = "",
        is_duplicate: bool = False
    ):
        self.rule_id = rule_id
        self.rule_text = rule_text
        self.simple_rule = simple_rule
        self.similarity_score = similarity_score
        self.source_analysis = source_analysis
        self.target_decision = target_decision
        self.predicted_outcome = predicted_outcome
        self.is_duplicate = is_duplicate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_text": self.rule_text,
            "simple_rule": self.simple_rule,
            "similarity_score": self.similarity_score,
            "source_analysis": self.source_analysis,
            "target_decision": self.target_decision,
            "predicted_outcome": self.predicted_outcome,
            "is_duplicate": self.is_duplicate
        }


class ShadowRulesVectorStore:
    """
    Manages FAISS index for shadow rules per session.
    
    Each session has its own isolated index stored in:
    shadow_rules_indices/{session_id}/
        - index.faiss
        - metadata.json
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.index_dir = os.path.join(VECTOR_STORE_DIR, session_id)
        self.index_path = os.path.join(self.index_dir, "index.faiss")
        self.metadata_path = os.path.join(self.index_dir, "metadata.json")
        
        # In-memory data
        self.index = None
        self.metadata: List[ShadowRuleMetadata] = []
        self.id_to_idx: Dict[str, int] = {}  # rule_id -> index position
        
        # Load existing index or create new
        self._load_or_create()
    
    def _ensure_dir_exists(self):
        """Ensure the index directory exists."""
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
    
    def _load_or_create(self):
        """Load existing index from disk or create a new one."""
        faiss = get_faiss()
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = [ShadowRuleMetadata.from_dict(m) for m in data.get("rules", [])]
                
                # Build id_to_idx mapping
                self.id_to_idx = {m.rule_id: i for i, m in enumerate(self.metadata)}
                
                logger.info(f"Loaded vector store for session {self.session_id} with {len(self.metadata)} rules")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new empty FAISS index."""
        faiss = get_faiss()
        # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.metadata = []
        self.id_to_idx = {}
        logger.info(f"Created new vector store for session {self.session_id}")
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a text string."""
        model = get_embedding_model()
        embedding = model.encode([text], normalize_embeddings=True)[0]
        return embedding.astype('float32')
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        model = get_embedding_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.astype('float32')
    
    def add_rule(
        self,
        rule_id: str,
        rule_text: str,
        source_analysis: str,
        simple_rule: str = "",
        target_decision: str = "",
        predicted_outcome: str = "",
        confidence_level: str = "",
        samples_affected: int = 0
    ) -> bool:
        """
        Add a single shadow rule to the index.
        
        Args:
            rule_id: Unique identifier for the rule
            rule_text: The text to embed (usually the simple_rule)
            source_analysis: 'decision-tree', 'chat-discovered', or 'manual'
            simple_rule: Human-readable version of the rule
            target_decision: 'l1' or 'l2'
            predicted_outcome: 'block', 'release', or 'escalate'
            confidence_level: 'high', 'medium', or 'low'
            samples_affected: Number of samples matching this rule
            
        Returns:
            True if added successfully, False if rule_id already exists
        """
        if rule_id in self.id_to_idx:
            logger.warning(f"Rule {rule_id} already exists in index")
            return False
        
        # Generate embedding
        embedding = self._embed_text(rule_text)
        
        # Add to FAISS index
        self.index.add(np.array([embedding]))
        
        # Add metadata
        metadata = ShadowRuleMetadata(
            rule_id=rule_id,
            rule_text=rule_text,
            source_analysis=source_analysis,
            simple_rule=simple_rule or rule_text,
            target_decision=target_decision,
            predicted_outcome=predicted_outcome,
            confidence_level=confidence_level,
            samples_affected=samples_affected
        )
        self.metadata.append(metadata)
        self.id_to_idx[rule_id] = len(self.metadata) - 1
        
        # Auto-save
        self.save_index()
        
        return True
    
    def add_rules_bulk(self, rules: List[Dict[str, Any]]) -> int:
        """
        Add multiple shadow rules at once (more efficient for batch operations).
        
        Args:
            rules: List of rule dicts with keys:
                   id, text, source_analysis, simple_rule, target_decision,
                   predicted_outcome, confidence_level, samples_affected
                   
        Returns:
            Number of rules successfully added
        """
        if not rules:
            return 0
        
        # Filter out rules that already exist
        new_rules = [r for r in rules if r.get("id") not in self.id_to_idx]
        
        if not new_rules:
            return 0
        
        # Generate embeddings for all texts at once
        texts = [r.get("text", r.get("simple_rule", "")) for r in new_rules]
        embeddings = self._embed_texts(texts)
        
        # Add all to FAISS index
        self.index.add(embeddings)
        
        # Add metadata for each
        for i, rule in enumerate(new_rules):
            rule_text = rule.get("text", rule.get("simple_rule", ""))
            metadata = ShadowRuleMetadata(
                rule_id=rule.get("id", f"rule_{len(self.metadata)}"),
                rule_text=rule_text,
                source_analysis=rule.get("source_analysis", "manual"),
                simple_rule=rule.get("simple_rule", rule_text),
                target_decision=rule.get("target_decision", ""),
                predicted_outcome=rule.get("predicted_outcome", ""),
                confidence_level=rule.get("confidence_level", ""),
                samples_affected=rule.get("samples_affected", 0)
            )
            self.metadata.append(metadata)
            self.id_to_idx[metadata.rule_id] = len(self.metadata) - 1
        
        # Auto-save
        self.save_index()
        
        logger.info(f"Added {len(new_rules)} rules to index for session {self.session_id}")
        return len(new_rules)
    
    def find_similar(
        self,
        query_text: str,
        threshold: float = 0.95,
        top_k: int = 5
    ) -> List[SimilarRuleResult]:
        """
        Search for shadow rules similar to the query text.
        
        Args:
            query_text: Text to search for similar rules
            threshold: Similarity threshold for marking as duplicate (0-1)
            top_k: Maximum number of results to return
            
        Returns:
            List of SimilarRuleResult sorted by similarity (highest first)
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate embedding for query
        query_embedding = self._embed_text(query_text)
        
        # Search FAISS index
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            metadata = self.metadata[idx]
            similarity = float(score)  # Already cosine similarity due to normalized vectors
            
            result = SimilarRuleResult(
                rule_id=metadata.rule_id,
                rule_text=metadata.rule_text,
                simple_rule=metadata.simple_rule,
                similarity_score=similarity,
                source_analysis=metadata.source_analysis,
                target_decision=metadata.target_decision,
                predicted_outcome=metadata.predicted_outcome,
                is_duplicate=similarity >= threshold
            )
            results.append(result)
        
        return results
    
    def delete_rule(self, rule_id: str) -> bool:
        """
        Delete a single rule from the index.
        Note: FAISS doesn't support direct deletion, so we rebuild the index.
        
        Args:
            rule_id: ID of the rule to delete
            
        Returns:
            True if rule was deleted, False if not found
        """
        if rule_id not in self.id_to_idx:
            return False
        
        # Remove from metadata
        idx = self.id_to_idx[rule_id]
        del self.metadata[idx]
        
        # Rebuild index and mappings
        self._rebuild_index()
        
        logger.info(f"Deleted rule {rule_id} from session {self.session_id}")
        return True
    
    def delete_rules_by_source(self, source_analysis: str) -> int:
        """
        Delete ALL rules with a given source.
        Used when re-running analysis to clear old decision tree rules.
        
        Args:
            source_analysis: Source to delete ('decision-tree', 'chat-discovered', 'manual')
            
        Returns:
            Number of rules deleted
        """
        original_count = len(self.metadata)
        
        # Filter out rules with matching source
        self.metadata = [m for m in self.metadata if m.source_analysis != source_analysis]
        
        deleted_count = original_count - len(self.metadata)
        
        if deleted_count > 0:
            # Rebuild index
            self._rebuild_index()
            logger.info(f"Deleted {deleted_count} '{source_analysis}' rules from session {self.session_id}")
        
        return deleted_count
    
    def _rebuild_index(self):
        """Rebuild FAISS index from current metadata."""
        faiss = get_faiss()
        
        # Create new empty index
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.id_to_idx = {}
        
        if self.metadata:
            # Re-embed all texts
            texts = [m.rule_text for m in self.metadata]
            embeddings = self._embed_texts(texts)
            
            # Add to new index
            self.index.add(embeddings)
            
            # Rebuild id_to_idx mapping
            self.id_to_idx = {m.rule_id: i for i, m in enumerate(self.metadata)}
        
        # Save updated index
        self.save_index()
    
    def save_index(self):
        """Persist FAISS index and metadata to disk."""
        self._ensure_dir_exists()
        faiss = get_faiss()
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            data = {
                "session_id": self.session_id,
                "count": len(self.metadata),
                "updated_at": datetime.now().isoformat(),
                "rules": [m.to_dict() for m in self.metadata]
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        source_counts = {}
        for m in self.metadata:
            source = m.source_analysis
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "session_id": self.session_id,
            "total_rules": len(self.metadata),
            "decision_tree_rules": source_counts.get("decision-tree", 0),
            "chat_discovered_rules": source_counts.get("chat-discovered", 0),
            "manual_rules": source_counts.get("manual", 0),
            "index_size": self.index.ntotal if self.index else 0
        }
    
    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all rules with their metadata."""
        return [m.to_dict() for m in self.metadata]
    
    def clear_all(self):
        """Clear all rules from the index."""
        self._create_new_index()
        self.save_index()
        logger.info(f"Cleared all rules from session {self.session_id}")


# Global cache for vector stores (one per session)
_vector_store_cache: Dict[str, ShadowRulesVectorStore] = {}


def get_vector_store(session_id: str) -> ShadowRulesVectorStore:
    """Get or create a vector store for a session."""
    if session_id not in _vector_store_cache:
        _vector_store_cache[session_id] = ShadowRulesVectorStore(session_id)
    return _vector_store_cache[session_id]


def clear_vector_store_cache(session_id: str = None):
    """Clear vector store from cache (useful when session is deleted)."""
    global _vector_store_cache
    if session_id:
        if session_id in _vector_store_cache:
            del _vector_store_cache[session_id]
    else:
        _vector_store_cache = {}
