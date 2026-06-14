import logging

logger = logging.getLogger(__name__)

import threading

from importlib.util import find_spec

import numpy as np


def embeddings_available() -> bool:
    """
    Check whether the sentence-transformers package is installed.

    Returns:
        True if sentence-transformers can be imported, False otherwise.
    """
    return find_spec('sentence_transformers') is not None


def entity_text(label: str, description: str = None) -> str:
    """
    Build the canonical text representation of an entity for embedding.

    Args:
        label: Label of the entity (e.g., 'Easter Island').
        description: Optional short description of the entity
            (e.g., 'island in the Pacific Ocean, special territory of Chile').
    Returns:
        Text representation combining label and description, or None if there is no label.
    """
    if not label:
        return None

    label = label.strip()
    description = description.strip() if description else ''

    if description:
        return f'{label}, {description}'
    return label


def rank_by_similarity(
        candidates: list[dict],
        similarities,
        top_k: int = 50,
        min_similarity: float = 0.3,
    ) -> list[dict]:
    """
    Rank candidate entities by precomputed similarity scores.

    Args:
        candidates: List of dictionaries.
        similarities: Similarity score per candidate, aligned with candidates.
        top_k: Maximum number of candidates to keep (0 for unlimited).
        min_similarity: Minimum similarity for a candidate to be kept.
    Returns:
        Copies of the candidate dictionaries with an added 'similarity' key,
        sorted by similarity in descending order, filtered by min_similarity
        and cut off after top_k items.
    """
    ranked = [c | { 'similarity': float(s) } for c, s in zip(candidates, similarities)]
    ranked = [c for c in ranked if c['similarity'] >= min_similarity]
    ranked.sort(key=lambda c: c['similarity'], reverse=True)

    if top_k > 0:
        ranked = ranked[:top_k]

    return ranked


def rank_candidates(
        original_text: str,
        candidates: list[dict],
        embedder,
        top_k: int = 50,
        min_similarity: float = 0.3,
    ) -> list[dict]:
    """
    Rank candidate entities by embedding similarity to the original entity.

    Args:
        original_text: Text representation of the original entity (see entity_text).
        candidates: List of dictionaries, each containing at least a 'text' key.
        embedder: Object with an encode(list[str]) -> np.ndarray method returning
            L2-normalized embeddings (one row per input text).
        top_k: Maximum number of candidates to keep (0 for unlimited).
        min_similarity: Minimum cosine similarity for a candidate to be kept.
    Returns:
        Copies of the candidate dictionaries with an added 'similarity' key,
        sorted by similarity in descending order, filtered by min_similarity
        and cut off after top_k items.
    """
    if not original_text or not candidates:
        return []

    texts = [c['text'] for c in candidates]
    vectors = embedder.encode([original_text] + texts)

    # embeddings are L2-normalized, so the dot product equals cosine similarity
    similarities = vectors[1:] @ vectors[0]

    return rank_by_similarity(candidates, similarities, top_k=top_k, min_similarity=min_similarity)


class SentenceTransformerEmbedder:
    """
    Embedder backed by a sentence-transformers model.

    The model is loaded lazily on first use, so creating an instance is cheap
    and the web service can start before the model weights are available.
    Embeddings are L2-normalized and cached per text.
    """

    def __init__(self, model_name: str, device: str = None, cache_size: int = 100_000):
        """
        Initialize the embedder.

        Args:
            model_name: Name or path of the sentence-transformers model,
                e.g. 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'.
            device: Device to run the model on (e.g., 'cpu', 'cuda'). None for auto-detection.
            cache_size: Maximum number of cached text embeddings (0 to disable caching).
        """
        self.model_name = model_name
        self.device = device
        self.cache_size = max(0, cache_size)
        self._model = None
        self._cache = {}
        self._lock = threading.Lock()

    def _load(self):
        """Load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f'Loading embedding model "{self.model_name}"...')
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info('Embedding model loaded successfully.')

        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into L2-normalized embeddings.

        Args:
            texts: List of texts to encode.
        Returns:
            Array of shape (len(texts), dim) with one embedding per text.
        """
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._cache]

            new_vectors = {}
            if missing:
                model = self._load()
                encoded = model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
                new_vectors = { t: np.asarray(v) for t, v in zip(missing, encoded) }

            lookup = { t: self._cache.get(t, new_vectors.get(t)) for t in texts }

            if new_vectors and self.cache_size:
                if len(self._cache) + len(new_vectors) > self.cache_size:
                    self._cache.clear()
                self._cache.update(new_vectors)

            return np.stack([lookup[t] for t in texts])
