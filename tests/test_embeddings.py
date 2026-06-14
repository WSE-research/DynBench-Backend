import numpy as np
import pytest

from utils.embeddings import (
    embeddings_available,
    entity_text,
    rank_candidates,
    SentenceTransformerEmbedder,
)


class FakeEmbedder:
    """Embedder returning predefined similarities relative to the first (original) text."""

    def __init__(self, similarities: dict):
        # text -> intended cosine similarity to the original text
        self.similarities = similarities

    def encode(self, texts):
        # the original text maps to [1, 0]; a candidate with intended similarity s
        # maps to [s, sqrt(1 - s^2)], so the dot product with the original equals s
        vectors = [np.array([1.0, 0.0])]
        for text in texts[1:]:
            s = self.similarities[text]
            vectors.append(np.array([s, np.sqrt(max(0.0, 1.0 - s * s))]))
        return np.stack(vectors)


class FakeModel:
    """Stand-in for a sentence-transformers model that counts encode calls."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.calls.append(list(texts))
        return np.stack([np.array([1.0, 0.0]) for _ in texts])


class TestEntityText:
    """Tests for entity_text function."""

    def test_label_only(self):
        assert entity_text('Finland') == 'Finland'

    def test_label_and_description(self):
        result = entity_text('Finland', 'country in Northern Europe')

        assert result == 'Finland, country in Northern Europe'

    def test_no_label(self):
        assert entity_text(None) is None
        assert entity_text('') is None
        assert entity_text(None, 'some description') is None

    def test_whitespace_is_stripped(self):
        assert entity_text(' Finland ', ' country ') == 'Finland, country'

    def test_empty_description(self):
        assert entity_text('Finland', '') == 'Finland'
        assert entity_text('Finland', None) == 'Finland'


class TestRankCandidates:
    """Tests for rank_candidates function."""

    def test_sorted_by_similarity(self):
        candidates = [
            {'subst': 'wd:Q1', 'text': 'low'},
            {'subst': 'wd:Q2', 'text': 'high'},
            {'subst': 'wd:Q3', 'text': 'middle'},
        ]
        embedder = FakeEmbedder({'low': 0.4, 'high': 0.9, 'middle': 0.6})

        result = rank_candidates('original', candidates, embedder, top_k=10, min_similarity=0.0)

        assert [c['subst'] for c in result] == ['wd:Q2', 'wd:Q3', 'wd:Q1']
        assert [round(c['similarity'], 4) for c in result] == [0.9, 0.6, 0.4]

    def test_min_similarity_filter(self):
        candidates = [
            {'subst': 'wd:Q1', 'text': 'similar'},
            {'subst': 'wd:Q2', 'text': 'dissimilar'},
        ]
        embedder = FakeEmbedder({'similar': 0.8, 'dissimilar': 0.1})

        result = rank_candidates('original', candidates, embedder, min_similarity=0.3)

        assert [c['subst'] for c in result] == ['wd:Q1']

    def test_top_k_cut(self):
        candidates = [{'subst': f'wd:Q{i}', 'text': f't{i}'} for i in range(10)]
        embedder = FakeEmbedder({f't{i}': i / 10 for i in range(10)})

        result = rank_candidates('original', candidates, embedder, top_k=3, min_similarity=0.0)

        assert len(result) == 3
        assert [c['subst'] for c in result] == ['wd:Q9', 'wd:Q8', 'wd:Q7']

    def test_top_k_zero_is_unlimited(self):
        candidates = [{'subst': f'wd:Q{i}', 'text': f't{i}'} for i in range(5)]
        embedder = FakeEmbedder({f't{i}': 0.5 for i in range(5)})

        result = rank_candidates('original', candidates, embedder, top_k=0, min_similarity=0.0)

        assert len(result) == 5

    def test_empty_inputs(self):
        embedder = FakeEmbedder({})

        assert rank_candidates('original', [], embedder) == []
        assert rank_candidates(None, [{'text': 't'}], embedder) == []
        assert rank_candidates('', [{'text': 't'}], embedder) == []

    def test_candidates_are_not_mutated(self):
        candidates = [{'subst': 'wd:Q1', 'text': 'a'}]
        embedder = FakeEmbedder({'a': 0.9})

        result = rank_candidates('original', candidates, embedder)

        assert 'similarity' not in candidates[0]
        assert 'similarity' in result[0]


class TestSentenceTransformerEmbedder:
    """Tests for SentenceTransformerEmbedder (with an injected fake model)."""

    def test_lazy_initialization(self):
        embedder = SentenceTransformerEmbedder('some-model')

        assert embedder._model is None

    def test_encode_caches_per_text(self):
        embedder = SentenceTransformerEmbedder('some-model')
        embedder._model = FakeModel()

        embedder.encode(['a', 'b'])
        embedder.encode(['a', 'b'])

        assert embedder._model.calls == [['a', 'b']]

    def test_encode_only_encodes_missing_texts(self):
        embedder = SentenceTransformerEmbedder('some-model')
        embedder._model = FakeModel()

        embedder.encode(['a'])
        result = embedder.encode(['a', 'b', 'b'])

        assert embedder._model.calls == [['a'], ['b']]
        assert result.shape == (3, 2)

    def test_encode_with_caching_disabled(self):
        embedder = SentenceTransformerEmbedder('some-model', cache_size=0)
        embedder._model = FakeModel()

        embedder.encode(['a'])
        embedder.encode(['a'])

        assert embedder._model.calls == [['a'], ['a']]
        assert embedder._cache == {}

    def test_cache_eviction_keeps_current_batch_consistent(self):
        embedder = SentenceTransformerEmbedder('some-model', cache_size=2)
        embedder._model = FakeModel()

        embedder.encode(['a', 'b'])
        result = embedder.encode(['a', 'c'])

        assert result.shape == (2, 2)
        # cache was full, so it was cleared before storing the new entry
        assert 'c' in embedder._cache


class TestEmbeddingsAvailable:
    """Tests for embeddings_available function."""

    def test_returns_bool(self):
        assert isinstance(embeddings_available(), bool)
