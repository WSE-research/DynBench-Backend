import numpy as np
import pytest
from unittest.mock import patch

from utils.rdf2vec import rdf2vec_available, RDF2VecEmbedder


class TestRdf2vecAvailable:
    """Tests for rdf2vec_available function."""

    def test_returns_bool(self):
        assert isinstance(rdf2vec_available(), bool)


class TestRDF2VecEmbedder:
    """Tests for RDF2VecEmbedder (with an injected fake _fit_transform)."""

    def test_lazy_initialization(self):
        """Test that neither pyRDF2Vec nor the KG are loaded on construction."""
        embedder = RDF2VecEmbedder('https://example.org/sparql')

        assert embedder._classes is None
        assert embedder._kg is None

    def test_parameters_are_stored(self):
        embedder = RDF2VecEmbedder(
            'https://example.org/sparql',
            max_depth=3, max_walks=7, epochs=5, vector_size=64,
            skip_predicates=('http://example.org/p',), random_state=7,
        )

        assert embedder.endpoint == 'https://example.org/sparql'
        assert embedder.max_depth == 3
        assert embedder.max_walks == 7
        assert embedder.epochs == 5
        assert embedder.vector_size == 64
        assert embedder.skip_predicates == ('http://example.org/p',)
        assert embedder.random_state == 7

    def test_encode_entities_normalizes_and_aligns(self):
        """Test L2 normalization and input-order alignment of the vectors."""
        embedder = RDF2VecEmbedder('https://example.org/sparql')
        embedder._fit_transform = lambda uris: [[3.0, 4.0] if 'Q1' in u else [0.0, 2.0] for u in uris]

        result = embedder.encode_entities(['http://e/Q1', 'http://e/Q2', 'http://e/Q1'])

        assert result.shape == (3, 2)
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)
        assert np.allclose(result[0], [0.6, 0.8])
        assert np.allclose(result[1], [0.0, 1.0])
        assert np.allclose(result[2], result[0])

    def test_encode_entities_zero_vector(self):
        """Test that a zero vector does not lead to a division by zero."""
        embedder = RDF2VecEmbedder('https://example.org/sparql')
        embedder._fit_transform = lambda uris: [[0.0, 0.0] for _ in uris]

        result = embedder.encode_entities(['http://e/Q1'])

        assert np.allclose(result, [[0.0, 0.0]])

    def test_encode_entities_propagates_errors(self):
        """Test that errors from the walk extraction/training are propagated."""
        embedder = RDF2VecEmbedder('https://example.org/sparql')

        def fail(uris):
            raise ValueError('walk extraction failed')
        embedder._fit_transform = fail

        with pytest.raises(ValueError):
            embedder.encode_entities(['http://e/Q1'])


@pytest.mark.skipif(not rdf2vec_available(), reason='pyrdf2vec is not installed')
class TestRemoteKGConstruction:
    """Tests against the real pyRDF2Vec library (no network access needed)."""

    def test_sparql_endpoint_location_is_accepted(self):
        """
        pyRDF2Vec validates remote locations with a HEAD request expecting HTTP 200,
        which SPARQL endpoints like Wikidata reject. _load replaces that liveness
        check with a syntactic one, so the KG must be constructible offline.
        """
        embedder = RDF2VecEmbedder('https://query.wikidata.org/sparql')
        _, _, _, KG = embedder._load()

        kg = KG('https://query.wikidata.org/sparql', mul_req=False, skip_verify=True)

        assert kg._is_remote

    def test_user_agent_is_set_on_the_connector(self):
        """
        pyRDF2Vec sends no User-Agent, which e.g. Wikimedia endpoints reject.
        The agent must be added to the connector headers (without mutating the
        class-level default header dictionary).
        """
        embedder = RDF2VecEmbedder('https://query.wikidata.org/sparql', agent='TestAgent/1.0')

        kg = embedder._get_kg()

        assert kg.connector._headers['User-Agent'] == 'TestAgent/1.0'
        assert 'Accept' in kg.connector._headers

        other = RDF2VecEmbedder('https://query.wikidata.org/sparql', agent='Other/2.0')
        assert other._get_kg().connector._headers['User-Agent'] == 'Other/2.0'

    def test_connector_requests_the_endpoint_itself(self):
        """
        pyRDF2Vec appends '/query' to the endpoint URL, which most SPARQL endpoints
        do not support. The replacement connector must request the endpoint itself
        as defined by the SPARQL 1.1 protocol, with the configured User-Agent.
        """
        embedder = RDF2VecEmbedder('https://query.wikidata.org/sparql', agent='TestAgent/1.0')
        kg = embedder._get_kg()

        with patch('utils.rdf2vec.requests.get') as mock_get:
            response = mock_get.return_value.__enter__.return_value
            response.json.return_value = {'results': {'bindings': []}}

            result = kg.connector.fetch('SELECT ?p ?o WHERE { ?s ?p ?o }')

        url = mock_get.call_args.args[0]
        assert url.startswith('https://query.wikidata.org/sparql?query=')
        assert '/sparql/query' not in url
        assert mock_get.call_args.kwargs['headers']['User-Agent'] == 'TestAgent/1.0'
        assert result == {'results': {'bindings': []}}
