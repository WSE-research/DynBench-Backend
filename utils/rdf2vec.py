import logging

logger = logging.getLogger(__name__)

import operator
import threading
import zlib

from functools import partial
from importlib.util import find_spec
from urllib.parse import urlparse, quote

import numpy as np

import requests


def stable_hash(token: str) -> int:
    """
    Deterministic replacement for gensim's default hash function, which depends
    on PYTHONHASHSEED and would make the trained vectors irreproducible.
    """
    return zlib.crc32(token.encode('utf-8'))


# Wikimedia endpoints reject requests without a descriptive User-Agent
DEFAULT_AGENT = 'DynBench (https://github.com/WSE-research/DynBench-Backend)'


def rdf2vec_available() -> bool:
    """
    Check whether the pyRDF2Vec package and its embedding backend are installed.

    Returns:
        True if pyrdf2vec and gensim can be imported, False otherwise.
    """
    return find_spec('pyrdf2vec') is not None and find_spec('gensim') is not None


class RDF2VecEmbedder:
    """
    Entity embedder backed by pyRDF2Vec (https://github.com/predict-idlab/pyRDF2Vec).

    Unlike text embedders, RDF2Vec embeds knowledge graph *nodes*: random walks
    are extracted for the given entities directly from the SPARQL endpoint and a
    Word2Vec model is trained on them, so similar graph neighborhoods lead to
    similar vectors. The model is trained per call, therefore vectors are only
    comparable within one encode_entities call and are not cached.

    pyRDF2Vec is imported lazily on first use, so creating an instance is cheap
    and the web service can start without the package being installed.
    """

    def __init__(
            self,
            endpoint: str,
            max_depth: int = 2,
            max_walks: int = 10,
            epochs: int = 10,
            vector_size: int = 100,
            skip_predicates: tuple = (),
            random_state: int = 42,
            agent: str = DEFAULT_AGENT,
        ):
        """
        Initialize the embedder.

        Args:
            endpoint: URL of the SPARQL endpoint to extract walks from.
            max_depth: Maximum depth of the random walks (in hops).
            max_walks: Maximum number of walks per entity.
            epochs: Number of Word2Vec training epochs.
            vector_size: Dimensionality of the entity vectors.
            skip_predicates: Predicates (full URIs) to exclude from the walks.
            random_state: Seed for walk sampling and Word2Vec training.
            agent: User-Agent string for the walk extraction requests.
        """
        self.endpoint = endpoint
        self.max_depth = max_depth
        self.max_walks = max_walks
        self.epochs = epochs
        self.vector_size = vector_size
        self.skip_predicates = tuple(skip_predicates)
        self.random_state = random_state
        self.agent = agent
        self._classes = None
        self._connector_class = None
        self._kg = None
        self._lock = threading.Lock()

    def _load(self):
        """Import pyRDF2Vec on first use."""
        if self._classes is None:
            logger.info('Loading pyRDF2Vec...')
            from pyrdf2vec import RDF2VecTransformer
            from pyrdf2vec.embedders import Word2Vec
            from pyrdf2vec.graphs import KG
            from pyrdf2vec.walkers import RandomWalker

            # pyRDF2Vec validates remote KG locations with a bare HEAD request and
            # expects HTTP 200, but SPARQL endpoints (e.g., Wikidata) answer such
            # requests with an error status. Replace the liveness check with a
            # syntactic one; connection problems still surface on walk extraction.
            from pyrdf2vec.utils import validation

            def is_valid_url(url: str) -> bool:
                parsed = urlparse(url)
                return parsed.scheme in ('http', 'https') and bool(parsed.netloc)

            validation.is_valid_url = is_valid_url

            # pyRDF2Vec appends '/query' to the endpoint URL, a convention that
            # most SPARQL endpoints (e.g., Wikidata) do not follow; this connector
            # requests the endpoint itself as defined by the SPARQL 1.1 protocol
            from cachetools import cachedmethod
            from cachetools.keys import hashkey
            from pyrdf2vec.connectors import SPARQLConnector

            class StandardSPARQLConnector(SPARQLConnector):

                @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fetch'))
                def fetch(self, query: str):
                    url = f'{self.endpoint}?query={quote(query)}'
                    with requests.get(url, headers=self._headers) as res:
                        res.raise_for_status()
                        return res.json()

                async def _fetch(self, query: str):
                    url = f'{self.endpoint}?query={quote(query)}'
                    async with self._asession.get(url, headers=self._headers) as res:
                        return await res.json()

            self._connector_class = StandardSPARQLConnector
            self._classes = (RDF2VecTransformer, Word2Vec, RandomWalker, KG)
            logger.info('pyRDF2Vec loaded successfully.')

        return self._classes

    def _get_kg(self):
        """Build the (remote or local) knowledge graph on first use."""
        *_, KG = self._load()

        if self._kg is None:
            # the KG keeps a TTL cache of fetched hops, so it is reused between calls;
            # skip_verify avoids one ASK query per entity (pool entities are known to exist)
            self._kg = KG(
                self.endpoint,
                skip_predicates=set(self.skip_predicates),
                mul_req=False,
                skip_verify=True,
            )
            if self._kg.connector is not None:
                # standards-compliant requests (see _load), sharing the hop cache
                self._kg.connector = self._connector_class(self.endpoint, cache=self._kg.cache)
                if self.agent:
                    # pyRDF2Vec sends no User-Agent, which e.g. Wikimedia endpoints reject
                    self._kg.connector._headers = {
                        **self._kg.connector._headers,
                        'User-Agent': self.agent,
                    }

        return self._kg

    def _fit_transform(self, uris: list[str]) -> list:
        """
        Extract walks for the given entities and train a Word2Vec model on them.

        Args:
            uris: Full entity URIs (e.g., 'http://www.wikidata.org/entity/Q123').
        Returns:
            List with one (unnormalized) vector per URI, aligned with the input.
        """
        RDF2VecTransformer, Word2Vec, RandomWalker, KG = self._load()

        kg = self._get_kg()

        # a fresh transformer per call: Word2Vec must be retrained for each entity set;
        # skip-gram (sg=1) is the standard RDF2Vec choice and works better than CBOW
        # on the small per-request corpora; subsampling of frequent words (sample)
        # must be disabled on such small corpora because it would randomly drop
        # most of the walk tokens; workers=1 and the stable hash function make
        # the training reproducible
        transformer = RDF2VecTransformer(
            Word2Vec(
                vector_size=self.vector_size,
                epochs=self.epochs,
                sg=1,
                sample=0,
                workers=1,
                seed=self.random_state,
                hashfxn=stable_hash,
            ),
            walkers=[RandomWalker(
                self.max_depth,
                self.max_walks,
                n_jobs=1,
                random_state=self.random_state,
            )],
            verbose=0,
        )

        embeddings, _ = transformer.fit_transform(kg, list(uris))
        return embeddings

    def encode_entities(self, uris: list[str]) -> np.ndarray:
        """
        Encode entities into L2-normalized RDF2Vec embeddings.

        Args:
            uris: Full entity URIs (e.g., 'http://www.wikidata.org/entity/Q123').
        Returns:
            Array of shape (len(uris), vector_size) with one embedding per entity,
            aligned with the input order.
        """
        with self._lock:
            vectors = np.asarray(self._fit_transform(list(uris)), dtype=float)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms
