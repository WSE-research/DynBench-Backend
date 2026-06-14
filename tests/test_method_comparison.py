"""
Comparison of the three substitute search methods on a shared scenario:

    sparql     - structural matching (find_substitutes)
    embedding  - sentence embeddings of label + description (find_substitutes_embedding)
    rdf2vec    - graph embeddings from random walks (find_substitutes_rdf2vec)

The mock-based tests run always and compare result structure and ranking
behavior. The offline integration test exercises the real pyRDF2Vec library on
a small local graph (skipped if pyRDF2Vec is not installed). The live tests
compare all three methods against the public Wikidata endpoint and are only
run when the environment variable RUN_LIVE_COMPARISON is set.

Run the single-record live comparison with:
    RUN_LIVE_COMPARISON=1 python -m pytest tests/test_method_comparison.py -s -k three_way

Run the bulk live comparison (defaults to ca. 1000 random resources sampled
from the bundled benchmarks; takes several hours against the public endpoint):
    RUN_LIVE_COMPARISON=1 LIVE_COMPARISON_SIZE=1000 \
        python -m pytest tests/test_method_comparison.py -s -k bulk

Per-resource results are appended to live_comparison_results.jsonl, the
aggregated statistics are written to live_comparison_summary.json.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pytest

from utils.wikidata import (
    find_substitutes,
    find_substitutes_embedding,
    find_substitutes_rdf2vec,
    WIKIDATA_PREFIX,
)
from utils.rdf2vec import rdf2vec_available, RDF2VecEmbedder

WD = 'http://www.wikidata.org/entity/'

QUERY = 'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'

INFO = {
    'resources': ['wd:Q14452'],
    'conditions': {
        'wd:Q14452': {'wdt:P31': ['?subst wdt:P31 wd:Q23442'], 'wdt:P279': []}
    },
    'query conditions': {'wd:Q14452': ['?subst wdt:P17 ?v0']}
}

# candidate pool: one semantically similar island and one dissimilar decoy that
# is structurally compatible (it satisfies the same type and query conditions)
POOL_BINDINGS = [
    {
        'subst': {'type': 'uri', 'value': WD + 'Q34497'},
        'label': {'type': 'literal', 'value': 'Saint Helena'},
        'lang': {'type': 'literal', 'value': 'en'},
        'description': {'type': 'literal', 'value': 'island in the South Atlantic'}
    },
    {
        'subst': {'type': 'uri', 'value': WD + 'Q34497'},
        'label': {'type': 'literal', 'value': 'St. Helena'},
        'lang': {'type': 'literal', 'value': 'de'}
    },
    {
        'subst': {'type': 'uri', 'value': WD + 'Q99'},
        'label': {'type': 'literal', 'value': 'California'},
        'lang': {'type': 'literal', 'value': 'en'},
        'description': {'type': 'literal', 'value': 'state of the United States'}
    },
]


def fake_execute(query):
    """Serve the entity profile and the same candidate pool for all methods."""
    if 'COALESCE' in query:
        return {
            'results': {
                'bindings': [{
                    'label': {'type': 'literal', 'value': 'Easter Island'},
                    'description': {'type': 'literal', 'value': 'island in the Pacific Ocean'}
                }]
            }
        }
    return {'results': {'bindings': POOL_BINDINGS}}


class FakeTextEmbedder:
    """Text embedder with predefined similarities to the first (original) text."""

    SIMILARITIES = {
        'Saint Helena, island in the South Atlantic': 0.9,
        'California, state of the United States': 0.1,
    }

    def encode(self, texts):
        vectors = [np.array([1.0, 0.0])]
        for text in texts[1:]:
            s = self.SIMILARITIES[text]
            vectors.append(np.array([s, np.sqrt(1.0 - s * s)]))
        return np.stack(vectors)


class FakeGraphEmbedder:
    """Graph embedder with predefined similarities to the original entity URI."""

    SIMILARITIES = {
        WD + 'Q14452': 1.0,
        WD + 'Q34497': 0.85,
        WD + 'Q99': 0.05,
    }

    def encode_entities(self, uris):
        vectors = []
        for uri in uris:
            s = self.SIMILARITIES[uri]
            vectors.append(np.array([s, np.sqrt(max(0.0, 1.0 - s * s))]))
        return np.stack(vectors)


def run_all_methods(min_similarity=0.3):
    """Run the three substitute searches on the shared scenario."""
    return {
        'sparql': find_substitutes(QUERY, fake_execute, INFO),
        'embedding': find_substitutes_embedding(
            QUERY, fake_execute, INFO, FakeTextEmbedder(), min_similarity=min_similarity),
        'rdf2vec': find_substitutes_rdf2vec(
            QUERY, fake_execute, INFO, FakeGraphEmbedder(), min_similarity=min_similarity),
    }


def substitutes_in_order(entry):
    """Unique substitutes of one method result, in result order."""
    return list(dict.fromkeys(row['subst'] for row in entry['results']))


def format_comparison(results):
    """Side-by-side table of the methods' candidates (visible with pytest -s)."""
    lines = ['', f'{"method":<12}{"rank":<6}{"substitute":<14}{"similarity":<12}label']
    for method, subs in results.items():
        for entry in subs:
            rows = {r['subst']: r for r in reversed(entry['results'])}
            for rank, subst in enumerate(substitutes_in_order(entry), 1):
                row = rows[subst]
                similarity = f"{row['similarity']:.2f}" if 'similarity' in row else '-'
                lines.append(f'{method:<12}{rank:<6}{subst:<14}{similarity:<12}{row["label"]}')
    return '\n'.join(lines)


class TestMethodComparison:
    """Compare the three substitute search methods on the shared scenario."""

    def test_all_methods_return_the_same_structure(self):
        """All methods must be drop-in compatible for the downstream pipeline."""
        results = run_all_methods()

        for method, subs in results.items():
            assert len(subs) == 1, method
            entry = subs[0]
            assert {'query', 'entity', 'extract', 'results'} <= set(entry), method
            assert entry['entity'] == 'wd:Q14452', method
            for row in entry['results']:
                assert {'subst', 'label', 'lang', 'old'} <= set(row), method
                assert row['old'] == 'wd:Q14452', method

    def test_sparql_is_unranked_and_includes_the_decoy(self):
        """The structural method returns all compatible candidates without ranking."""
        results = run_all_methods()

        rows = results['sparql'][0]['results']
        assert {r['subst'] for r in rows} == {'wd:Q34497', 'wd:Q99'}
        assert all('similarity' not in r for r in rows)

    def test_embedding_methods_filter_the_decoy(self):
        """Both embedding methods drop the dissimilar candidate via the threshold."""
        results = run_all_methods(min_similarity=0.3)

        for method in ('embedding', 'rdf2vec'):
            rows = results[method][0]['results']
            assert {r['subst'] for r in rows} == {'wd:Q34497'}, method
            assert all('similarity' in r for r in rows), method

    def test_embedding_methods_rank_by_similarity(self):
        """Without a threshold both embedding methods rank the similar candidate first."""
        results = run_all_methods(min_similarity=0.0)

        for method in ('embedding', 'rdf2vec'):
            assert substitutes_in_order(results[method][0]) == ['wd:Q34497', 'wd:Q99'], method

        # the structural method has no such guarantee: its order is arbitrary
        assert set(substitutes_in_order(results['sparql'][0])) == {'wd:Q34497', 'wd:Q99'}

    def test_methods_agree_on_the_best_substitute(self):
        """Both embedding methods choose the same top substitute in this scenario."""
        results = run_all_methods()

        tops = {
            method: substitutes_in_order(results[method][0])[0]
            for method in ('embedding', 'rdf2vec')
        }

        assert tops['embedding'] == tops['rdf2vec'] == 'wd:Q34497'

        print(format_comparison(results))


@pytest.mark.skipif(not rdf2vec_available(), reason='pyrdf2vec is not installed')
class TestRdf2vecOfflineIntegration:
    """Run the real pyRDF2Vec library on a small local graph (no network)."""

    GRAPH = str(Path(__file__).parent / 'data' / 'small_graph.ttl')
    EX = 'http://example.org/'

    def test_cluster_entities_are_more_similar(self):
        """Entities sharing their graph neighborhood must embed closer together."""
        embedder = RDF2VecEmbedder(
            self.GRAPH, max_depth=2, max_walks=100, epochs=100, vector_size=16,
        )

        vectors = embedder.encode_entities([
            self.EX + 'Finland',
            self.EX + 'Sweden',
            self.EX + 'Norway',
            self.EX + 'EasterIsland',
            self.EX + 'Pitcairn',
        ])

        finland_sweden = float(vectors[0] @ vectors[1])
        finland_norway = float(vectors[0] @ vectors[2])
        finland_easter = float(vectors[0] @ vectors[3])
        easter_pitcairn = float(vectors[3] @ vectors[4])

        assert finland_sweden > finland_easter
        assert finland_norway > finland_easter
        assert easter_pitcairn > finland_easter

    def test_vectors_are_normalized(self):
        embedder = RDF2VecEmbedder(self.GRAPH, max_walks=10, epochs=5, vector_size=8)

        vectors = embedder.encode_entities([self.EX + 'Finland', self.EX + 'Sweden'])

        assert vectors.shape == (2, 8)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


LIVE_ENDPOINT = 'https://query.wikidata.org/sparql'
LIVE_AGENT = 'DynBench/0.1 (https://github.com/WSE-research/DynBench-Backend; method comparison test)'
LIVE_PREDICATES = ('wdt:P31', 'wdt:P279')

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def make_caching_execute(raw_execute, cache, max_size=3000):
    """Cache SPARQL results in memory, so the methods share the pool queries."""
    def execute(query):
        if query in cache:
            return cache[query]
        result = raw_execute(query)
        if result is not None:
            if len(cache) >= max_size:
                cache.clear()
            cache[query] = result
        return result
    return execute


def build_info(query, entity, execute):
    """Build the query information for one entity (see core.get_info)."""
    from utils.sparql import parse_query
    from utils.sparql import get_conditions_by_predicates, get_query_conditions
    from utils.wikidata import get_resources_types

    info = {}
    info['triples'] = [i for i in parse_query(query) if all(i)]
    # restrict the substitute search to the sampled entity
    info['resources'] = [entity]
    info['types'] = get_resources_types(info, execute, LIVE_PREDICATES)
    info['conditions'] = get_conditions_by_predicates(info, LIVE_PREDICATES)
    info['query conditions'] = get_query_conditions(info)
    return info


def load_benchmark_entities(n, seed):
    """Sample up to n unique entities (with one query each) from the benchmarks."""
    from utils.sparql import normal_sparql, extract_entities

    records = []
    for name in ('DynQALD.json', 'DynRuBQ.json'):
        with open(Path(__file__).parent.parent / 'benchmarks' / name, encoding='utf-8') as f:
            records += [r['query'] for r in json.load(f) if r.get('query')]

    random.Random(seed).shuffle(records)

    samples = {}
    for query in records:
        if len(samples) >= n:
            break
        try:
            query = normal_sparql(query, WIKIDATA_PREFIX)
            for entity in extract_entities(query):
                if entity.startswith('wd:Q') and entity not in samples:
                    samples[entity] = query
        except KeyboardInterrupt:
            raise
        except Exception:
            continue

    return list(samples.items())[:n]


@pytest.mark.skipif(not os.environ.get('RUN_LIVE_COMPARISON'),
                    reason='live comparison only runs with RUN_LIVE_COMPARISON=1')
class TestLiveComparison:
    """Three-way comparison against the public Wikidata endpoint (slow)."""

    def test_live_three_way_comparison(self):
        from functools import partial

        from utils.sparql import execute as raw_execute
        from utils.sparql import normal_sparql
        from utils.embeddings import SentenceTransformerEmbedder

        execute = partial(raw_execute, endpoint_url=LIVE_ENDPOINT, agent=LIVE_AGENT, delay=1.0, timeout=60.0)

        query = normal_sparql(QUERY, WIKIDATA_PREFIX)
        info = build_info(query, 'wd:Q14452', execute)

        results = {'sparql': find_substitutes(query, execute, info)}

        if rdf2vec_available():
            results['rdf2vec'] = find_substitutes_rdf2vec(
                query, execute, info,
                RDF2VecEmbedder(LIVE_ENDPOINT, max_depth=2, max_walks=5, agent=LIVE_AGENT),
                pool_limit=25, top_k=10,
            )

        from utils.embeddings import embeddings_available
        if embeddings_available():
            results['embedding'] = find_substitutes_embedding(
                query, execute, info,
                SentenceTransformerEmbedder(EMBEDDING_MODEL),
                pool_limit=100, top_k=10,
            )

        for method, subs in results.items():
            assert subs and subs[0]['results'], f'{method} returned no substitutes'

        # sparql results are unranked and large; cut them for the printed table
        results['sparql'][0]['results'] = results['sparql'][0]['results'][:10]
        print(format_comparison(results))

    def test_live_bulk_comparison(self):
        """
        Compare the three methods on ca. 1000 random resources sampled from the
        bundled benchmarks (LIVE_COMPARISON_SIZE/LIVE_COMPARISON_SEED to adjust).

        This is a measurement harness rather than a gate: per-resource outcomes
        (including failures) are recorded in live_comparison_results.jsonl and
        aggregated into live_comparison_summary.json; the assertions only check
        that the run produced data for every method.
        """
        from functools import partial

        from utils.sparql import execute as raw_execute
        from utils.embeddings import embeddings_available, SentenceTransformerEmbedder

        size = int(os.environ.get('LIVE_COMPARISON_SIZE', '1000'))
        seed = int(os.environ.get('LIVE_COMPARISON_SEED', '42'))
        pool_limit = int(os.environ.get('LIVE_COMPARISON_POOL_LIMIT', '20'))

        samples = load_benchmark_entities(size, seed)
        assert samples, 'no entities could be sampled from the benchmarks'
        print(f'\nComparing 3 methods on {len(samples)} resources (seed {seed}, pool limit {pool_limit})...')

        raw = partial(raw_execute, endpoint_url=LIVE_ENDPOINT, agent=LIVE_AGENT, delay=0.5, timeout=20.0)
        execute = make_caching_execute(raw, {})

        text_embedder = SentenceTransformerEmbedder(EMBEDDING_MODEL) if embeddings_available() else None
        # one shared embedder: its knowledge graph caches fetched hops across resources
        graph_embedder = RDF2VecEmbedder(
            LIVE_ENDPOINT, max_depth=2, max_walks=4, agent=LIVE_AGENT,
        ) if rdf2vec_available() else None

        results_path = Path('live_comparison_results.jsonl')
        summary_path = Path('live_comparison_summary.json')

        # Resume support: if a results file already exists, keep its rows and
        # skip the entities already measured, appending the rest. The summary is
        # written only at the end, so an interrupted run is continued, not redone.
        rows = []
        done = set()
        if results_path.exists():
            with open(results_path, encoding='utf-8') as prev:
                for line in prev:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    rows.append(record)
                    done.add(record['entity'])
            if done:
                print(f'Resuming: {len(done)} resources already measured, '
                      f'{len(samples) - len(done)} to go.', flush=True)

        with open(results_path, 'a' if done else 'w', encoding='utf-8') as out:
            for index, (entity, query) in enumerate(samples, 1):
                if entity in done:
                    continue
                row = {'entity': entity, 'query': query}

                try:
                    info = build_info(query, entity, execute)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    row['error'] = f'info: {e}'
                    rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    out.flush()
                    continue

                finders = { 'sparql': lambda: find_substitutes(query, execute, info) }
                if text_embedder is not None:
                    finders['embedding'] = lambda: find_substitutes_embedding(
                        query, execute, info, text_embedder,
                        pool_limit=pool_limit, top_k=10, min_similarity=0.3)
                if graph_embedder is not None:
                    finders['rdf2vec'] = lambda: find_substitutes_rdf2vec(
                        query, execute, info, graph_embedder,
                        pool_limit=pool_limit, top_k=10, min_similarity=0.0)

                for method, finder in finders.items():
                    started = time.time()
                    try:
                        subs = finder()
                        candidates = substitutes_in_order(subs[0]) if subs else []
                        top_row = subs[0]['results'][0] if subs and subs[0]['results'] else {}
                        row[method] = {
                            'n': len(candidates),
                            'top': candidates[:10],
                            'top similarity': round(top_row['similarity'], 4) if 'similarity' in top_row else None,
                            'elapsed': round(time.time() - started, 1),
                        }
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        row[method] = { 'error': str(e), 'elapsed': round(time.time() - started, 1) }

                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                out.flush()

                if index % 10 == 0 or index == len(samples):
                    counts = { m: sum(1 for r in rows if r.get(m, {}).get('n')) for m in ('sparql', 'embedding', 'rdf2vec') }
                    print(f'[{index}/{len(samples)}] {entity}; non-empty so far: {counts}', flush=True)

        summary = summarize_bulk_comparison(rows)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(json.dumps(summary, ensure_ascii=False, indent=2))

        for method in ('sparql', 'embedding', 'rdf2vec'):
            assert summary['methods'][method]['non-empty'] > 0, f'{method} never returned substitutes'


def summarize_bulk_comparison(rows):
    """Aggregate per-resource results of the bulk comparison."""
    summary = { 'resources': len(rows), 'methods': {}, 'agreement': {} }

    for method in ('sparql', 'embedding', 'rdf2vec'):
        outcomes = [r[method] for r in rows if method in r]
        candidates = [o['n'] for o in outcomes if 'n' in o]
        similarities = [o['top similarity'] for o in outcomes if o.get('top similarity') is not None]
        elapsed = [o['elapsed'] for o in outcomes if 'elapsed' in o]
        summary['methods'][method] = {
            'run': len(outcomes),
            'non-empty': sum(1 for n in candidates if n),
            'errors': sum(1 for o in outcomes if 'error' in o),
            'mean candidates': round(float(np.mean(candidates)), 1) if candidates else None,
            'mean top similarity': round(float(np.mean(similarities)), 3) if similarities else None,
            'mean seconds': round(float(np.mean(elapsed)), 1) if elapsed else None,
        }

    # agreement between the two embedding methods, and their containment in the
    # structurally compatible (sparql) candidates
    both = [r for r in rows if r.get('embedding', {}).get('n') and r.get('rdf2vec', {}).get('n')]
    if both:
        top1 = [r['embedding']['top'][0] == r['rdf2vec']['top'][0] for r in both]
        jaccard = [
            len(set(r['embedding']['top']) & set(r['rdf2vec']['top']))
            / len(set(r['embedding']['top']) | set(r['rdf2vec']['top']))
            for r in both
        ]
        summary['agreement']['embedding vs rdf2vec'] = {
            'resources': len(both),
            'top-1 agreement': round(float(np.mean(top1)), 3),
            'mean jaccard@10': round(float(np.mean(jaccard)), 3),
        }

    for method in ('embedding', 'rdf2vec'):
        pairs = [r for r in rows if r.get(method, {}).get('n') and r.get('sparql', {}).get('n')]
        if pairs:
            contained = [r[method]['top'][0] in set(r['sparql']['top']) for r in pairs]
            summary['agreement'][f'{method} top-1 in sparql top-10'] = round(float(np.mean(contained)), 3)

    return summary
