import pytest
from unittest.mock import Mock, patch
import json

import numpy as np

from utils.wikidata import (
    query_wikidata_label,
    get_wikidata_label,
    get_resources_types,
    find_substitutes,
    find_substitutes_embedding,
    build_substitutes_query,
    build_pool_condition_sets,
    get_legacy_conditions,
    get_entity_profile,
    check_productivity_single,
    WIKIDATA_PREFIX,
    FIXED_LABELS,
    LABEL_LANGS
)


class TestQueryWikidataLabel:
    """Tests for query_wikidata_label function."""

    def test_with_fixed_labels(self):
        """Test that fixed labels are used when provided."""
        uri = 'http://www.w3.org/2000/01/rdf-schema#label'
        fixed_labels = {'http://www.w3.org/2000/01/rdf-schema#label': 'rdfs:label'}
        
        result = query_wikidata_label(uri, Mock(), fixed_labels=fixed_labels)
        
        assert result == 'rdfs:label'

    def test_with_none_fixed_labels(self):
        """Test that None fixed_labels doesn't cause issues."""
        uri = 'wd:Q123'
        
        mock_execute = Mock(return_value={
            'results': {'bindings': [{'label': {'value': 'Test Label'}}]}
        })
        
        result = query_wikidata_label(uri, mock_execute, fixed_labels=None)
        
        assert result == 'Test Label'

    def test_successful_label_query(self):
        """Test successful label query with English language."""
        uri = 'wd:Q123'

        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'label': {'type': 'literal', 'value': 'Test Entity'}}]
            }
        })
        
        result = query_wikidata_label(uri, mock_execute)
        
        assert result == 'Test Entity'

    def test_label_query_with_default_language(self):
        """Test label query falls back to default language."""
        uri = 'wd:Q123'
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'label': {'value': 'Default Label'}}]
            }
        })
        
        result = query_wikidata_label(uri, mock_execute, lang='fr')
        
        assert result == 'Default Label'

    def test_label_query_with_redirect(self):
        """Test label query with redirect handling."""
        uri = 'wd:Q123'
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'label': {'value': 'Redirected Label'}}]
            }
        })
        
        result = query_wikidata_label(uri, mock_execute)
        
        assert result == 'Redirected Label'

    def test_label_query_no_results(self):
        """Test label query with no results."""
        uri = 'wd:Q123'
        
        mock_execute = Mock(return_value = {'results': {'bindings': []}})
        
        result = query_wikidata_label(uri, mock_execute)
        
        assert result is None

    def test_label_query_exception_handling(self):
        """Test exception handling in label query."""
        uri = 'wd:Q123'
        
        mock_execute = Mock(error=True)

        result = query_wikidata_label(uri, mock_execute)
        
        assert result is None

    def test_label_query_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is re-raised."""
        uri = 'wd:Q123'
        mock_execute = Mock(side_effect=KeyboardInterrupt())
        
        with pytest.raises(KeyboardInterrupt):
            query_wikidata_label(uri, mock_execute)


class TestGetWikidataLabel:
    """Tests for get_wikidata_label function."""

    def test_successful_label_with_prefix(self):
        """Test getting label with prefix conversion."""
        uri = 'http://www.wikidata.org/entity/Q123'
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'label': {'value': 'Test Entity'}}]
            }
        })
        
        result = get_wikidata_label(uri, mock_execute)
        
        assert result == 'Test Entity'

    def test_get_label_with_custom_prefixes(self):
        """Test getting label with custom prefixes."""
        uri = 'http://www.wikidata.org/entity/Q456'
        custom_prefixes = WIKIDATA_PREFIX.copy()
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'label': {'value': 'Custom Prefix Entity'}}]
            }
        })
        
        result = get_wikidata_label(uri, mock_execute)
        
        assert result == 'Custom Prefix Entity'

    def test_get_label_exception_handling(self):
        """Test exception handling in get_wikidata_label."""
        uri = 'http://www.wikidata.org/entity/Q123'
        
        mock_execute = Mock(error=True)

        result = get_wikidata_label(uri, mock_execute)
        
        assert result is None


class TestGetResourcesTypes:
    """Tests for get_resources_types function."""

    def test_empty_predicates(self):
        """Test with empty predicates list."""
        info = {'resources': ['wd:Q123']}
        result = get_resources_types(info, Mock(), predicates=[])
        
        assert result == {}

    def test_non_wikidata_resources(self):
        """Test filtering of non-Wikidata resources."""
        info = {'resources': ['wd:Q123', 'dbpedia:Res1', 'wd:Q456']}
        predicates = ['wdt:P31', 'wdt:P279']
        
        mock_execute = Mock(return_value={
            'results': {
                'bindings': [
                    {'p': {'type': 'uri', 'value': 'wdt:P31'}, 'o': {'type': 'uri', 'value': 'wd:Q1'}},
                    {'p': {'type': 'uri', 'value': 'wdt:P279'}, 'o': {'type': 'uri', 'value': 'wd:Q2'}}
                ]
            }
        })
        
        result = get_resources_types(info, mock_execute, predicates=predicates)
        
        assert 'wd:Q123' in result
        assert 'dbpedia:Res1' not in result
        assert 'wd:Q456' in result

    def test_successful_type_collection(self):
        """Test successful type collection."""
        info = {'resources': ['wd:Q123']}
        predicates = ['wdt:P31', 'wdt:P279']
        
        mock_execute = Mock(return_value={
            'results': {
                'bindings': [
                    {'p': {'type': 'uri', 'value': 'wdt:Q123'}, 'o': {'type': 'uri', 'value': 'wd:Q1'}},
                    {'p': {'type': 'uri', 'value': 'wdt:Q456'}, 'o': {'type': 'uri', 'value': 'wd:Q2'}}
                ]
            }
        })
        
        result = get_resources_types(info, mock_execute, predicates=predicates)
        
        assert 'wd:Q123' in result
        assert 'wdt:P31' in result['wd:Q123']
        assert 'wdt:P279' in result['wd:Q123']

    def test_type_collection_exception_handling(self):
        """Test exception handling in type collection."""
        info = {'resources': ['wd:Q123']}
        predicates = ['wdt:P31']
        
        mock_execute = Mock(side_effect = Exception("SPARQL error"))
        
        result = get_resources_types(info, mock_execute, predicates=predicates)
        
        assert result == {'wd:Q123': {}}

    def test_type_collection_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is re-raised."""
        info = {'resources': ['wd:Q123']}
        predicates = ['wdt:P31']
        mock_execute = Mock(side_effect=KeyboardInterrupt())
        
        with pytest.raises(KeyboardInterrupt):
            get_resources_types(info, mock_execute, predicates=predicates)


class TestFindSubstitutes:
    """Tests for find_substitutes function."""

    def test_empty_entities(self):
        """Test with no Wikidata entities."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        info = {'resources': ['dbpedia:Res1'], 'conditions': {}, 'query conditions': {}}
        
        result = find_substitutes(query, Mock(), info)
        
        assert result == []

    def test_successful_substitute_finding(self):
        """Test successful substitute finding."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        info = {
            'resources': ['wd:Q123'],
            'conditions': {
                'wd:Q123': {
                    'wdt:P31': ['wd:Q1'],
                    'wdt:P279': ['wd:Q2']
                }
            },
            'query conditions': {
                'wd:Q123': ['wdt:P31']
            }
        }
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [
                    {
                        'subst': {'type': 'uri', 'value': 'wd:Q456'},
                        'label': {'type': 'literal', 'value': 'Substitute'},
                        'lang': {'type': 'literal', 'value': 'en'}
                    }
                ]
            }
        })
        
        result = find_substitutes(query, mock_execute, info)
        
        assert len(result) == 1
        assert result[0]['entity'] == 'wd:Q123'
        assert len(result[0]['results']) == 1
        assert result[0]['results'][0]['old'] == 'wd:Q123'

    def test_substitute_finding_exception_handling(self):
        """Test exception handling in substitute finding."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        info = {
            'resources': ['wd:Q123'],
            'conditions': {
                'wd:Q123': {
                    'wdt:P31': [],
                    'wdt:P279': []
                }
            },
            'query conditions': {
                'wd:Q123': []
            }
        }
        
        mock_execute = Mock(side_effect = Exception("SPARQL error"))
        
        result = find_substitutes(query, mock_execute, info)
        
        assert result == []

    def test_substitute_finding_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is re-raised."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        info = {
            'resources': ['wd:Q123'],
            'conditions': {
                'wd:Q123': {
                    'wdt:P31': [],
                    'wdt:P279': []
                }
            },
            'query conditions': {
                'wd:Q123': []
            }
        }
        mock_execute = Mock(side_effect=KeyboardInterrupt())
        
        with pytest.raises(KeyboardInterrupt):
            find_substitutes(query, mock_execute, info)


class TestCheckProductivitySingle:
    """Tests for check_productivity_single function."""

    def test_productive_query(self):
        """Test with productive query returning results."""
        query = "SELECT ?x WHERE { ?x ?p wd:Q123 }"
        replace = {'old': 'wd:Q123', 'new': 'wd:Q456'}
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'?x': {'type': 'literal', 'value': 'test'}}]
            }
        })
        
        result = check_productivity_single(query, mock_execute, replace)
        
        assert result is True

    def test_non_productive_query(self):
        """Test with non-productive query returning no results."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        replace = {'old': '?x', 'new': 'wd:Q123'}
        
        mock_execute = Mock(return_value = {'results': {'bindings': []}})
        
        result = check_productivity_single(query, mock_execute, replace)
        
        assert result is False

    def test_query_with_default_replace(self):
        """Test query with default replace dictionary."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        
        mock_execute = Mock(return_value = { 'results': {'bindings': []} })
        
        result = check_productivity_single(query, mock_execute, {})
        
        assert result is False

    def test_productivity_check_exception_handling(self):
        """Test exception handling in productivity check."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        replace = {'old': '?x', 'new': 'wd:Q123'}
        
        mock_execute = Mock(side_effect = Exception("SPARQL error"))
        
        result = check_productivity_single(query, mock_execute, replace)
        
        assert result is False

    def test_custom_prefixes(self):
        """Test with custom prefixes."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        replace = {'old': '?x', 'new': 'wd:Q123'}
        custom_prefixes = {'custom': 'http://custom.namespace/'}

        mock_execute = Mock(return_value = { 'results': { 'bindings': [] } })

        result = check_productivity_single(query, mock_execute, replace, custom_prefixes)

        assert result is False


class TestBuildSubstitutesQuery:
    """Tests for build_substitutes_query function."""

    def test_legacy_query_format(self):
        """Test that the default parameters reproduce the legacy SPARQL-based query."""
        query = build_substitutes_query('wd:Q123', ['?subst wdt:P31 wd:Q1'])

        expected = '\n'.join((
            'SELECT DISTINCT ?subst ?label (lang(?label) as ?lang) WHERE {',
            '?subst wdt:P31 wd:Q1 .',
            '    ?subst rdfs:label ?label .',
            '    FILTER (lang(?label) IN ("en", "de", "fr", "ru", "uk"))',
            '    FILTER(?subst != wd:Q123)',
            '} LIMIT 1000',
        ))

        assert query == expected

    def test_query_with_description(self):
        """Test query with description selection enabled."""
        query = build_substitutes_query('wd:Q123', ['?subst wdt:P31 wd:Q1'], limit=500, description_lang='en')

        assert 'SELECT DISTINCT ?subst ?label (lang(?label) as ?lang) ?description WHERE {' in query
        assert 'OPTIONAL { ?subst schema:description ?description . FILTER(lang(?description) = "en") }' in query
        assert query.endswith('} LIMIT 500')

    def test_custom_label_languages(self):
        """Test query with custom label languages."""
        query = build_substitutes_query('wd:Q123', [], label_langs=('en', 'de'))

        assert 'FILTER (lang(?label) IN ("en", "de"))' in query

    def test_query_ordered_by_sitelinks(self):
        """Test that the popularity-ordered query selects candidates in a subquery."""
        query = build_substitutes_query(
            'wd:Q123', ['?subst wdt:P31 wd:Q1'],
            limit=100, description_lang='en', order_by_sitelinks=True,
        )

        assert 'SELECT DISTINCT ?subst ?sitelinks WHERE {' in query
        assert '?subst wikibase:sitelinks ?sitelinks .' in query
        assert '} ORDER BY DESC(?sitelinks) LIMIT 100' in query
        assert 'FILTER(?subst != wd:Q123)' in query
        assert 'OPTIONAL { ?subst schema:description ?description . FILTER(lang(?description) = "en") }' in query


class TestGetLegacyConditions:
    """Tests for get_legacy_conditions function."""

    def test_combines_query_and_type_conditions(self):
        """Test that query conditions and the first two type conditions are combined."""
        info = {
            'conditions': {
                'wd:Q123': {
                    'wdt:P31': ['?subst wdt:P31 wd:Q1', '?subst wdt:P31 wd:Q2'],
                    'wdt:P279': ['?subst wdt:P279 wd:Q3']
                }
            },
            'query conditions': {
                'wd:Q123': ['?subst wdt:P17 ?v0']
            }
        }

        result = get_legacy_conditions('wd:Q123', info)

        assert result == ['?subst wdt:P17 ?v0', '?subst wdt:P31 wd:Q1', '?subst wdt:P31 wd:Q2']


class TestGetEntityProfile:
    """Tests for get_entity_profile function."""

    def test_successful_profile(self):
        """Test successful label and description retrieval."""
        mock_execute = Mock(return_value={
            'results': {
                'bindings': [{
                    'label': {'type': 'literal', 'value': 'Easter Island'},
                    'description': {'type': 'literal', 'value': 'island in the Pacific Ocean'}
                }]
            }
        })

        result = get_entity_profile('wd:Q14452', mock_execute)

        assert result == {'label': 'Easter Island', 'description': 'island in the Pacific Ocean'}

    def test_profile_without_description(self):
        """Test profile when no description is available."""
        mock_execute = Mock(return_value={
            'results': {'bindings': [{'label': {'type': 'literal', 'value': 'Easter Island'}}]}
        })

        result = get_entity_profile('wd:Q14452', mock_execute)

        assert result == {'label': 'Easter Island', 'description': None}

    def test_profile_no_results(self):
        """Test profile when the query returns no results."""
        mock_execute = Mock(return_value={'results': {'bindings': []}})

        result = get_entity_profile('wd:Q14452', mock_execute)

        assert result is None

    def test_profile_exception_handling(self):
        """Test exception handling in profile retrieval."""
        mock_execute = Mock(side_effect=Exception('SPARQL error'))

        result = get_entity_profile('wd:Q14452', mock_execute)

        assert result is None

    def test_profile_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is re-raised."""
        mock_execute = Mock(side_effect=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            get_entity_profile('wd:Q14452', mock_execute)


class TestBuildPoolConditionSets:
    """Tests for build_pool_condition_sets function."""

    def test_strict_and_relaxed_sets(self):
        """Test that the strict combination and per-type relaxed sets are built."""
        info = {
            'conditions': {
                'wd:Q123': {
                    'wdt:P31': ['?subst wdt:P31 wd:Q1', '?subst wdt:P31 wd:Q2'],
                    'wdt:P279': []
                }
            },
            'query conditions': {
                'wd:Q123': ['?subst wdt:P17 ?v0']
            }
        }

        sets = build_pool_condition_sets('wd:Q123', info)

        # strict combination plus one relaxed set per type condition
        assert sets == [
            ['?subst wdt:P17 ?v0', '?subst wdt:P31 wd:Q1', '?subst wdt:P31 wd:Q2'],
            ['?subst wdt:P17 ?v0', '?subst wdt:P31 wd:Q1'],
            ['?subst wdt:P17 ?v0', '?subst wdt:P31 wd:Q2'],
        ]

    def test_single_type_condition_is_deduplicated(self):
        """Test that with one type condition the strict and relaxed sets coincide."""
        info = {
            'conditions': {
                'wd:Q123': {'wdt:P31': ['?subst wdt:P31 wd:Q1'], 'wdt:P279': []}
            },
            'query conditions': {'wd:Q123': []}
        }

        sets = build_pool_condition_sets('wd:Q123', info)

        assert sets == [['?subst wdt:P31 wd:Q1']]

    def test_no_conditions(self):
        """Test that no sets are built without any conditions."""
        info = {
            'conditions': {'wd:Q123': {'wdt:P31': [], 'wdt:P279': []}},
            'query conditions': {'wd:Q123': []}
        }

        sets = build_pool_condition_sets('wd:Q123', info)

        assert sets == []

    def test_max_type_conditions(self):
        """Test that the number of relaxed sets is limited."""
        info = {
            'conditions': {
                'wd:Q123': {
                    'wdt:P31': [f'?subst wdt:P31 wd:Q{i}' for i in range(1, 6)],
                    'wdt:P279': []
                }
            },
            'query conditions': {'wd:Q123': []}
        }

        sets = build_pool_condition_sets('wd:Q123', info, max_type_conditions=2)

        # strict combination plus two relaxed sets
        assert len(sets) == 3


class FakeEmbedder:
    """Embedder returning predefined similarities relative to the first (original) text."""

    def __init__(self, similarities: dict):
        self.similarities = similarities

    def encode(self, texts):
        vectors = [np.array([1.0, 0.0])]
        for text in texts[1:]:
            s = self.similarities[text]
            vectors.append(np.array([s, np.sqrt(max(0.0, 1.0 - s * s))]))
        return np.stack(vectors)


class TestFindSubstitutesEmbedding:
    """Tests for find_substitutes_embedding function."""

    INFO = {
        'resources': ['wd:Q14452'],
        'conditions': {
            'wd:Q14452': {'wdt:P31': ['?subst wdt:P31 wd:Q23442'], 'wdt:P279': []}
        },
        'query conditions': {'wd:Q14452': ['?subst wdt:P17 ?v0']}
    }

    @staticmethod
    def fake_execute(query):
        """Return a profile for the original entity and a pool for the candidate queries."""
        if 'COALESCE' in query:
            return {
                'results': {
                    'bindings': [{
                        'label': {'type': 'literal', 'value': 'Easter Island'},
                        'description': {'type': 'literal', 'value': 'island in the Pacific Ocean'}
                    }]
                }
            }
        return {
            'results': {
                'bindings': [
                    {
                        'subst': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q34497'},
                        'label': {'type': 'literal', 'value': 'Saint Helena'},
                        'lang': {'type': 'literal', 'value': 'en'},
                        'description': {'type': 'literal', 'value': 'island in the South Atlantic'}
                    },
                    {
                        'subst': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q34497'},
                        'label': {'type': 'literal', 'value': 'St. Helena'},
                        'lang': {'type': 'literal', 'value': 'de'}
                    },
                    {
                        'subst': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q99'},
                        'label': {'type': 'literal', 'value': 'California'},
                        'lang': {'type': 'literal', 'value': 'en'},
                        'description': {'type': 'literal', 'value': 'state of the United States'}
                    },
                    {
                        # the original entity itself must be ignored
                        'subst': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q14452'},
                        'label': {'type': 'literal', 'value': 'Easter Island'},
                        'lang': {'type': 'literal', 'value': 'en'}
                    },
                ]
            }
        }

    EMBEDDER = FakeEmbedder({
        'Saint Helena, island in the South Atlantic': 0.8,
        'California, state of the United States': 0.1,
    })

    def test_successful_embedding_substitutes(self):
        """Test ranking, filtering and the result structure."""
        query = 'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'

        result = find_substitutes_embedding(query, self.fake_execute, self.INFO, self.EMBEDDER, min_similarity=0.3)

        assert len(result) == 1
        assert result[0]['entity'] == 'wd:Q14452'
        assert result[0]['original text'] == 'Easter Island, island in the Pacific Ocean'

        rows = result[0]['results']
        # California is below the similarity threshold, the entity itself is excluded
        assert {r['subst'] for r in rows} == {'wd:Q34497'}
        # one row per label language, as in the SPARQL-based search
        assert {r['lang'] for r in rows} == {'en', 'de'}
        assert all(r['old'] == 'wd:Q14452' for r in rows)
        assert all(round(r['similarity'], 4) == 0.8 for r in rows)

    def test_min_similarity_keeps_all_when_zero(self):
        """Test that a zero threshold keeps all pool candidates."""
        query = 'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'

        result = find_substitutes_embedding(query, self.fake_execute, self.INFO, self.EMBEDDER, min_similarity=0.0)

        assert {r['subst'] for r in result[0]['results']} == {'wd:Q34497', 'wd:Q99'}
        # most similar substitute comes first
        assert result[0]['results'][0]['subst'] == 'wd:Q34497'

    def test_fallback_to_unordered_pool_query(self):
        """Test that an empty popularity-ordered retrieval falls back to the unordered query."""
        query = 'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'

        def execute(q):
            if 'sitelinks' in q:
                return None  # e.g., timeout on a very large class
            return self.fake_execute(q)

        result = find_substitutes_embedding(query, execute, self.INFO, self.EMBEDDER, min_similarity=0.3)

        assert {r['subst'] for r in result[0]['results']} == {'wd:Q34497'}
        assert 'sitelinks' not in result[0]['extract']

    def test_fallback_without_text_representation(self):
        """Test the fallback to the SPARQL-based search when the entity has no label."""
        query = 'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'

        def execute(q):
            if 'COALESCE' in q:
                return {'results': {'bindings': []}}
            return self.fake_execute(q)

        result = find_substitutes_embedding(query, execute, self.INFO, self.EMBEDDER)

        assert len(result) == 1
        assert 'original text' not in result[0]
        assert result[0]['results']
        assert all('similarity' not in r for r in result[0]['results'])

    def test_exception_handling(self):
        """Test that an exception skips the entity."""
        query = 'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'
        mock_execute = Mock(side_effect=Exception('SPARQL error'))

        result = find_substitutes_embedding(query, mock_execute, self.INFO, self.EMBEDDER)

        assert result == []

    def test_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is re-raised."""
        query = 'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'
        mock_execute = Mock(side_effect=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            find_substitutes_embedding(query, mock_execute, self.INFO, self.EMBEDDER)

    def test_no_wikidata_entities(self):
        """Test with no Wikidata entities."""
        info = {'resources': ['dbpedia:Res1'], 'conditions': {}, 'query conditions': {}}

        result = find_substitutes_embedding('SELECT ?x WHERE { ?x ?p ?o }', Mock(), info, self.EMBEDDER)

        assert result == []
