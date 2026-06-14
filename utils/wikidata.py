import logging

logger = logging.getLogger(__name__)

from typing import Callable

from .sparql import uri2short, short2uri
from .sparql import sparql_results_to_list_of_dicts

from .text import extract_number

from .embeddings import entity_text, rank_candidates, rank_by_similarity


FIXED_LABELS = {
    'rdfs:label': 'label',
    'http://www.w3.org/2000/01/rdf-schema#label': 'label',
    'skos:altLabel': 'alternative label',
    'xsd:integer': 'integer',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': 'type',
    'http://www.w3.org/2001/XMLSchema#integer': 'integer',
    'http://www.w3.org/2001/XMLSchema#gYear': 'year',
    'http://www.w3.org/2001/XMLSchema#double': 'double',
}


# standard wikidata prefixes
WIKIDATA_PREFIX = {
    'bd': 'http://www.bigdata.com/rdf#',
    'cc': 'http://creativecommons.org/ns#',
    'dct': 'http://purl.org/dc/terms/',
    'geo': 'http://www.opengis.net/ont/geosparql#',
    'ontolex': 'http://www.w3.org/ns/lemon/ontolex#',
    'owl': 'http://www.w3.org/2002/07/owl#',
    'p': 'http://www.wikidata.org/prop/',
    'pq': 'http://www.wikidata.org/prop/qualifier/',
    'pqn': 'http://www.wikidata.org/prop/qualifier/value-normalized/',
    'pqv': 'http://www.wikidata.org/prop/qualifier/value/',
    'pr': 'http://www.wikidata.org/prop/reference/',
    'prn': 'http://www.wikidata.org/prop/reference/value-normalized/',
    'prov': 'http://www.w3.org/ns/prov#',
    'prv': 'http://www.wikidata.org/prop/reference/value/',
    'ps': 'http://www.wikidata.org/prop/statement/',
    'psn': 'http://www.wikidata.org/prop/statement/value-normalized/',
    'psv': 'http://www.wikidata.org/prop/statement/value/',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'schema': 'http://schema.org/',
    'skos': 'http://www.w3.org/2004/02/skos/core#',
    'wd': 'http://www.wikidata.org/entity/',
    'wdata': 'http://www.wikidata.org/wiki/Special:EntityData/',
    'wdno': 'http://www.wikidata.org/prop/novalue/',
    'wdref': 'http://www.wikidata.org/reference/',
    'wds': 'http://www.wikidata.org/entity/statement/',
    'wdt': 'http://www.wikidata.org/prop/direct/',
    'wdtn': 'http://www.wikidata.org/prop/direct-normalized/',
    'wdv': 'http://www.wikidata.org/value/',
    'wikibase': 'http://wikiba.se/ontology#',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
}

# reverse order for standard prefixes
for k, v in list(WIKIDATA_PREFIX.items()):
    WIKIDATA_PREFIX[v] = k


# languages in which substitute labels are collected
LABEL_LANGS = ('en', 'de', 'fr', 'ru', 'uk')


def query_wikidata_label(
        uri: str, 
        execute: Callable,
        lang: str='en', 
        fixed_labels: dict=None,
    ) -> str:
    """
    Query the Wikidata label for a given URI.
    
    Args:
        uri: Wikidata resource URI
        endpoint_url: SPARQL endpoint URL
        agent: User-Agent string for the request
        lang: Language code for the label (default is 'en')
    Returns:
        Label for the URI in the specified language, or None if not found
    """
    if fixed_labels and uri in fixed_labels:
        return fixed_labels[uri]

    query = (
        'SELECT ?label WHERE {',
            f"OPTIONAL {{ {uri} rdfs:label ?lang_label. FILTER(LANG(?lang_label) = '{lang}') }}",
            f"OPTIONAL {{ {uri} rdfs:label ?default_label. FILTER(LANG(?default_label) = 'mul') }}",
            f"OPTIONAL {{ {uri} owl:sameAs+ ?redirect . ?redirect rdfs:label ?redirect_label . FILTER(LANG(?redirect_label) = '{lang}') }}",
            "BIND(COALESCE(?lang_label, ?default_label, ?redirect_label) AS ?label) . ",
        '}',
    )

    try:
        data = execute('\n'.join(query))
        data = data['results']['bindings'][0]
        data = data['label']['value']
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f'Exception in function "query_wikidata_label". URI: {uri}, lang: {lang}, error: {e}')
        return None

    return data


def get_wikidata_label(
        uri: str, 
        execute: Callable, 
        lang: str='en', 
        prefixes: dict=WIKIDATA_PREFIX
    ) -> str:
    """ Get label for a given Wikidata URI.

    Args:
        uri: Wikidata resource URI
        endpoint_url: SPARQL endpoint URL
        agent: User-Agent string for the request
        lang: language code for the label (default is 'en')
    Returns:
        label for the URI in the specified language, or None if not found
    """
    try:
        uri = uri2short(uri, prefixes)
        prefix = 'wd' # it works despite property or entity
        index = uri.split(':')[-1]
        return query_wikidata_label(f'{prefix}:{index}', execute, lang, prefixes)
    except Exception as e:
        logger.error(f'Exception in function "get_wikidata_label". URI: {uri}, lang: {lang}, error: {e}')
        return None


def get_resources_types(info: dict, execute: Callable, predicates: list=None) -> dict:
    """
    Collect properties of entities for the item by given predicates.

    Args:
        info (dict): Information about a benchmark's record.
        predicates (list): List of predicates to filter types.
    Returns:
        Dictionary with properties for each entity (predicates are ignored).
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user.
    """
    results = {}

    if not predicates:
        return results

    for entity in info['resources']:
        if not entity.startswith('wd:'):
            continue

        query = 'SELECT DISTINCT ?p ?o WHERE { VALUES ?p { ' + ' '.join(predicates) + ' } ' + entity + ' ?p ?o }'

        try:
            r = execute(query)
            r = sparql_results_to_list_of_dicts(r, WIKIDATA_PREFIX)
            r = {p: [v['o'] for v in r if v['p'] == p] for p in predicates}
            for i in predicates:
                r[i].sort(key=extract_number)

            logger.debug(f'Collected properties for entity {entity}.')
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Exception {e} in function get_resources_types for entity {entity}.')
            r = {}

        results[entity] = r
        
    return results


def build_substitutes_query(
        entity: str,
        conditions: list[str],
        label_langs: tuple = LABEL_LANGS,
        limit: int = 1000,
        description_lang: str = None,
        order_by_sitelinks: bool = False,
    ) -> str:
    """
    Build a SPARQL query that selects substitute candidates for an entity.

    Args:
        entity: Entity to be replaced (short form, e.g. 'wd:Q123').
        conditions: Condition substrings constraining the ?subst variable.
        label_langs: Languages in which candidate labels are collected.
        limit: Maximum number of result rows (of candidate entities if
            order_by_sitelinks is set).
        description_lang: If set, also select an optional ?description in this language.
        order_by_sitelinks: If set, the most popular candidates (by number of
            sitelinks) are selected in a subquery instead of an arbitrary subset.
    Returns:
        SPARQL query string selecting ?subst, ?label, ?lang (and optionally ?description).
    """
    lang_filter = ', '.join(f'"{i}"' for i in label_langs)
    description = ' ?description' if description_lang else ''

    if order_by_sitelinks:
        extract_query = [
            f'SELECT DISTINCT ?subst ?label (lang(?label) as ?lang){description} WHERE {{',
            '    {',
            '        SELECT DISTINCT ?subst ?sitelinks WHERE {',
            *(f'            {i} .' for i in conditions),
            '            ?subst wikibase:sitelinks ?sitelinks .',
            f'            FILTER(?subst != {entity})',
            f'        }} ORDER BY DESC(?sitelinks) LIMIT {limit}',
            '    }',
            '    ?subst rdfs:label ?label .',
            f'    FILTER (lang(?label) IN ({lang_filter}))',
        ]
        if description_lang:
            extract_query.append(
                f'    OPTIONAL {{ ?subst schema:description ?description . FILTER(lang(?description) = "{description_lang}") }}'
            )
        extract_query.append('}')
    else:
        extract_query = [
            f'SELECT DISTINCT ?subst ?label (lang(?label) as ?lang){description} WHERE {{',
            *(f'{i} .' for i in conditions),
            '    ?subst rdfs:label ?label .',
            f'    FILTER (lang(?label) IN ({lang_filter}))',
            f'    FILTER(?subst != {entity})',
        ]
        if description_lang:
            extract_query.append(
                f'    OPTIONAL {{ ?subst schema:description ?description . FILTER(lang(?description) = "{description_lang}") }}'
            )
        extract_query.append(f'}} LIMIT {limit}')

    return '\n'.join(extract_query)


def get_legacy_conditions(entity: str, info: dict) -> list[str]:
    """
    Collect the condition substrings used by the SPARQL-based substitute search:
    all query conditions plus the first two type (P31/P279) conditions.

    Args:
        entity: Entity to be replaced (short form, e.g. 'wd:Q123').
        info (dict): Information about a benchmark's record.
    Returns:
        List of condition substrings constraining the ?subst variable.
    """
    entity_conditions = info['conditions'][entity]['wdt:P31'] + info['conditions'][entity]['wdt:P279']
    return info['query conditions'][entity] + entity_conditions[:2]


def find_substitutes(query: str, execute: Callable, info: dict) -> list:
    """
    Find possible substitutes for the each entity in the item.

    Substitutes are determined purely structurally with a SPARQL query: they have
    to share the first two type (P31/P279) conditions of the original entity and
    satisfy the query conditions. See find_substitutes_embedding for the
    embedding-based alternative.

    Args:
        query (str): SPARQL query.
        info (dict): Information about a benchmark's record.
    Returns:
        List of dictionaries with substitute information for each entity.
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user.
    """
    entities = [i for i in info['resources'] if i.startswith('wd:')]
    substitutes = []

    for entity in entities:
        extract_query = build_substitutes_query(entity, get_legacy_conditions(entity, info))

        try:
            result = execute(extract_query)
            result = sparql_results_to_list_of_dicts(result, WIKIDATA_PREFIX)

            logger.info(f'Found {len(result)} potential substitutes for entity {entity}.')

            for r in result:
                r['old'] = entity
            substitutes.append({ 'query': query, 'entity': entity, 'extract': extract_query, 'results': result })
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Exception in function find_substitutes: {e}')
            continue

    return substitutes


def get_entity_profile(entity: str, execute: Callable, lang: str = 'en') -> dict | None:
    """
    Get label and description of a Wikidata entity for its text representation.

    Args:
        entity: Entity (short form, e.g. 'wd:Q123').
        execute: Function to execute SPARQL queries.
        lang: Language code for label and description (default is 'en').
    Returns:
        Dictionary with 'label' and 'description' keys (values may be None),
        or None if the lookup failed.
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user.
    """
    query = (
        'SELECT ?label ?description WHERE {',
        f"    OPTIONAL {{ {entity} rdfs:label ?lang_label . FILTER(LANG(?lang_label) = '{lang}') }}",
        f"    OPTIONAL {{ {entity} rdfs:label ?default_label . FILTER(LANG(?default_label) = 'mul') }}",
        f"    OPTIONAL {{ {entity} schema:description ?description . FILTER(LANG(?description) = '{lang}') }}",
        '    BIND(COALESCE(?lang_label, ?default_label) AS ?label)',
        '} LIMIT 1',
    )

    try:
        result = execute('\n'.join(query))
        result = sparql_results_to_list_of_dicts(result, WIKIDATA_PREFIX)
        if not result:
            return None
        return { 'label': result[0].get('label'), 'description': result[0].get('description') }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f'Exception in function "get_entity_profile". Entity: {entity}, lang: {lang}, error: {e}')
        return None


def build_pool_condition_sets(entity: str, info: dict, max_type_conditions: int = 3) -> list[list[str]]:
    """
    Build the condition sets that retrieve the candidate pool for the embedding-based
    substitute search.

    The pool is recall-oriented: next to the strict legacy combination (all query
    conditions plus the first two type conditions) it relaxes the type constraint
    by querying each of the first max_type_conditions type (P31/P279) conditions
    separately. The semantic filtering is left to the embedding ranking.

    Args:
        entity: Entity to be replaced (short form, e.g. 'wd:Q123').
        info (dict): Information about a benchmark's record.
        max_type_conditions: Number of type conditions queried separately.
    Returns:
        List of condition lists (duplicates and empty sets are skipped).
    """
    query_conditions = info['query conditions'][entity]
    entity_conditions = info['conditions'][entity]['wdt:P31'] + info['conditions'][entity]['wdt:P279']

    condition_sets = [query_conditions + entity_conditions[:2]]
    for condition in entity_conditions[:max_type_conditions]:
        condition_sets.append(query_conditions + [condition])

    results = []
    seen = set()
    for conditions in condition_sets:
        key = tuple(conditions)
        if not conditions or key in seen:
            continue
        seen.add(key)
        results.append(conditions)

    return results


def collect_candidate_pool(
        entity: str,
        execute: Callable,
        info: dict,
        label_langs: tuple = LABEL_LANGS,
        description_lang: str = 'en',
        pool_limit: int = 500,
        max_type_conditions: int = 3,
    ) -> tuple[dict, list[str]]:
    """
    Collect the candidate pool for the embedding-based substitute searches.

    Executes the pool queries built from build_pool_condition_sets, preferring
    the most popular candidates (by sitelinks). This ordered retrieval can time
    out on very large classes, in which case an arbitrary subset is taken as in
    the SPARQL-based search.

    Args:
        entity: Entity to be replaced (short form, e.g. 'wd:Q123').
        execute: Function to execute SPARQL queries.
        info (dict): Information about a benchmark's record.
        label_langs: Languages in which candidate labels are collected.
        description_lang: Language of the candidate descriptions.
        pool_limit: Maximum number of result rows per pool query.
        max_type_conditions: Number of type conditions queried separately.
    Returns:
        Tuple of (pool, pool_queries): pool maps each candidate to a dictionary
        with 'subst', 'labels' (one per language) and 'description'; pool_queries
        are the executed SPARQL queries.
    """
    condition_sets = build_pool_condition_sets(entity, info, max_type_conditions)

    pool = {}
    pool_queries = []
    for conditions in condition_sets:
        extract_query = build_substitutes_query(
            entity, conditions, label_langs,
            limit=pool_limit, description_lang=description_lang, order_by_sitelinks=True,
        )
        result = sparql_results_to_list_of_dicts(execute(extract_query), WIKIDATA_PREFIX)

        if not result:
            extract_query = build_substitutes_query(
                entity, conditions, label_langs,
                limit=pool_limit, description_lang=description_lang,
            )
            result = sparql_results_to_list_of_dicts(execute(extract_query), WIKIDATA_PREFIX)

        pool_queries.append(extract_query)

        for row in result:
            subst = row.get('subst')
            if not subst or subst == entity:
                continue
            candidate = pool.setdefault(subst, { 'subst': subst, 'labels': {}, 'description': None })
            if row.get('label') and row.get('lang'):
                candidate['labels'].setdefault(row['lang'], row['label'])
            if row.get('description') and not candidate['description']:
                candidate['description'] = row['description']

    logger.info(f'Collected a pool of {len(pool)} candidates for entity {entity} from {len(pool_queries)} queries.')

    return pool, pool_queries


def build_substitute_rows(ranked: list[dict], entity: str) -> list[dict]:
    """
    Convert ranked pool candidates into result rows with the same structure as
    find_substitutes: one row per substitute and label language.

    Args:
        ranked: Ranked candidates (see rank_by_similarity).
        entity: The original entity the substitutes belong to.
    Returns:
        List of result rows with 'subst', 'label', 'lang', 'similarity' and 'old' keys.
    """
    results = []
    for candidate in ranked:
        for lang, label in candidate['labels'].items():
            results.append({
                'subst': candidate['subst'],
                'label': label,
                'lang': lang,
                'similarity': candidate['similarity'],
                'old': entity,
            })
    return results


def find_substitutes_embedding(
        query: str,
        execute: Callable,
        info: dict,
        embedder,
        embedding_lang: str = 'en',
        label_langs: tuple = LABEL_LANGS,
        pool_limit: int = 500,
        top_k: int = 50,
        min_similarity: float = 0.3,
        max_type_conditions: int = 3,
    ) -> list:
    """
    Find possible substitutes for each entity in the item using embedding similarity.

    Works in two stages (retrieve and rerank): first a recall-oriented candidate
    pool is retrieved with relaxed SPARQL queries (see build_pool_queries), then
    the candidates are ranked by the cosine similarity between the embeddings of
    their text representation (label plus description) and the one of the original
    entity. Only the top_k candidates with a similarity of at least min_similarity
    are kept, so substitutes are semantically similar resources instead of an
    arbitrary subset of structurally compatible ones.

    Falls back to the SPARQL-based search for entities without a text representation.

    Args:
        query (str): SPARQL query.
        execute: Function to execute SPARQL queries.
        info (dict): Information about a benchmark's record.
        embedder: Object with an encode(list[str]) -> np.ndarray method returning
            L2-normalized embeddings (one row per input text).
        embedding_lang: Language of the texts that are embedded.
        label_langs: Languages in which substitute labels are collected.
        pool_limit: Maximum number of result rows per pool query.
        top_k: Maximum number of substitutes kept per entity.
        min_similarity: Minimum cosine similarity for a substitute to be kept.
        max_type_conditions: Number of type conditions queried separately for the pool.
    Returns:
        List of dictionaries with substitute information for each entity. Same
        structure as find_substitutes, with an additional 'similarity' key on
        each result row.
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user.
    """
    entities = [i for i in info['resources'] if i.startswith('wd:')]
    substitutes = []

    for entity in entities:
        try:
            profile = get_entity_profile(entity, execute, lang=embedding_lang)
            original_text = entity_text(profile.get('label'), profile.get('description')) if profile else None

            if not original_text:
                logger.warning(f'No text representation found for entity {entity}, falling back to SPARQL-based search.')
                extract_query = build_substitutes_query(entity, get_legacy_conditions(entity, info))
                result = sparql_results_to_list_of_dicts(execute(extract_query), WIKIDATA_PREFIX)
                for r in result:
                    r['old'] = entity
                substitutes.append({ 'query': query, 'entity': entity, 'extract': extract_query, 'results': result })
                continue

            pool, pool_queries = collect_candidate_pool(
                entity, execute, info, label_langs,
                description_lang=embedding_lang,
                pool_limit=pool_limit,
                max_type_conditions=max_type_conditions,
            )

            candidates = []
            for candidate in pool.values():
                label = candidate['labels'].get(embedding_lang) or next(iter(candidate['labels'].values()), None)
                text = entity_text(label, candidate['description'])
                if text:
                    candidates.append(candidate | { 'text': text })

            ranked = rank_candidates(original_text, candidates, embedder, top_k=top_k, min_similarity=min_similarity)

            logger.info(f'Kept {len(ranked)} of {len(candidates)} candidates for entity {entity} after embedding ranking.')

            substitutes.append({
                'query': query,
                'entity': entity,
                'extract': '\n\n'.join(pool_queries),
                'original text': original_text,
                # same result structure as find_substitutes: one row per substitute and label language
                'results': build_substitute_rows(ranked, entity),
            })
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Exception in function find_substitutes_embedding: {e}')
            continue

    return substitutes


def find_substitutes_rdf2vec(
        query: str,
        execute: Callable,
        info: dict,
        embedder,
        label_langs: tuple = LABEL_LANGS,
        pool_limit: int = 50,
        top_k: int = 50,
        min_similarity: float = 0.0,
        max_type_conditions: int = 3,
    ) -> list:
    """
    Find possible substitutes for each entity in the item using RDF2Vec graph embeddings.

    Like find_substitutes_embedding this works in two stages (retrieve and rerank),
    but the ranking signal is the knowledge graph structure instead of textual
    descriptions: pyRDF2Vec extracts random walks for the original entity and all
    pool candidates from the SPARQL endpoint and trains a Word2Vec model on them,
    so entities with similar graph neighborhoods obtain similar vectors. The
    Word2Vec model is trained per entity, hence similarity values are only
    comparable within one substitute search (their scale also differs from the
    sentence embedding similarities).

    Args:
        query (str): SPARQL query.
        execute: Function to execute SPARQL queries (used for the candidate pool;
            walk extraction connects to the endpoint directly).
        info (dict): Information about a benchmark's record.
        embedder: Object with an encode_entities(list[str]) -> np.ndarray method
            returning L2-normalized embeddings for full entity URIs (see
            utils.rdf2vec.RDF2VecEmbedder).
        label_langs: Languages in which substitute labels are collected.
        pool_limit: Maximum number of result rows per pool query. Should be kept
            small: every pool candidate costs additional SPARQL requests during
            walk extraction.
        top_k: Maximum number of substitutes kept per entity.
        min_similarity: Minimum cosine similarity for a substitute to be kept.
        max_type_conditions: Number of type conditions queried separately for the pool.
    Returns:
        List of dictionaries with substitute information for each entity. Same
        structure as find_substitutes, with an additional 'similarity' key on
        each result row.
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user.
    """
    entities = [i for i in info['resources'] if i.startswith('wd:')]
    substitutes = []

    for entity in entities:
        try:
            pool, pool_queries = collect_candidate_pool(
                entity, execute, info, label_langs,
                pool_limit=pool_limit,
                max_type_conditions=max_type_conditions,
            )

            candidates = list(pool.values())
            if not candidates:
                substitutes.append({ 'query': query, 'entity': entity, 'extract': '\n\n'.join(pool_queries), 'results': [] })
                continue

            uris = [short2uri(i, WIKIDATA_PREFIX) for i in [entity] + [c['subst'] for c in candidates]]
            vectors = embedder.encode_entities(uris)

            # embeddings are L2-normalized, so the dot product equals cosine similarity
            similarities = vectors[1:] @ vectors[0]

            ranked = rank_by_similarity(candidates, similarities, top_k=top_k, min_similarity=min_similarity)

            logger.info(f'Kept {len(ranked)} of {len(candidates)} candidates for entity {entity} after RDF2Vec ranking.')

            substitutes.append({
                'query': query,
                'entity': entity,
                'extract': '\n\n'.join(pool_queries),
                # same result structure as find_substitutes: one row per substitute and label language
                'results': build_substitute_rows(ranked, entity),
            })
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Exception in function find_substitutes_rdf2vec: {e}')
            continue

    return substitutes


def check_productivity_single(query: str, execute: Callable, replace: dict, prefixes: dict = WIKIDATA_PREFIX) -> bool:
    """
    Check if query is productive (i.e., returns non-empty results).
    
    Args:
        query: SPARQL query
        execute: Function to execute SPARQL queries
        replace: Dictionary containing 'old' and 'new' keys for string replacement in the query
        prefixes: Dictionary of prefixes for SPARQL queries (default is WIKIDATA_PREFIX)
    Returns:
        True if the query returns non-empty results, False otherwise
    """
    sparql = query.replace(replace.get('old', ''), replace.get('new', '')).strip()
    try:
        result = sparql_results_to_list_of_dicts(execute(sparql), prefixes)
        return bool(result)
    except Exception as e:
        logger.error(f'Exception in function "check_productivity_single": {e}')
        return False