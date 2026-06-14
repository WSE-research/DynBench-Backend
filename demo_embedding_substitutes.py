#!/usr/bin/env python3
"""
Live demonstration of the embedding-based substitute search.

Runs the candidate determination (without LLM rewriting) for a few benchmark
records against the public Wikidata SPARQL endpoint. Requires neither MongoDB
nor an LLM service, only network access.
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
for noisy in ('httpx', 'urllib3', 'filelock', 'huggingface_hub'):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from functools import partial

from utils.sparql import execute as raw_execute
from utils.sparql import normal_sparql, parse_query, extract_entities
from utils.sparql import get_conditions_by_predicates, get_query_conditions
from utils.wikidata import get_resources_types, find_substitutes_embedding, WIKIDATA_PREFIX
from utils.embeddings import SentenceTransformerEmbedder

ENDPOINT = 'https://query.wikidata.org/sparql'
AGENT = 'DynBench/0.1 (https://github.com/WSE-research/DynBench-Backend; embedding substitute demo)'
PREDICATES = ('wdt:P31', 'wdt:P279', )

EXAMPLES = (
    ('Which country does the famous Easter island belong to?',
     'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'),
    ('When did Finland join the EU?',
     'SELECT ?date WHERE { wd:Q33 p:P463 ?membership . ?membership pq:P580 ?date . ?membership ps:P463 wd:Q458 }'),
    ('List all boardgames by GMT.',
     'SELECT ?uri WHERE { ?uri wdt:P31 wd:Q131436 . }'),
)


def main():
    execute = partial(raw_execute, endpoint_url=ENDPOINT, agent=AGENT, delay=1.0, timeout=60.0)
    embedder = SentenceTransformerEmbedder('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    for question, query in EXAMPLES:
        print('=' * 100)
        print(f'Question: {question}')
        print(f'Query:    {query}')

        query = normal_sparql(query, WIKIDATA_PREFIX)

        info = {}
        info['triples'] = [i for i in parse_query(query) if all(i)]
        info['resources'] = extract_entities(query)
        info['types'] = get_resources_types(info, execute, PREDICATES)
        info['conditions'] = get_conditions_by_predicates(info, PREDICATES)
        info['query conditions'] = get_query_conditions(info)

        substitutes = find_substitutes_embedding(query, execute, info, embedder)

        for sub in substitutes:
            print(f"\nEntity: {sub['entity']} (text: \"{sub.get('original text')}\")")

            # prefer the English label of each substitute for display
            labels = {}
            for row in sub['results']:
                current = labels.setdefault(row['subst'], row)
                if row['lang'] == 'en':
                    labels[row['subst']] = row

            for row in list(labels.values())[:10]:
                similarity = f"{row['similarity']:.4f}" if 'similarity' in row else '   -  '
                print(f"  {similarity}  {row['subst']:<12}  {row['label']}")
        print()


if __name__ == '__main__':
    main()
