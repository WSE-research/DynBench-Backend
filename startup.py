import logging

logger = logging.getLogger(__name__)

import requests

from decouple import config

from pymongo import MongoClient

from utils.timer import wait_time

from utils.mongocache import MongoCache

from utils.sparql import execute as raw_execute

from utils.wikidata import get_wikidata_label

from utils.embeddings import embeddings_available, SentenceTransformerEmbedder

from utils.rdf2vec import rdf2vec_available, RDF2VecEmbedder

# from utils.llm import call_LLM as raw_call_LLM

from openai import OpenAI


MONGO_HOST = config('MONGO_HOST')
MONGO_USER = config('MONGO_USER')
MONGO_PASS = config('MONGO_PASS')

# LLM_URL = config('LLM_URL')
# BASE_URL = LLM_URL.replace('/api/generate', '').replace('/v1/chat/completions', '')
BASE_URL = config('LLM_URL')
LLM_URL = BASE_URL + '/v1/chat/completions'
# LLM_MODELS = f'{BASE_URL}/v1/models'

WIKIDATA_AGENT = config('WIKIDATA_AGENT')
WIKIDATA_ENDPOINT = config('WIKIDATA_ENDPOINT')

KEY = config('KEY')
ALLOWED_MODELS = config('ALLOWED_MODELS').split(',')
# MODEL=config('MODEL')

logger.info(f'Mongo host: {MONGO_HOST}')
# logger.info(f'Mongo user: {MONGO_USER}')
logger.info(f'Wikidata endpoint: {WIKIDATA_ENDPOINT}')
# logger.info(f'Wikidata agent: {WIKIDATA_AGENT}')
# logger.info(f'LLM URL: {BASE_URL}')

mongo = MongoClient(
    MONGO_HOST,
    username=MONGO_USER,
    password=MONGO_PASS,
)

db = mongo.dynbench
cache_collection = db.cache
feedback_collection = db.feedback


# make sure all documents have "order" field
doc = cache_collection.find_one({ 'order': {"$exists": True} }, sort={ 'order': -1 })
if doc:
    order = doc['order']
else:
    order = 0

for doc in cache_collection.find({ 'order': {"$exists": False} }, sort=[('_id', 1)]):
    order += 1
    cache_collection.update_one({ '_id': doc['_id'] }, { '$set': { 'order': order } })

cache = MongoCache(cache_collection, 1024*1024)


@cache.cache(using={'query'})
def execute(query: str, delay=2.0, timeout=30.0,) -> dict | None:
    return raw_execute(query, WIKIDATA_ENDPOINT, WIKIDATA_AGENT, delay=delay, timeout=timeout)


# @cache.cache(using={'url', 'model', 'prompt', 'temp', 'max_tokens'})
# def call_LLM(url: str, key: str, model: str, prompt, temp: float=0.0, max_tokens: int=1000, timeout=30.0) -> dict | None:
#     return raw_call_LLM(url, key, model, prompt, temp, max_tokens, timeout)
        
try: 
    client = OpenAI(base_url=BASE_URL, api_key=KEY)
    models_list = list(client.models.list())
    models_list = [dict(i) for i in models_list]
    models_list = sorted([i['id'] for i in models_list if i['id'] in ALLOWED_MODELS])
    # print('\n'.join(models_list))
except Exception as e:
    logger.error('Error connecting to LLM, exiting...')
    logger.error(str(e))
    exit(1)


@cache.cache(using={'url', 'model', 'prompt'})
def call_LLM(url, model, prompt):
    response = client.responses.create(
        model=model,
        input=prompt
    )
    return response.output_text


def get_label(entity: str, lang: str='en') -> str:
    return get_wikidata_label(entity, execute, lang=lang)


logger.info(f'Cache contains {cache_collection.count_documents({})} records.')


# try:
#     r = requests.get(BASE_URL)
#     logger.info(f'LLM status (http code): {r.status_code}')
# except:
#     logger.error('Error connecting LLM, exiting...')
#     exit(1)
    
    
# Load PageRank file into memory
page_rank = {}

wait_time(0.0, 'pagerank file load') # init timer to skip "Loaded 1 record" message
logger.info('Loading PageRank file...')
try:
    with open('pagerank/allwiki.rank', 'r') as f:
        for x, line in enumerate(f):
            entity, rank = line.split('\t')
            page_rank[entity.strip()] = float(rank.strip())
            if wait_time(1.0, 'pagerank file load'):
                logger.info(f'Loaded {x+1:,} records.'.replace(',', ' '))
    logger.info('PageRank file loaded successfully')
except Exception as e:
    logger.error(f'Error loading PageRank file in settings.py: {e}. Exiting...')
    exit(1)


# Top 20 languages by number of speakers in Europe
LANGUAGES = {
    'English':    'en',
    'German':     'de',
    'French':     'fr',
    'Russian':    'ru',
    'Ukrainian':  'uk',
    'Italian':    'it',
    'Spanish':    'es',
    'Polish':     'pl',
    'Romanian':   'ro',
    'Dutch':      'nl',
    'Turkish':    'tr',
    'Bavarian':   'bar',
    'Portuguese': 'pt',
    'Hungarian':  'hu',
    'Greek':      'el',
    'Czech':      'cs',
    'Swedish':    'sv',
    'Catalan':    'ca',
    'Serbian':    'sr',
    'Bulgarian':  'bg',
}

# Add reverse order
for k, v in list(LANGUAGES.items()):
    LANGUAGES[v] = k

PREDICATES = ('wdt:P31', 'wdt:P279', )


# Substitute search method: 'embedding' (sentence embedding similarity, default),
# 'rdf2vec' (graph embedding similarity) or 'sparql' (structural matching)
SUBSTITUTE_METHOD = config('SUBSTITUTE_METHOD', default='embedding').strip().lower()

EMBEDDING_MODEL = config('EMBEDDING_MODEL', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
EMBEDDING_LANG = config('EMBEDDING_LANG', default='en')
EMBEDDING_TOP_K = config('EMBEDDING_TOP_K', default=50, cast=int)
EMBEDDING_MIN_SIMILARITY = config('EMBEDDING_MIN_SIMILARITY', default=0.3, cast=float)
EMBEDDING_POOL_LIMIT = config('EMBEDDING_POOL_LIMIT', default=500, cast=int)
EMBEDDING_MAX_TYPE_CONDITIONS = config('EMBEDDING_MAX_TYPE_CONDITIONS', default=3, cast=int)

RDF2VEC_MAX_DEPTH = config('RDF2VEC_MAX_DEPTH', default=2, cast=int)
RDF2VEC_MAX_WALKS = config('RDF2VEC_MAX_WALKS', default=10, cast=int)
RDF2VEC_EPOCHS = config('RDF2VEC_EPOCHS', default=10, cast=int)
RDF2VEC_VECTOR_SIZE = config('RDF2VEC_VECTOR_SIZE', default=100, cast=int)
# small pool: every pool candidate costs additional SPARQL requests during walk extraction
RDF2VEC_POOL_LIMIT = config('RDF2VEC_POOL_LIMIT', default=50, cast=int)
# Word2Vec cosine similarities are scaled differently than sentence embedding ones
RDF2VEC_MIN_SIMILARITY = config('RDF2VEC_MIN_SIMILARITY', default=0.0, cast=float)

if SUBSTITUTE_METHOD not in ('embedding', 'rdf2vec', 'sparql'):
    logger.warning(f'Unknown SUBSTITUTE_METHOD "{SUBSTITUTE_METHOD}", falling back to "sparql".')
    SUBSTITUTE_METHOD = 'sparql'

embedder = None
if SUBSTITUTE_METHOD == 'embedding':
    if embeddings_available():
        # the model itself is loaded lazily on the first transformation
        embedder = SentenceTransformerEmbedder(EMBEDDING_MODEL)
        logger.info(f'Substitute search: embedding-based (model: {EMBEDDING_MODEL}).')
    else:
        logger.error('sentence-transformers is not installed, falling back to SPARQL-based substitute search.')
        SUBSTITUTE_METHOD = 'sparql'
elif SUBSTITUTE_METHOD == 'rdf2vec':
    if rdf2vec_available():
        # pyRDF2Vec itself is loaded lazily on the first transformation
        embedder = RDF2VecEmbedder(
            WIKIDATA_ENDPOINT,
            max_depth=RDF2VEC_MAX_DEPTH,
            max_walks=RDF2VEC_MAX_WALKS,
            epochs=RDF2VEC_EPOCHS,
            vector_size=RDF2VEC_VECTOR_SIZE,
            **({ 'agent': WIKIDATA_AGENT } if WIKIDATA_AGENT else {}),
        )
        logger.info(f'Substitute search: RDF2Vec-based (endpoint: {WIKIDATA_ENDPOINT}).')
    else:
        logger.error('pyrdf2vec/gensim is not installed, falling back to SPARQL-based substitute search.')
        SUBSTITUTE_METHOD = 'sparql'

if SUBSTITUTE_METHOD == 'sparql':
    logger.info('Substitute search: SPARQL-based.')

