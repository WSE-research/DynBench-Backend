"""Test configuration: make the repo root importable (so `utils`, `core`, … resolve)
and provide harmless defaults for the environment variables the service reads at
import time, so the unit tests run without a live deployment.
"""
import os

for _k, _v in {
    "MODEL": "gpt-4o",
    "KEY": "test-key",
    "LLM_URL": "http://localhost:9999/v1",
    "ALLOWED_MODELS": "gpt-4o",
    "MONGO_HOST": "localhost",
    "MONGO_USER": "test",
    "MONGO_PASS": "test",
    "WIKIDATA_AGENT": "wse-test-agent/1.0",
    "WIKIDATA_ENDPOINT": "https://query.wikidata.org/sparql",
}.items():
    os.environ.setdefault(_k, _v)
