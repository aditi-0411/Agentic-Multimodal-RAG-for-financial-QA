"""
config.py

Central place for model/config setup. Reads secrets from environment
variables ONLY — never hardcode API keys in source or notebooks.

Usage:
    export GROQ_API_KEY="your-key-here"   # or put it in a .env file (see .env.example)
"""

import os

from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder, SentenceTransformer

LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_llm(temperature: float = 0):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Export it as an environment variable or "
            "put it in a .env file (see .env.example) — never hardcode it in source."
        )
    return ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=temperature)


def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


def embed_texts(embedding_model, texts):
    return embedding_model.encode(texts, normalize_embeddings=True)
