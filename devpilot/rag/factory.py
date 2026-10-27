from pathlib import Path

import openai

from devpilot.config import Settings
from devpilot.rag.embeddings import APIEmbedder, Embedder, HashEmbedder
from devpilot.rag.store import DocStore


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder()
    client = openai.AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    return APIEmbedder(client, settings.embedding_model)


def build_store(settings: Settings) -> DocStore:
    return DocStore(Path(settings.chroma_dir), build_embedder(settings))
