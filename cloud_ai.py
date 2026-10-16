"""Shared AI adapter for Nexus AI.

With GEMINI_API_KEY configured, use the hosted Gemini API.
Without a key, retain the original local Ollama workflow.
Only use non-confidential, sample documents with the public hosted demo.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable


CHAT_API_MODEL = os.getenv("NEXUS_GEMINI_MODEL", "gemini-3.8-flash")
EMBED_API_MODEL = os.getenv("NEXUS_GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSIONS = 768


def _api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        return key or None
    except Exception:
        return None


def using_remote_ai() -> bool:
    """Return True when a Gemini API key is configured."""
    return bool(_api_key())


@lru_cache(maxsize=2)
def _gemini_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def embed_texts(
    texts: Iterable[str],
    *,
    local_model: str = "nomic-embed-text",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Embed text using Gemini in cloud mode or Ollama in local mode."""
    items = [str(text) for text in texts]
    if not items:
        return []

    key = _api_key()
    if not key:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "No GEMINI_API_KEY is configured and the Ollama Python "
                "package is not installed. Add GEMINI_API_KEY for cloud mode."
            ) from exc
        response = ollama.embed(model=local_model, input=items)
        return response["embeddings"]

    from google.genai import types

    client = _gemini_client(key)
    all_vectors: list[list[float]] = []
    # Small batches avoid oversized requests when a report has many chunks.
    for start in range(0, len(items), 32):
        batch = items[start:start + 32]
        response = client.models.embed_content(
            model=EMBED_API_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise RuntimeError(
                "The embedding service returned an unexpected number of vectors."
            )
        all_vectors.extend([list(item.values) for item in embeddings])
    return all_vectors


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    local_model: str = "qwen2.5:3b",
    json_mode: bool = False,
) -> str:
    """Generate a response with Gemini API or the local Ollama model."""
    key = _api_key()
    if not key:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "No GEMINI_API_KEY is configured and the Ollama Python "
                "package is not installed. Add GEMINI_API_KEY for cloud mode."
            ) from exc
        kwargs = {
            "model": local_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0},
        }
        if json_mode:
            kwargs["format"] = "json"
        response = ollama.chat(**kwargs)
        return response["message"]["content"].strip()

    from google.genai import types

    config_args = {
        "system_instruction": system_prompt,
        "temperature": 0,
    }
    if json_mode:
        config_args["response_mime_type"] = "application/json"

    response = _gemini_client(key).models.generate_content(
        model=CHAT_API_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(**config_args),
    )
    content = response.text
    if not content or not content.strip():
        raise RuntimeError("The hosted AI model returned an empty response.")
    return content.strip()
