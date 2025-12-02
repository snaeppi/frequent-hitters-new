"""Factory for creating chat models across providers."""

from __future__ import annotations

from typing import Any, Mapping

from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_XAI_MODEL = "grok-4-1-fast-reasoning"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_VERTEX_MODEL = "gemini-1.5-pro"


def create_llm(
    *,
    model_provider: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> tuple[BaseChatModel, str]:
    provider = (model_provider or "xai").lower()
    extra = dict(model_kwargs or {})

    if provider == "xai":
        from langchain_xai import ChatXAI

        resolved_name = model_name or DEFAULT_XAI_MODEL
        params: dict[str, Any] = {"model": resolved_name}
        params.update(extra)
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        return ChatXAI(**params), resolved_name

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        resolved_name = model_name or DEFAULT_OPENAI_MODEL
        params = {"model": resolved_name}
        params.update(extra)
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        return ChatOpenAI(**params), resolved_name

    if provider == "vertexai":
        from langchain_google_vertexai import ChatVertexAI

        resolved_name = model_name or DEFAULT_VERTEX_MODEL
        params = {"model": resolved_name}
        params.update(extra)
        return ChatVertexAI(**params), resolved_name

    raise ValueError(
        f"Unsupported model_provider '{model_provider}'. Choose from xai, openai, vertexai."
    )


__all__ = ["create_llm"]

