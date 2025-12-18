from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional
import os
from groq import Groq

from src.core.config import Settings
from dotenv import load_dotenv

load_dotenv()

LLM_MODELS: Dict[str, Dict[str, Any]] = {
    "daily": {
        "model": "openai/gpt-oss-20b",
        "name": "Daily Use LLM",
        "system_prompt": "You are a helpful assistant for everyday tasks. Provide clear, concise answers.",
        "supports_vision": False,
    },
    "image": {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "name": "Image Reasoning LLM",
        "system_prompt": "You are an expert in image analysis and visual reasoning. Provide detailed insights about images and visual content.",
        "supports_vision": True,
    },
    "complex": {
        "model": "qwen/qwen3-32b",
        "name": "Complex Tasks LLM",
        "system_prompt": "You are an advanced AI assistant capable of handling complex reasoning, analysis, and problem-solving tasks. Provide thorough, well-reasoned responses.",
        "supports_vision": False,
    },
}


class LLMService:
    """Wrapper around Groq chat completions for multiple models."""

    def __init__(self, settings: Settings, *, logger: Optional[logging.Logger] = None) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY must be configured for LLMService")

        self.logger = logger or logging.getLogger(__name__)
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)

    def _build_messages(
        self,
        model_key: str,
        prompt: Optional[str],
        image_bytes: Optional[bytes],
        image_mime_type: Optional[str],
        history: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        if model_key not in LLM_MODELS:
            raise ValueError(f"Unknown model key: {model_key}")

        model_info = LLM_MODELS[model_key]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": model_info["system_prompt"]}
        ]

        if history:
            messages.extend(history)

        supports_vision = model_info["supports_vision"]

        if supports_vision:
            content_blocks: list[dict[str, Any]] = []
            if prompt:
                content_blocks.append({"type": "text", "text": prompt})
            if image_bytes:
                encoded = base64.b64encode(image_bytes).decode("utf-8")
                mime = image_mime_type or "application/octet-stream"
                data_url = f"data:{mime};base64,{encoded}"
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                )

            if not content_blocks:
                raise ValueError("Vision model requests require text or image content")

            messages.append({"role": "user", "content": content_blocks})
        else:
            if image_bytes:
                raise ValueError("Selected model does not support vision inputs")
            if not prompt:
                raise ValueError("Text prompt is required for the selected model")

            messages.append({"role": "user", "content": prompt})

        return messages

    def chat(
        self,
        model_key: str,
        *,
        prompt: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        model_info = LLM_MODELS.get(model_key)
        if model_info is None:
            raise ValueError(f"Unknown model key: {model_key}")

        messages = self._build_messages(model_key, prompt, image_bytes, image_mime_type, history)

        self.logger.info("Sending chat request to model %s", model_info["model"])
        response = self.client.chat.completions.create(
            model=model_info["model"],
            messages=messages,
        )

        content = response.choices[0].message.content if response.choices else ""
        usage_data = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_data = {
                key: getattr(usage, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if hasattr(usage, key)
            }

        return {
            "model_key": model_key,
            "model_name": model_info["name"],
            "response": content,
            "usage": usage_data,
        }

    def stream_chat(
        self,
        model_key: str,
        *,
        prompt: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        history: Optional[list[dict[str, Any]]] = None,
    ):
        model_info = LLM_MODELS.get(model_key)
        if model_info is None:
            raise ValueError(f"Unknown model key: {model_key}")

        messages = self._build_messages(model_key, prompt, image_bytes, image_mime_type, history)

        self.logger.info("Sending streaming chat request to model %s", model_info["model"])
        stream = self.client.chat.completions.create(
            model=model_info["model"],
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
