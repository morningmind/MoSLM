"""Optional native adapters for hosted model-provider SDKs."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .backends import Backend
from .types import ChatResponse, Message, ToolCall, Usage


def _optional_sdk(package: str, install_extra: str):
    try:
        return __import__(package, fromlist=[""])
    except ImportError as exc:
        raise ImportError(
            f"This backend requires the optional '{install_extra}' extra. "
            f"Install it with: pip install 'moslm[{install_extra}]'"
        ) from exc


def _tool_arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value or "{}")
        except json.JSONDecodeError:
            return {"_raw": value}
    return value if isinstance(value, dict) else {}


class OpenAIBackend(Backend):
    """Calls OpenAI's SDK, loading ``OPENAI_API_KEY`` when no key is provided."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        client: Any = None,
    ):
        if client is None:
            sdk = _optional_sdk("openai", "openai")
            options: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
            if base_url is not None:
                options["base_url"] = base_url
            client = sdk.OpenAI(**options)
        self.client = client
        self.model = model

    def chat(self, messages, tools=None, max_tokens=512, temperature=0.0) -> ChatResponse:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai() for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        start = time.perf_counter()
        response = self.client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - start) * 1000
        raw = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=call.id or "",
                name=call.function.name,
                arguments=_tool_arguments(call.function.arguments),
            )
            for call in (raw.tool_calls or [])
        ]
        usage = response.usage
        return ChatResponse(
            message=Message(role="assistant", content=raw.content, tool_calls=tool_calls),
            usage=Usage(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=latency_ms,
        )


class AnthropicBackend(Backend):
    """Calls Anthropic's SDK, loading ``ANTHROPIC_API_KEY`` when no key is provided."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        client: Any = None,
    ):
        if client is None:
            sdk = _optional_sdk("anthropic", "anthropic")
            client = sdk.Anthropic(api_key=api_key, timeout=timeout)
        self.client = client
        self.model = model

    def chat(self, messages, tools=None, max_tokens=512, temperature=0.0) -> ChatResponse:
        system = "\n\n".join(
            message.content or "" for message in messages if message.role == "system"
        )
        conversation = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                conversation.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": message.content or "",
                    }],
                })
                continue

            content: List[Dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            if message.role == "assistant":
                content.extend({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                } for call in message.tool_calls)
            conversation.append({"role": message.role, "content": content})

        request: Dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"]["parameters"],
                }
                for tool in tools
            ]

        start = time.perf_counter()
        response = self.client.messages.create(**request)
        latency_ms = (time.perf_counter() - start) * 1000
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input or {},
                ))
        usage = response.usage
        return ChatResponse(
            message=Message(
                role="assistant",
                content="".join(text_parts) or None,
                tool_calls=tool_calls,
            ),
            usage=Usage(
                getattr(usage, "input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0,
            ),
            latency_ms=latency_ms,
        )


class GoogleGeminiBackend(Backend):
    """Calls Google's Gen AI SDK, loading ``GEMINI_API_KEY`` when no key is provided."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Any = None,
    ):
        if client is None:
            sdk = _optional_sdk("google.genai", "gemini")
            client = sdk.Client(api_key=api_key)
        self.client = client
        self.model = model

    def chat(self, messages, tools=None, max_tokens=512, temperature=0.0) -> ChatResponse:
        system = "\n\n".join(
            message.content or "" for message in messages if message.role == "system"
        )
        contents = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                parts = [{
                    "function_response": {
                        "name": message.name or "",
                        "response": {"result": message.content or ""},
                        "id": message.tool_call_id,
                    }
                }]
                role = "user"
            elif message.role == "assistant":
                parts = [{"text": message.content}] if message.content else []
                parts.extend({
                    "function_call": {
                        "name": call.name,
                        "args": call.arguments,
                        "id": call.id,
                    }
                } for call in message.tool_calls)
                role = "model"
            else:
                parts = [{"text": message.content or ""}]
                role = "user"
            contents.append({"role": role, "parts": parts})

        config: Dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            config["system_instruction"] = system
        if tools:
            config["tools"] = [{
                "function_declarations": [
                    {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"]["parameters"],
                    }
                    for tool in tools
                ]
            }]

        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text_parts = []
        tool_calls = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)
                function_call = getattr(part, "function_call", None)
                if function_call:
                    tool_calls.append(ToolCall(
                        id=getattr(function_call, "id", "") or "",
                        name=function_call.name,
                        arguments=dict(function_call.args or {}),
                    ))
        usage = getattr(response, "usage_metadata", None)
        return ChatResponse(
            message=Message(
                role="assistant",
                content="".join(text_parts) or None,
                tool_calls=tool_calls,
            ),
            usage=Usage(
                getattr(usage, "prompt_token_count", 0) or 0,
                getattr(usage, "candidates_token_count", 0) or 0,
            ),
            latency_ms=latency_ms,
        )


class LiteLLMBackend(Backend):
    """Calls LiteLLM's unified completion API for its supported providers."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 120.0,
        completion: Any = None,
    ):
        if completion is None:
            litellm = _optional_sdk("litellm", "litellm")
            completion = litellm.completion
        self.completion = completion
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout

    def chat(self, messages, tools=None, max_tokens=512, temperature=0.0) -> ChatResponse:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai() for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        if self.api_key is not None:
            request["api_key"] = self.api_key
        if self.api_base is not None:
            request["api_base"] = self.api_base

        start = time.perf_counter()
        response = self.completion(**request)
        latency_ms = (time.perf_counter() - start) * 1000
        raw = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=call.id or "",
                name=call.function.name,
                arguments=_tool_arguments(call.function.arguments),
            )
            for call in (raw.tool_calls or [])
        ]
        usage = response.usage
        return ChatResponse(
            message=Message(role="assistant", content=raw.content, tool_calls=tool_calls),
            usage=Usage(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=latency_ms,
        )
