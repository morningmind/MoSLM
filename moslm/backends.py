"""Model backends. Any server with an OpenAI-compatible /v1/chat/completions
endpoint works: vLLM, Ollama, llama.cpp server, LM Studio, SGLang, or a hosted API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from .types import ChatResponse, Message, ToolCall, Usage


class Backend:
    """Interface every backend implements."""

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ChatResponse:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAICompatibleBackend(Backend):
    """Calls an OpenAI-compatible chat completions endpoint using only the standard library.

    Example (vLLM):
        vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --port 8001
        OpenAICompatibleBackend("http://localhost:8001/v1", "Qwen/Qwen2.5-Math-1.5B-Instruct")

    For the orchestrator model, start vLLM with tool calling enabled:
        vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 \\
            --enable-auto-tool-choice --tool-call-parser hermes
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        request_logprobs: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.request_logprobs = request_logprobs

    def chat(self, messages, tools=None, max_tokens=512, temperature=0.0) -> ChatResponse:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.request_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 5

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.model}: HTTP {e.code}: {detail}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        choice = data["choices"][0]
        raw = choice["message"]
        tool_calls = []
        for tc in raw.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                parsed = {"_raw": args}
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=parsed))

        u = data.get("usage") or {}
        return ChatResponse(
            message=Message(role="assistant", content=raw.get("content"), tool_calls=tool_calls),
            usage=Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0)),
            latency_ms=latency_ms,
        )


class MockBackend(Backend):
    """A scripted backend for tests and offline demos.

    Pass either a list of ChatResponse objects (returned in order) or a function
    that receives (messages, tools) and returns a ChatResponse.
    """

    def __init__(self, script):
        self._fn: Optional[Callable] = script if callable(script) else None
        self._queue: List[ChatResponse] = [] if callable(script) else list(script)
        self.calls: List[List[Message]] = []

    def chat(self, messages, tools=None, max_tokens=512, temperature=0.0) -> ChatResponse:
        self.calls.append(list(messages))
        if self._fn is not None:
            return self._fn(messages, tools)
        if not self._queue:
            raise RuntimeError("MockBackend script exhausted")
        return self._queue.pop(0)

    @staticmethod
    def text(content: str, prompt_tokens: int = 10, completion_tokens: int = 10) -> ChatResponse:
        return ChatResponse(
            message=Message(role="assistant", content=content),
            usage=Usage(prompt_tokens, completion_tokens),
        )

    @staticmethod
    def call(name: str, tool_probs: Optional[Dict[str, float]] = None, **arguments) -> ChatResponse:
        return ChatResponse(
            message=Message(
                role="assistant",
                tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)],
            ),
            usage=Usage(20, 15),
            tool_probs=tool_probs,
        )
