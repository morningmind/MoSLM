"""Plain data types shared across the SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_openai(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            import json
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None and self.role == "tool":
            msg["name"] = self.name
        return msg


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatResponse:
    message: Message
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    # Optional: probability the model assigned to each tool name at the decision point.
    # Filled by backends that return logprobs; used as MoE-style gate scores.
    tool_probs: Optional[Dict[str, float]] = None


@dataclass
class Step:
    """One expert call made by the orchestrator."""
    expert: str
    arguments: Dict[str, Any]
    output: str
    usage: Usage
    latency_ms: float
    cost: float
    escalated: bool = False
    error: Optional[str] = None


@dataclass
class RunResult:
    answer: str
    steps: List[Step]
    orchestrator_usage: Usage
    total_cost: float
    total_latency_ms: float
    stopped_reason: str  # "answered" | "max_steps" | "error"
    orchestrator_model: Optional[str] = None
    expert_models: Dict[str, str] = field(default_factory=dict)

    @property
    def experts_used(self) -> List[str]:
        return [s.expert for s in self.steps]

    @property
    def escalated(self) -> bool:
        return any(s.escalated for s in self.steps)

    def summary(self) -> str:
        route = " -> ".join(self.experts_used) or "(no experts)"
        lines = [
            f"route: {route}\n"
            f"cost: {self.total_cost:.6f}  latency: {self.total_latency_ms:.0f} ms  "
            f"escalated: {self.escalated}  stop: {self.stopped_reason}"
        ]
        lines.append("tokens:")
        orchestrator_label = "orchestrator"
        if self.orchestrator_model:
            orchestrator_label += f" ({self.orchestrator_model})"
        lines.append(
            f"  {orchestrator_label}: input={self.orchestrator_usage.prompt_tokens} "
            f"output={self.orchestrator_usage.completion_tokens} "
            f"total={self.orchestrator_usage.total_tokens}"
        )

        expert_usage: Dict[str, Usage] = {}
        for step in self.steps:
            usage = expert_usage.setdefault(step.expert, Usage())
            usage.prompt_tokens += step.usage.prompt_tokens
            usage.completion_tokens += step.usage.completion_tokens

        for name, usage in expert_usage.items():
            label = name
            model = self.expert_models.get(name)
            if model:
                label += f" ({model})"
            lines.append(
                f"  {label}: input={usage.prompt_tokens} "
                f"output={usage.completion_tokens} total={usage.total_tokens}"
            )
        return "\n".join(lines)
