"""An expert is a small language model exposed to the orchestrator as a tool."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .backends import Backend
from .types import ChatResponse, Message

_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")


@dataclass
class Expert:
    """A specialist model the orchestrator can call.

    name:        tool name the orchestrator sees, e.g. "math_expert"
    description: what it is good at; this is the main routing signal for a tool-calling orchestrator
    backend:     where the model runs
    system_prompt: instructions sent to the expert on every call
    cost_per_1k_tokens: relative cost used in traces and the RL reward
    is_fallback: True for the large model used only on escalation
    """

    name: str
    description: str
    backend: Backend
    system_prompt: Optional[str] = None
    cost_per_1k_tokens: float = 0.001
    max_tokens: int = 1024
    temperature: float = 0.0
    is_fallback: bool = False
    tags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"Expert name {self.name!r} must be a valid tool name: letters, digits, underscores"
            )

    def tool_schema(self) -> Dict[str, Any]:
        """OpenAI-style function definition shown to the orchestrator."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "A self-contained instruction for this expert, "
                                           "including any inputs it needs from earlier steps.",
                        }
                    },
                    "required": ["task"],
                },
            },
        }

    def run(self, task: str) -> ChatResponse:
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=task))
        return self.backend.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature)

    def cost(self, total_tokens: int) -> float:
        return total_tokens / 1000.0 * self.cost_per_1k_tokens
