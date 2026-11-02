"""The orchestrator: a small model that plans, calls experts as tools, and answers."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from .backends import Backend
from .expert import Expert
from .types import Message, RunResult, Step, ToolCall, Usage

DEFAULT_SYSTEM_PROMPT = """You are an orchestrator. You do not solve tasks yourself.
Break the user's request into small subtasks and call the most suitable expert tool for each one.
Pass each expert a self-contained task, including any results from earlier experts it needs.
Call one expert at a time. When you have enough information, reply with the final answer as plain text.
{fallback_note}"""

FALLBACK_NOTE = (
    "Only call {name} when no other expert can handle a subtask or an expert's answer looks wrong; "
    "it is much more expensive."
)


class Orchestrator:
    """Runs the plan -> call expert -> read result -> answer loop.

    Gate scores: if the orchestrator backend returns tool_probs (e.g. from logprobs),
    calls whose top probability is below `min_gate_confidence` are redirected to the
    fallback expert. This is the MoE-style confidence gate described in the proposal.
    """

    def __init__(
        self,
        backend: Backend,
        experts: List[Expert],
        system_prompt: Optional[str] = None,
        max_steps: int = 6,
        orchestrator_cost_per_1k_tokens: float = 0.0005,
        min_gate_confidence: Optional[float] = None,
        on_step: Optional[Callable[[Step], None]] = None,
    ):
        if not experts:
            raise ValueError("Orchestrator needs at least one expert")
        names = [e.name for e in experts]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate expert names: {names}")
        self.backend = backend
        self.experts: Dict[str, Expert] = {e.name: e for e in experts}
        self.max_steps = max_steps
        self.orchestrator_cost_per_1k_tokens = orchestrator_cost_per_1k_tokens
        self.min_gate_confidence = min_gate_confidence
        self.on_step = on_step

        fallback = self.fallback
        note = FALLBACK_NOTE.format(name=fallback.name) if fallback else ""
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT.format(fallback_note=note).strip()

    # -- registry ---------------------------------------------------------
    @property
    def fallback(self) -> Optional[Expert]:
        return next((e for e in self.experts.values() if e.is_fallback), None)

    def add_expert(self, expert: Expert) -> None:
        if expert.name in self.experts:
            raise ValueError(f"Expert {expert.name!r} already registered")
        self.experts[expert.name] = expert

    def remove_expert(self, name: str) -> None:
        self.experts.pop(name)

    def tools(self) -> List[dict]:
        return [e.tool_schema() for e in self.experts.values()]

    # -- main loop ----------------------------------------------------------
    def run(self, request: str) -> RunResult:
        messages: List[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=request),
        ]
        steps: List[Step] = []
        orch_usage = Usage()
        orch_latency = 0.0
        started = time.perf_counter()

        for _ in range(self.max_steps):
            resp = self.backend.chat(messages, tools=self.tools())
            orch_usage.prompt_tokens += resp.usage.prompt_tokens
            orch_usage.completion_tokens += resp.usage.completion_tokens
            orch_latency += resp.latency_ms
            msg = resp.message

            if not msg.tool_calls:
                return self._finish(msg.content or "", steps, orch_usage, started, "answered")

            messages.append(msg)
            for call in msg.tool_calls:
                step = self._call_expert(call, resp.tool_probs)
                steps.append(step)
                if self.on_step:
                    self.on_step(step)
                content = step.output if step.error is None else f"ERROR: {step.error}"
                messages.append(
                    Message(role="tool", content=content, tool_call_id=call.id, name=call.name)
                )

        # Out of steps: return the last expert output as a best effort.
        last = steps[-1].output if steps else ""
        return self._finish(last, steps, orch_usage, started, "max_steps")

    # -- helpers -----------------------------------------------------------
    def _call_expert(self, call: ToolCall, tool_probs: Optional[Dict[str, float]]) -> Step:
        task = str(call.arguments.get("task", "")) or str(call.arguments)
        expert = self.experts.get(call.name)
        escalated = False

        if expert is None:
            return Step(call.name, call.arguments, "", Usage(), 0.0, 0.0,
                        error=f"unknown expert {call.name!r}; available: {list(self.experts)}")

        if (
            self.min_gate_confidence is not None
            and tool_probs
            and not expert.is_fallback
            and self.fallback is not None
            and tool_probs.get(call.name, 0.0) < self.min_gate_confidence
        ):
            expert = self.fallback
            escalated = True
        elif expert.is_fallback:
            escalated = True

        try:
            resp = expert.run(task)
        except Exception as e:  # keep the run alive; the orchestrator sees the error
            return Step(expert.name, call.arguments, "", Usage(), 0.0, 0.0, escalated, error=str(e))

        return Step(
            expert=expert.name,
            arguments=call.arguments,
            output=resp.message.content or "",
            usage=resp.usage,
            latency_ms=resp.latency_ms,
            cost=expert.cost(resp.usage.total_tokens),
            escalated=escalated,
        )

    def _finish(self, answer, steps, orch_usage, started, reason) -> RunResult:
        orch_cost = orch_usage.total_tokens / 1000.0 * self.orchestrator_cost_per_1k_tokens
        orchestrator_model = getattr(self.backend, "model", None)
        expert_models = {
            name: model
            for name, expert in self.experts.items()
            if (model := getattr(expert.backend, "model", None)) is not None
        }
        return RunResult(
            answer=answer,
            steps=steps,
            orchestrator_usage=orch_usage,
            total_cost=orch_cost + sum(s.cost for s in steps),
            total_latency_ms=(time.perf_counter() - started) * 1000,
            stopped_reason=reason,
            orchestrator_model=orchestrator_model,
            expert_models=expert_models,
        )
