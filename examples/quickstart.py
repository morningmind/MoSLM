"""Quickstart using native OpenAI, Anthropic, and Google Gemini SDKs.

Install provider SDKs with: pip install -e ".[providers]"
Set OPENAI_API_KEY, ANTHROPIC_API_KEY, and GEMINI_API_KEY before running.
"""

import os

from moslm import (
    AnthropicBackend,
    Expert,
    GoogleGeminiBackend,
    OpenAIBackend,
    Orchestrator,
)

orchestrator_model = OpenAIBackend(
    os.getenv("MOSLM_ORCHESTRATOR_MODEL", "gpt-4o-mini")
)

experts = [
    Expert(
        name="math_expert",
        description="Solves arithmetic, algebra and word problems. Returns the steps and a final number.",
        backend=AnthropicBackend(
            os.getenv("MOSLM_MATH_MODEL", "claude-sonnet-4-5")
        ),
        system_prompt="Solve step by step. End with 'Answer: <number>'.",
    ),
    Expert(
        name="code_expert",
        description="Writes, explains and fixes Python code. Returns a code block.",
        backend=GoogleGeminiBackend(
            os.getenv("MOSLM_CODE_MODEL", "gemini-2.5-flash")
        ),
        system_prompt="Return only a correct, runnable Python code block.",
    ),
    Expert(
        name="general_expert",
        description="General knowledge, explanations, writing and anything not covered by other experts.",
        backend=OpenAIBackend(
            os.getenv("MOSLM_GENERAL_MODEL", "gpt-4o-mini")
        ),
    ),
]

orch = Orchestrator(
    backend=orchestrator_model,
    experts=experts,
    max_steps=6,
    on_step=lambda s: print(f"EXPERT  -> {s.expert}: {s.output!r}"),
)

if __name__ == "__main__":
    request = (
        "create a lesson on what is Weight in AI?"
        "what is next topic to learn after that?"
    )
    result = orch.run(request)
    print("\nANSWER:\n", result.answer)
    print("\nSUMMARY:\n" + result.summary())
