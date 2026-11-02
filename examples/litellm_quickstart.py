"""Quickstart using LiteLLM to route to different provider models.

Install LiteLLM with: pip install -e ".[litellm]"
Set OPENAI_API_KEY, ANTHROPIC_API_KEY, and GEMINI_API_KEY before running.
"""

import os

from moslm import Expert, LiteLLMBackend, Orchestrator

orchestrator_model = LiteLLMBackend(
    os.getenv("MOSLM_ORCHESTRATOR_MODEL", "openai/gpt-4o-mini")
)

experts = [
    Expert(
        name="math_expert",
        description="Solves arithmetic, algebra and word problems. Returns the steps and a final number.",
        backend=LiteLLMBackend(
            os.getenv("MOSLM_MATH_MODEL", "anthropic/claude-3-5-haiku-latest")
        ),
        system_prompt="Solve step by step. End with 'Answer: <number>'.",
    ),
    Expert(
        name="code_expert",
        description="Writes, explains and fixes Python code. Returns a code block.",
        backend=LiteLLMBackend(
            os.getenv("MOSLM_CODE_MODEL", "gemini/gemini-2.5-flash")
        ),
        system_prompt="Return only a correct, runnable Python code block.",
    ),
    Expert(
        name="general_expert",
        description="General knowledge, explanations, writing and anything not covered by other experts.",
        backend=LiteLLMBackend(
            os.getenv("MOSLM_GENERAL_MODEL", "openai/gpt-4o-mini")
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
        "Calculate the compound interest on 50,000 at 8% for 3 years, "
        "then write Python code to verify the result."
    )
    result = orch.run(request)
    print("\nANSWER:\n", result.answer)
    print("\nSUMMARY:\n" + result.summary())
