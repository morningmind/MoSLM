# MoSLM SDK

Orchestrate specialized **small language models** as tools. A small orchestrator model
breaks a request into subtasks, calls the right expert for each one, and escalates to a
large model only when needed. Mixture-of-Experts, one level up.

The core has zero runtime dependencies. Use native optional adapters for OpenAI,
Anthropic, or Google Gemini, or connect to any OpenAI-compatible server such as vLLM,
Ollama, llama.cpp, LM Studio, or SGLang.

## Install

```bash
pip install -e .            # core
pip install -e ".[eval]"    # + datasets, for the oracle experiment
pip install -e ".[dev]"     # + pytest
pip install -e ".[openai]"  # + OpenAI SDK
pip install -e ".[anthropic]"  # + Anthropic SDK
pip install -e ".[gemini]"  # + Google Gen AI SDK
pip install -e ".[litellm]"  # + LiteLLM unified provider API
pip install -e ".[providers]"  # + all three provider SDKs
```

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and/or `GEMINI_API_KEY` in your environment.
Each native adapter also accepts an explicit `api_key`.

## Quickstart

The runnable provider example mixes OpenAI, Anthropic, and Gemini:

```bash
pip install -e ".[providers]"
python examples/quickstart.py
```

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY` before running.
Model names can be overridden with `MOSLM_ORCHESTRATOR_MODEL`, `MOSLM_MATH_MODEL`,
`MOSLM_CODE_MODEL`, and `MOSLM_GENERAL_MODEL`.

To route each specialist through LiteLLM, install the LiteLLM extra and run the
separate example. It uses different provider models for the orchestrator and experts:

```bash
pip install -e ".[litellm]"
python examples/litellm_quickstart.py
```

Set the API keys for the providers used by the default model names
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`). Override the
LiteLLM model names with the same `MOSLM_*_MODEL` environment variables.

You can also mix native SDK backends and LiteLLM in one orchestrator. This example
uses native OpenAI for orchestration, LiteLLM for the Anthropic math expert, and
the native Gemini SDK for the code expert:

```bash
pip install -e ".[providers,litellm]"
python examples/mixed_providers.py
```

```python
from moslm import Expert, OpenAICompatibleBackend, Orchestrator

orch = Orchestrator(
    backend=OpenAICompatibleBackend("http://localhost:8000/v1", "Qwen/Qwen2.5-1.5B-Instruct"),
    experts=[
        Expert(
            name="math_expert",
            description="Solves arithmetic and word problems. Returns a final number.",
            backend=OpenAICompatibleBackend("http://localhost:8001/v1", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
        ),
        Expert(
            name="code_expert",
            description="Writes and fixes Python code.",
            backend=OpenAICompatibleBackend("http://localhost:8002/v1", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        ),
    ],
)

result = orch.run("Compute 8% compound interest on 50,000 for 3 years, then write Python to check it.")
print(result.answer)
print(result.summary())   # route, cost, latency, escalation
```

Swap a backend for a provider SDK without changing the orchestrator or expert setup:

```python
from moslm import AnthropicBackend, Expert, OpenAIBackend, Orchestrator

orch = Orchestrator(
    backend=OpenAIBackend("gpt-4o-mini"),
    experts=[
        Expert("research", "Answers general questions.", AnthropicBackend("claude-sonnet-4-5")),
    ],
)
print(orch.run("Explain mixture-of-experts briefly.").answer)
```

`GoogleGeminiBackend("gemini-2.5-flash")` works the same way after installing
`.[gemini]`. Existing OpenAI-compatible services continue to use
`OpenAICompatibleBackend(base_url, model, api_key=...)`.

Alternatively, install `.[litellm]` and use `LiteLLMBackend` to access LiteLLM's
supported providers through one adapter. Use the model names and provider prefixes
required by LiteLLM, such as `openai/gpt-4o-mini`,
`anthropic/claude-3-5-haiku-latest`, or `gemini/gemini-2.5-flash`. LiteLLM can also
route to additional providers and local model services; follow its model naming and
credential configuration for the provider you select.

```python
from moslm import Expert, LiteLLMBackend, Orchestrator

orch = Orchestrator(
    backend=LiteLLMBackend("openai/gpt-4o-mini"),
    experts=[
        Expert(
            "research",
            "Answers general questions.",
            LiteLLMBackend("anthropic/claude-3-5-haiku-latest"),
        ),
    ],
)
print(orch.run("Explain mixture-of-experts briefly.").answer)
```

The orchestrator model must support tool calling. With vLLM:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 --enable-auto-tool-choice --tool-call-parser hermes
```

## Concepts

| Piece | What it does |
| --- | --- |
| `Expert` | A model exposed as a tool: `name`, `description` (the routing signal), `backend`, `system_prompt`, `cost_per_1k_tokens` |
| `Expert(is_fallback=True)` | The large model, used only on escalation |
| `Orchestrator` | Runs plan → call expert → read result → answer, up to `max_steps` |
| `RunResult` | `answer`, `steps` (every expert call), `total_cost`, `escalated`, `summary()` with per-orchestrator and per-expert input/output token totals |
| `MockBackend` | Scripted backend for tests and offline demos |
| `build_oracle_table` | Runs every expert on every question and writes a CSV |
| `summarize_oracle` | Best single expert vs oracle accuracy: the routing headroom |

### Adding an expert at runtime

```python
orch.add_expert(Expert(name="sql_expert", description="Writes SQL queries.", backend=...))
```

No retraining: the orchestrator sees it as a new tool on the next call.

### Confidence gate and escalation

If the orchestrator backend returns probabilities for each tool name (`ChatResponse.tool_probs`),
set `min_gate_confidence=0.5`. Calls below that confidence are sent to the fallback expert instead.
This is the MoE-style gate from the proposal. Backends that do not return `tool_probs` skip it.

## The first experiment

```bash
python examples/build_oracle.py
```

Writes `oracle.csv` and prints the headroom. If the oracle is at least 5 points above the best
single expert, routing has room to help.

## Tests

```bash
pytest -q
```

## Roadmap

- [ ] `tool_probs` from vLLM logprobs (read the first token of each tool name)
- [ ] Parallel expert calls for independent subtasks
- [ ] Trajectory logging to JSONL for supervised fine-tuning of the orchestrator
- [ ] GRPO training loop with the cost and load-balancing reward
- [ ] Embedding-classifier router baseline
- [ ] Code-execution scorer for MBPP / HumanEval

## License

Apache-2.0
