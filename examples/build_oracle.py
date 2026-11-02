"""Week-1 experiment: does routing have room to help?

Runs each expert on GSM8K and MMLU questions, writes oracle.csv, and prints
best-single-expert vs oracle accuracy. Needs: pip install "moslm[eval]"
"""

import json
import random

from datasets import load_dataset

from moslm import Expert, OpenAICompatibleBackend, build_oracle_table, summarize_oracle

N = 100  # questions per dataset; raise to 300 once it works
random.seed(0)

items = []

gsm = load_dataset("openai/gsm8k", "main", split="test")
for i in random.sample(range(len(gsm)), N):
    ex = gsm[i]
    items.append({
        "id": f"gsm8k-{i}",
        "task": "gsm8k",
        "prompt": ex["question"] + "\nEnd with 'Answer: <number>'.",
        "reference": ex["answer"].split("####")[-1].strip(),
        "scorer": "number",
    })

mmlu = load_dataset("cais/mmlu", "all", split="test")
for i in random.sample(range(len(mmlu)), N):
    ex = mmlu[i]
    opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", ex["choices"]))
    items.append({
        "id": f"mmlu-{i}",
        "task": "mmlu",
        "prompt": f"{ex['question']}\n{opts}\nReply with only the letter.",
        "reference": "ABCD"[ex["answer"]],
        "scorer": "choice",
    })

experts = [
    Expert("math_expert", "math", OpenAICompatibleBackend("http://localhost:8001/v1", "Qwen/Qwen2.5-Math-1.5B-Instruct")),
    Expert("code_expert", "code", OpenAICompatibleBackend("http://localhost:8002/v1", "Qwen/Qwen2.5-Coder-1.5B-Instruct")),
    Expert("general_expert", "general", OpenAICompatibleBackend("http://localhost:8000/v1", "Qwen/Qwen2.5-1.5B-Instruct")),
]

rows = build_oracle_table(items, experts, "oracle.csv")
summary = summarize_oracle(rows)
print(json.dumps(summary, indent=2))

o = summary["overall"]
print(
    f"\nBest single expert: {o['best_expert']} at {o['best_single']:.1%}. "
    f"Oracle: {o['oracle']:.1%}. Headroom: {o['headroom_points']} points."
)
print("Go" if o["headroom_points"] >= 5 else "Weak headroom: try more specialised experts")
