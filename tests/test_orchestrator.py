import pytest

from moslm import Expert, MockBackend, Orchestrator, build_oracle_table, summarize_oracle


def make_experts(fallback=False):
    experts = [
        Expert("math_expert", "math", MockBackend([MockBackend.text("Answer: 62985.6")]),
               cost_per_1k_tokens=1.0),
        Expert("code_expert", "code", MockBackend([MockBackend.text("```python\nprint(1)\n```")]),
               cost_per_1k_tokens=1.0),
    ]
    if fallback:
        experts.append(Expert("large_model", "big", MockBackend([MockBackend.text("big answer")]),
                              cost_per_1k_tokens=10.0, is_fallback=True))
    return experts


def test_two_step_route_and_answer():
    orch_backend = MockBackend([
        MockBackend.call("math_expert", task="compound interest"),
        MockBackend.call("code_expert", task="write checker"),
        MockBackend.text("Final: 62985.6, plus code."),
    ])
    orch = Orchestrator(orch_backend, make_experts())
    res = orch.run("compute then code")

    assert res.answer == "Final: 62985.6, plus code."
    assert res.experts_used == ["math_expert", "code_expert"]
    assert res.stopped_reason == "answered"
    assert not res.escalated
    assert res.total_cost > 0
    # The expert result was fed back to the orchestrator as a tool message.
    last_msgs = orch_backend.calls[-1]
    assert any(m.role == "tool" and "62985.6" in (m.content or "") for m in last_msgs)


def test_summary_shows_orchestrator_and_per_expert_token_counts():
    orch_backend = MockBackend([
        MockBackend.call("math_expert", task="first"),
        MockBackend.call("math_expert", task="second"),
        MockBackend.text("done", prompt_tokens=13, completion_tokens=7),
    ])
    math_backend = MockBackend([
        MockBackend.text("one", prompt_tokens=4, completion_tokens=2),
        MockBackend.text("two", prompt_tokens=6, completion_tokens=3),
    ])
    orch_backend.model = "router-model"
    math_backend.model = "math-model"
    orch = Orchestrator(
        orch_backend,
        [Expert("math_expert", "math", math_backend)],
    )

    summary = orch.run("do math").summary()

    assert "orchestrator (router-model): input=53 output=37 total=90" in summary
    assert "math_expert (math-model): input=10 output=5 total=15" in summary


def test_tool_schema_shape():
    schema = make_experts()[0].tool_schema()
    assert schema["function"]["name"] == "math_expert"
    assert schema["function"]["parameters"]["required"] == ["task"]


def test_unknown_expert_returns_error_to_orchestrator():
    orch_backend = MockBackend([
        MockBackend.call("no_such_expert", task="x"),
        MockBackend.text("gave up"),
    ])
    res = Orchestrator(orch_backend, make_experts()).run("hi")
    assert res.steps[0].error and "unknown expert" in res.steps[0].error
    assert res.answer == "gave up"


def test_low_gate_confidence_escalates_to_fallback():
    orch_backend = MockBackend([
        MockBackend.call("math_expert", tool_probs={"math_expert": 0.3}, task="hard"),
        MockBackend.text("done"),
    ])
    orch = Orchestrator(orch_backend, make_experts(fallback=True), min_gate_confidence=0.5)
    res = orch.run("hard question")
    assert res.experts_used == ["large_model"]
    assert res.escalated


def test_max_steps_stops_loop():
    orch_backend = MockBackend(lambda msgs, tools: MockBackend.call("math_expert", task="again"))
    math = Expert("math_expert", "math",
                  MockBackend(lambda m, t: MockBackend.text("Answer: 1")))
    res = Orchestrator(orch_backend, [math], max_steps=3).run("loop")
    assert res.stopped_reason == "max_steps"
    assert len(res.steps) == 3


def test_duplicate_names_rejected():
    e = make_experts()[0]
    with pytest.raises(ValueError):
        Orchestrator(MockBackend([]), [e, e])


def test_add_expert_shows_up_as_tool():
    orch = Orchestrator(MockBackend([]), make_experts())
    orch.add_expert(Expert("sql_expert", "sql", MockBackend([])))
    assert "sql_expert" in [t["function"]["name"] for t in orch.tools()]


def test_oracle_headroom(tmp_path):
    items = [
        {"id": "q1", "task": "gsm8k", "prompt": "1+1", "reference": "2", "scorer": "number"},
        {"id": "q2", "task": "mmlu", "prompt": "pick", "reference": "B", "scorer": "choice"},
    ]
    a = Expert("a", "a", MockBackend(lambda m, t: MockBackend.text("Answer: 2")))
    b = Expert("b", "b", MockBackend(lambda m, t: MockBackend.text("B")))
    rows = build_oracle_table(items, [a, b], str(tmp_path / "o.csv"), progress=False)
    s = summarize_oracle(rows)["overall"]
    assert s["best_single"] == 0.5
    assert s["oracle"] == 1.0
    assert s["headroom_points"] == 50.0
