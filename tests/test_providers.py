from types import SimpleNamespace

from moslm import (
    AnthropicBackend,
    GoogleGeminiBackend,
    LiteLLMBackend,
    Message,
    OpenAIBackend,
    ToolCall,
)


TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a value.",
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
    },
}


class OpenAIClient:
    def __init__(self):
        self.request = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="lookup", arguments='{"task":"find"}'),
                )],
            ))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )


def test_openai_backend_maps_tool_calls_and_usage():
    client = OpenAIClient()
    backend = OpenAIBackend("test-model", client=client)

    result = backend.chat([Message(role="user", content="Find it")], tools=[TOOL])

    assert result.message.tool_calls == [ToolCall("call_1", "lookup", {"task": "find"})]
    assert result.usage.prompt_tokens == 5
    assert client.request["tools"] == [TOOL]
    assert client.request["tool_choice"] == "auto"


class AnthropicClient:
    def __init__(self):
        self.request = None
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="I'll look."),
                SimpleNamespace(
                    type="tool_use", id="tool_1", name="lookup", input={"task": "find"}
                ),
            ],
            usage=SimpleNamespace(input_tokens=8, output_tokens=3),
        )


def test_anthropic_backend_maps_tool_messages_and_usage():
    client = AnthropicClient()
    backend = AnthropicBackend("test-model", client=client)
    messages = [
        Message(role="system", content="Be precise."),
        Message(role="user", content="Find it"),
        Message(
            role="assistant",
            tool_calls=[ToolCall("tool_0", "lookup", {"task": "find"})],
        ),
        Message(role="tool", content="Found it", tool_call_id="tool_0", name="lookup"),
    ]

    result = backend.chat(messages, tools=[TOOL])

    assert result.message.content == "I'll look."
    assert result.message.tool_calls == [ToolCall("tool_1", "lookup", {"task": "find"})]
    assert result.usage.total_tokens == 11
    assert client.request["system"] == "Be precise."
    assert client.request["tools"][0]["input_schema"] == TOOL["function"]["parameters"]
    assert client.request["messages"][-1]["content"][0]["type"] == "tool_result"


class GeminiClient:
    def __init__(self):
        self.request = None
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **request):
        self.request = request
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
                SimpleNamespace(text=None, function_call=SimpleNamespace(
                    id="gemini_call", name="lookup", args={"task": "find"}
                )),
            ]))],
            usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=4),
        )


def test_gemini_backend_maps_tools_and_usage():
    client = GeminiClient()
    backend = GoogleGeminiBackend("test-model", client=client)

    result = backend.chat([
        Message(role="system", content="Be precise."),
        Message(role="user", content="Find it"),
    ], tools=[TOOL])

    assert result.message.tool_calls == [ToolCall("gemini_call", "lookup", {"task": "find"})]
    assert result.usage.total_tokens == 11
    assert client.request["config"]["system_instruction"] == "Be precise."
    assert client.request["config"]["tools"][0]["function_declarations"][0]["name"] == "lookup"


def test_litellm_backend_maps_tool_calls_and_usage():
    request = {}

    def completion(**kwargs):
        request.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="lookup", arguments='{"task":"find"}'),
                )],
            ))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )

    backend = LiteLLMBackend(
        "anthropic/claude-test",
        api_key="test-key",
        api_base="https://api.example.test",
        completion=completion,
    )

    result = backend.chat([Message(role="user", content="Find it")], tools=[TOOL])

    assert result.message.tool_calls == [ToolCall("call_1", "lookup", {"task": "find"})]
    assert result.usage.prompt_tokens == 5
    assert request["model"] == "anthropic/claude-test"
    assert request["api_key"] == "test-key"
    assert request["api_base"] == "https://api.example.test"
    assert request["tool_choice"] == "auto"
