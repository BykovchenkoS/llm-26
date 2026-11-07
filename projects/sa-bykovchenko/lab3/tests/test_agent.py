from pathlib import Path
import asyncio
import json
import sys
import time

import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

LAB3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB3_ROOT / "source"))

import tool_functions  # noqa: E402
import tools  # noqa: E402
from agent import (  # noqa: E402
    SYSTEM_PROMPT_PATH,
    RetryingChatOpenAI,
    build_model,
    create_apollo_agent,
    load_system_prompt,
)
from rag_adapter import get_settings  # noqa: E402
import run_agent  # noqa: E402
from run_agent import (  # noqa: E402
    STATUS_EMPTY_RESPONSE,
    STATUS_MAX_STEPS_EXCEEDED,
    STATUS_TIMEOUT,
    STATUS_TOOL_LIMIT_EXCEEDED,
    build_runtime_config,
    run_turn,
)
from tools import AGENT_TOOLS  # noqa: E402


class FakeToolCallingModel(BaseChatModel):
    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class SlowModel(BaseChatModel):
    delay: float = 0.5

    @property
    def _llm_type(self) -> str:
        return "slow-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(self.delay)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="late"))]
        )


def _calculate_call(expression: str, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate",
                "args": {"expression": expression},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 30,
    }


def test_system_prompt_exists_and_loaded():
    assert SYSTEM_PROMPT_PATH.is_file()
    prompt = load_system_prompt()
    assert "search_documents" in prompt
    assert "get_document_context" in prompt
    assert "calculate" in prompt


def test_settings_loaded_from_yaml():
    settings = get_settings()
    assert settings.model.name
    assert settings.model.tokenizer
    assert settings.agent.max_model_calls >= 1
    assert settings.agent.max_tool_calls >= 1
    assert settings.context.keep_recent_messages >= 0


def test_build_model_uses_config():
    model = build_model()
    assert model.model == get_settings().model.name


def test_build_model_disables_streaming_for_non_ascii_tool_calls():
    model = build_model()
    assert getattr(model, "streaming", None) is False
    assert getattr(model, "disable_streaming", None) is True


def test_build_model_has_configurable_empty_retry():
    model = build_model(empty_retry_limit=2)
    assert getattr(model, "empty_retry_limit", None) == 2
    assert isinstance(getattr(model, "retry_log", None), list)


def test_agent_shares_model_retry_log():
    model = build_model(empty_retry_limit=1)
    agent = create_apollo_agent(
        model=model,
        model_name="apollo-qwen:16k",
    )
    model.retry_log.append({"attempt": 0, "empty": True})
    assert agent.retry_log is model.retry_log
    assert agent.retry_log[-1]["empty"] is True


def test_tokenizer_matches_selected_model():
    settings = get_settings()
    assert settings.model.tokenizer_for("apollo-qwen:16k") == "Qwen/Qwen2.5-3B-Instruct"
    assert settings.model.tokenizer_for("apollo-llama:16k") == "unsloth/Llama-3.2-3B-Instruct"
    assert settings.model.tokenizer_for("gemma4:e2b") == "google/gemma-4-E2B-it"
    assert settings.model.tokenizer_for() == settings.model.tokenizer


def test_unknown_model_tokenizer_fails_fast():
    with pytest.raises(ValueError, match="No tokenizer is configured"):
        get_settings().model.tokenizer_for("unknown:1b")


def test_agent_records_selected_tokenizer():
    agent = create_apollo_agent(
        model=FakeToolCallingModel(responses=[AIMessage(content="ok")]),
        model_name="apollo-qwen:16k",
    )
    assert agent.model_name == "apollo-qwen:16k"
    assert agent.tokenizer_name == "Qwen/Qwen2.5-3B-Instruct"


def test_runtime_config_has_thread_and_limit():
    config = build_runtime_config("test-session")
    settings = get_settings()
    assert config["configurable"]["thread_id"] == "test-session"
    assert config["recursion_limit"] >= 100
    assert config["recursion_limit"] > (
        settings.agent.max_model_calls + settings.agent.max_tool_calls + 1
    )


def test_agent_executes_tool_and_finishes():
    fake_model = FakeToolCallingModel(
        responses=[
            _calculate_call("125 - 90"),
            AIMessage(content="Разница равна 35."),
        ]
    )

    agent = create_apollo_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Вычисли 125 - 90."}]},
        config=_config("test-thread"),
    )

    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert tool_messages
    assert "35" in tool_messages[0].content
    assert messages[-1].content == "Разница равна 35."
    assert fake_model.index == 2


def test_multiple_sequential_tool_calls():
    fake_model = FakeToolCallingModel(
        responses=[
            _calculate_call("1 + 1", "call_a"),
            _calculate_call("2 + 2", "call_b"),
            AIMessage(content="Готово: 2 и 4."),
        ]
    )
    agent = create_apollo_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Сложи два раза."}]},
        config=_config("multi-tool"),
    )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    assert '"result": 2' in tool_messages[0].content
    assert '"result": 4' in tool_messages[1].content
    assert result["messages"][-1].content == "Готово: 2 и 4."


def test_transient_tool_error_is_retried(monkeypatch):
    attempts = {"count": 0}

    def flaky_calculate(expression):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary")
        return {"expression": expression, "result": 4}

    monkeypatch.setattr(tools, "calculate_impl", flaky_calculate)

    fake_model = FakeToolCallingModel(
        responses=[
            _calculate_call("2 + 2"),
            AIMessage(content="Результат 4."),
        ]
    )
    agent = create_apollo_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Вычисли 2 + 2."}]},
        config=_config("retry-tool"),
    )

    assert attempts["count"] == 2
    assert result["messages"][-1].content == "Результат 4."


def test_unknown_chunk_id_is_returned_to_model(monkeypatch):
    class FakeRetriever:
        chunks = [
            {
                "chunk_id": "known",
                "pdf_page": 1,
                "section_id": "A",
                "citation_url": "u",
                "text": "known text",
            }
        ]

    monkeypatch.setattr(
        tool_functions,
        "get_retriever",
        lambda: FakeRetriever(),
    )

    fake_model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_context",
                        "args": {"chunk_id": "wrong_id"},
                        "id": "call_err",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Повторю поиск с корректным chunk_id."),
        ]
    )
    agent = create_apollo_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Уточни фрагмент."}]},
        config=_config("unknown-chunk"),
    )

    errors = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.status == "error"
    ]
    assert errors
    assert "ValueError" in errors[0].content
    assert result["messages"][-1].content == "Повторю поиск с корректным chunk_id."


def test_model_call_limit_returns_explicit_status():
    fake_model = FakeToolCallingModel(responses=[_calculate_call("1 + 1")])
    agent = create_apollo_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
        max_model_calls=1,
    )

    outcome = asyncio.run(
        run_turn(agent, "Вычисли.", _config("limit-model"), timeout_seconds=30)
    )

    assert outcome["status"] == STATUS_MAX_STEPS_EXCEEDED


def test_tool_call_limit_returns_explicit_status():
    fake_model = FakeToolCallingModel(responses=[_calculate_call("1 + 1")])
    agent = create_apollo_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
        max_model_calls=8,
        max_tool_calls=1,
    )

    outcome = asyncio.run(
        run_turn(agent, "Вычисли.", _config("limit-tool"), timeout_seconds=30)
    )

    assert outcome["status"] == STATUS_TOOL_LIMIT_EXCEEDED


def test_timeout_returns_explicit_status():
    agent = create_apollo_agent(
        model=SlowModel(delay=0.5),
        checkpointer=InMemorySaver(),
    )

    outcome = asyncio.run(
        run_turn(agent, "Вопрос.", _config("timeout"), timeout_seconds=0.05)
    )

    assert outcome["status"] == STATUS_TIMEOUT


def test_agent_uses_all_three_tools():
    assert [tool.name for tool in AGENT_TOOLS] == [
        "search_documents",
        "get_document_context",
        "calculate",
    ]


def test_run_turn_with_retry_retries_empty_message(monkeypatch):
    calls = []

    async def fake_run_traced_turn(**kwargs):
        calls.append(kwargs["question"])
        content = "" if len(calls) == 1 else "retried answer"
        return {
            "status": run_agent.STATUS_OK,
            "result": {"messages": [AIMessage(content=content)]},
            "error": None,
            "duration_s": 0.1,
        }

    monkeypatch.setattr(run_agent, "run_traced_turn", fake_run_traced_turn)
    outcome = asyncio.run(
        run_agent.run_turn_with_retry(
            agent=None,
            question="test question",
            config={},
            session_id="retry-test",
            turn_id=1,
            timeout_seconds=10,
            max_attempts=2,
        )
    )

    assert len(calls) == 2
    assert outcome["result"]["messages"][-1].content == "retried answer"
    assert outcome["status"] == run_agent.STATUS_OK


def test_run_turn_with_retry_reports_empty_after_attempts(monkeypatch):
    async def fake_run_traced_turn(**kwargs):
        return {
            "status": run_agent.STATUS_OK,
            "result": {"messages": [AIMessage(content="")]},
            "error": None,
            "duration_s": 0.1,
        }

    monkeypatch.setattr(run_agent, "run_traced_turn", fake_run_traced_turn)
    outcome = asyncio.run(
        run_agent.run_turn_with_retry(
            agent=None,
            question="test question",
            config={},
            session_id="retry-empty",
            turn_id=1,
            timeout_seconds=10,
            max_attempts=3,
        )
    )

    assert outcome["status"] == STATUS_EMPTY_RESPONSE
    assert len(outcome["attempts"]) == 3
    assert outcome["retry_count"] == 2
    assert outcome["attempts"][1]["prompt_variant"].startswith("original+U+200B")


def test_message_trace_preserves_full_json_tool_result():
    payload = {
        "results": [
            {"chunk_id": "chunk-1", "section_id": "4.12.2", "pdf_page": 40, "text": "x" * 9000}
        ],
        "evidence": {"heuristic_sufficient": True},
    }
    raw = json.dumps(payload, ensure_ascii=False)
    message = ToolMessage(
        content=raw,
        tool_call_id="call-1",
        name="search_documents",
        status="success",
    )
    trace = run_agent._message_trace([message])
    saved = trace["tool_messages"][0]
    assert saved["content"] == raw
    assert saved["content_json"] == payload
    assert saved["content_length"] == len(raw)
    assert len(saved["content_preview"]) <= 600


def test_model_level_empty_retry_logs_attempts(monkeypatch):
    calls = []

    def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        calls.append(messages)
        content = "" if len(calls) == 1 else "answer"
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    monkeypatch.setattr(
        "langchain_openai.ChatOpenAI._generate",
        fake_generate,
    )
    model = RetryingChatOpenAI(
        model="apollo-qwen:16k",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        temperature=0,
        streaming=False,
        disable_streaming=True,
        empty_retry_limit=1,
    )
    result = model._generate([HumanMessage(content="test")])

    assert result.generations[0].message.content == "answer"
    assert len(calls) == 2
    assert "\u200b" in calls[1][-1].content
    assert len(model.retry_log) == 2
    assert model.retry_log[0]["empty"] is True
    assert model.retry_log[1]["nonce_used"] is True
    assert model.retry_log[1]["reason"] == "ok"
