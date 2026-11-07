from pathlib import Path
import asyncio
import sys

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

LAB3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB3_ROOT / "source"))

from context_budget import (  # noqa: E402
    BudgetedSummarizationMiddleware,
    ContextBudgetError,
    ContextReporter,
    QwenTokenCounter,
)
from rag_adapter import ContextConfig, get_settings  # noqa: E402
from tools import AGENT_TOOLS  # noqa: E402


class SummaryModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "summary-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="summary"))]
        )


class CharCounter:
    def __call__(self, messages):
        return sum(len(str(getattr(message, "content", ""))) for message in messages)


def make_config(**overrides) -> ContextConfig:
    values = {
        "max_tokens": 100,
        "response_reserve": 10,
        "strategy": "summarize",
        "keep_recent_messages": 2,
    }
    values.update(overrides)
    return ContextConfig(**values)


def make_middleware(config, reporter, token_counter):
    return BudgetedSummarizationMiddleware(
        model=SummaryModel(),
        context_config=config,
        reporter=reporter,
        tokenizer_name="unused",
        tools=None,
        token_counter=token_counter,
    )


def test_no_compaction_under_budget():
    config = make_config()
    reporter = ContextReporter(thread_id="t1")
    middleware = make_middleware(config, reporter, CharCounter())

    state = {"messages": [HumanMessage(content="short")]}
    result = middleware.before_model(state, None)

    assert result is None
    diagnostics = reporter.last()
    assert diagnostics["thread_id"] == "t1"
    assert diagnostics["compacted"] is False
    assert diagnostics["tokenizer"] == "unused"
    assert diagnostics["tokens_before"] == diagnostics["tokens_after"] == 5
    assert diagnostics["current_user_message_preserved"] is True


def test_compaction_summarizes_old_history():
    config = make_config(keep_recent_messages=1)
    reporter = ContextReporter(thread_id="t2")
    middleware = make_middleware(config, reporter, CharCounter())

    messages = [
        HumanMessage(content="a" * 40),
        AIMessage(content="b" * 40),
        HumanMessage(content="c" * 40),
        HumanMessage(content="current query"),
    ]
    result = middleware.before_model({"messages": messages}, None)

    assert result is not None
    diagnostics = reporter.last()
    assert diagnostics["compacted"] is True
    assert diagnostics["tokens_before"] > config.max_tokens - config.response_reserve
    assert diagnostics["tokens_after"] <= config.max_tokens - config.response_reserve
    assert diagnostics["messages_after"] < diagnostics["messages_before"]
    assert diagnostics["current_user_message_preserved"] is True
    assert diagnostics["strategy"] == "summarize"
    assert diagnostics["summary_created"] is True
    assert diagnostics["summary_retained"] is True
    assert diagnostics["compaction_method"] == "summarize"
    assert diagnostics["removed_message_ids"] == []
    assert len(diagnostics["summarized_message_ids"]) == 3
    assert diagnostics["summarized_message_types"] == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]


def test_oversized_current_message_raises_clear_error():
    config = make_config(keep_recent_messages=1)
    reporter = ContextReporter(thread_id="t3")
    middleware = make_middleware(config, reporter, CharCounter())

    messages = [
        HumanMessage(content="small"),
        HumanMessage(content="x" * 200),
    ]

    with pytest.raises(ContextBudgetError):
        middleware.before_model({"messages": messages}, None)

    diagnostics = reporter.last()
    assert diagnostics["compacted"] is True
    assert diagnostics["tokens_after"] > config.max_tokens - config.response_reserve
    assert diagnostics["current_user_message_preserved"] is True


def test_async_before_model_records_diagnostics():
    config = make_config()
    reporter = ContextReporter(thread_id="async-t")
    middleware = make_middleware(config, reporter, CharCounter())

    result = asyncio.run(
        middleware.abefore_model({"messages": [HumanMessage(content="short")]}, None)
    )

    assert result is None
    diagnostics = reporter.last()
    assert diagnostics["thread_id"] == "async-t"
    assert diagnostics["compacted"] is False
    assert diagnostics["current_user_message_preserved"] is True


def test_async_compaction_enforces_budget():
    config = make_config(keep_recent_messages=1)
    reporter = ContextReporter(thread_id="async-compact")
    middleware = make_middleware(config, reporter, CharCounter())

    messages = [
        HumanMessage(content="a" * 40),
        AIMessage(content="b" * 40),
        HumanMessage(content="c" * 40),
        HumanMessage(content="current query"),
    ]
    result = asyncio.run(
        middleware.abefore_model({"messages": messages}, None)
    )

    assert result is not None
    diagnostics = reporter.last()
    assert diagnostics["compacted"] is True
    assert diagnostics["tokens_after"] <= diagnostics["input_budget"]
    assert diagnostics["current_user_message_preserved"] is True


def test_chat_counter_counts_system_and_tools():
    settings = get_settings()
    tokenizer = settings.model.tokenizer_for("apollo-qwen:16k")
    with_system_and_tools = QwenTokenCounter(
        tokenizer,
        AGENT_TOOLS,
        system_prompt="You are a research assistant.",
    )
    plain = QwenTokenCounter(tokenizer)

    full = with_system_and_tools.count([HumanMessage(content="hello")])
    minimal = plain.count([HumanMessage(content="hello")])

    assert full > minimal > 0


def test_compaction_preserves_tool_call_pairs():
    config = make_config(
        max_tokens=210,
        response_reserve=10,
        keep_recent_messages=2,
    )
    reporter = ContextReporter(thread_id="tool-compaction")
    middleware = make_middleware(config, reporter, CharCounter())

    call_1 = {
        "name": "calculate",
        "args": {"expression": "1 + 1"},
        "id": "c1",
        "type": "tool_call",
    }
    call_2 = {
        "name": "calculate",
        "args": {"expression": "2 + 2"},
        "id": "c2",
        "type": "tool_call",
    }
    messages = [
        HumanMessage(content="a" * 100),
        AIMessage(content="", tool_calls=[call_1]),
        ToolMessage(content="t" * 100, tool_call_id="c1"),
        HumanMessage(content="b" * 100),
        AIMessage(content="", tool_calls=[call_2]),
        ToolMessage(content="t" * 100, tool_call_id="c2"),
        HumanMessage(content="current query"),
    ]

    result = middleware.before_model({"messages": messages}, None)

    assert result is not None
    after = [
        message
        for message in result["messages"]
        if not isinstance(message, RemoveMessage)
    ]
    call_ids = {
        call["id"]
        for message in after
        if isinstance(message, AIMessage)
        for call in getattr(message, "tool_calls", [])
    }
    for message in after:
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in call_ids

    diagnostics = reporter.last()
    assert diagnostics["compacted"] is True
    assert diagnostics["tokens_after"] <= diagnostics["input_budget"]
    assert diagnostics["current_user_message_preserved"] is True


def test_insufficient_summarization_triggers_fallback():
    config = make_config(
        max_tokens=100,
        response_reserve=10,
        keep_recent_messages=2,
    )
    reporter = ContextReporter(thread_id="fallback-history")
    middleware = make_middleware(config, reporter, CharCounter())

    messages = [
        HumanMessage(content="a" * 40),
        AIMessage(content="b" * 40),
        HumanMessage(content="c" * 40),
        HumanMessage(content="current query"),
    ]
    result = middleware.before_model({"messages": messages}, None)

    assert result is not None
    diagnostics = reporter.last()
    assert diagnostics["fallback_used"] is True
    assert diagnostics["compacted"] is True
    assert diagnostics["tokens_after"] <= config.max_tokens - config.response_reserve
    assert diagnostics["current_user_message_preserved"] is True
    assert diagnostics["summary_created"] is True
    assert diagnostics["summary_retained"] is False
    assert diagnostics["compaction_method"] == "summarize+fallback"
    assert diagnostics["removed_message_ids"]
    assert diagnostics["removed_message_types"]
    assert diagnostics["summarized_message_ids"]
    assert diagnostics["summarized_message_types"] == [
        "HumanMessage",
        "AIMessage",
    ]
    assert any(
        isinstance(message, HumanMessage) and message.content == "current query"
        for message in result["messages"]
        if not isinstance(message, RemoveMessage)
    )


def test_fallback_truncates_large_tool_result_and_keeps_pair():
    config = make_config(
        max_tokens=200,
        response_reserve=10,
        keep_recent_messages=2,
    )
    reporter = ContextReporter(thread_id="fallback-tool")
    middleware = make_middleware(config, reporter, CharCounter())

    call = {
        "name": "search_documents",
        "args": {"question": "current question", "query": "current question"},
        "id": "call-large",
        "type": "tool_call",
    }
    tool_text = "section 11.6: 77 minutes; source chunk abc123. " + ("x" * 800)
    messages = [
        HumanMessage(content="current question"),
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content=tool_text, tool_call_id="call-large"),
    ]

    result = middleware.before_model({"messages": messages}, None)

    assert result is not None
    after = [
        message
        for message in result["messages"]
        if not isinstance(message, RemoveMessage)
    ]
    diagnostics = reporter.last()
    assert diagnostics["fallback_used"] is True
    assert diagnostics["truncated_tool_messages"] == 1
    assert diagnostics["truncated_tool_message_ids"]
    assert diagnostics["compaction_method"] == "summarize+fallback+tool_truncation"
    assert diagnostics["tokens_after"] <= config.max_tokens - config.response_reserve
    assert diagnostics["current_user_message_preserved"] is True

    ai_call_ids = {
        call["id"]
        for message in after
        if isinstance(message, AIMessage)
        for call in getattr(message, "tool_calls", [])
    }
    tool_messages = [message for message in after if isinstance(message, ToolMessage)]
    assert tool_messages
    assert all(message.tool_call_id in ai_call_ids for message in tool_messages)
    assert ai_call_ids == {message.tool_call_id for message in tool_messages}
    assert len(tool_messages[0].content) < len(tool_text)
    assert "77 minutes" in tool_messages[0].content
    assert "abc123" in tool_messages[0].content


def test_compaction_creates_separate_span(monkeypatch):
    class FakeObservation:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeClient:
        def __init__(self):
            self.compaction_call = None
            self.parent_update = None

        def update_current_span(self, **kwargs):
            self.parent_update = kwargs

        def start_as_current_observation(self, **kwargs):
            self.compaction_call = kwargs
            return FakeObservation()

    fake = FakeClient()
    monkeypatch.setattr("context_budget.get_client", lambda: fake)

    config = make_config(keep_recent_messages=1)
    reporter = ContextReporter(thread_id="span-test")
    middleware = make_middleware(config, reporter, CharCounter())
    messages = [
        HumanMessage(content="a" * 40),
        AIMessage(content="b" * 40),
        HumanMessage(content="c" * 40),
        HumanMessage(content="current query"),
    ]

    middleware.before_model({"messages": messages}, None)

    assert fake.compaction_call is not None
    assert fake.compaction_call["name"] == "context_compaction"
    output = fake.compaction_call["output"]
    assert output["summary_created"] is True
    assert output["summary_retained"] is True
    assert output["compaction_method"] == "summarize"
    assert output["current_user_message_preserved"] is True
    assert output["fits_budget"] is True
    assert output["history_after"]
    assert output["summarized_messages"]
    assert fake.parent_update["metadata"]["compaction_method"] == "summarize"


def test_async_fallback_handles_large_tool_result():
    config = make_config(
        max_tokens=200,
        response_reserve=10,
        keep_recent_messages=2,
    )
    reporter = ContextReporter(thread_id="async-fallback-tool")
    middleware = make_middleware(config, reporter, CharCounter())

    call = {
        "name": "search_documents",
        "args": {"question": "current question"},
        "id": "call-async",
        "type": "tool_call",
    }
    messages = [
        HumanMessage(content="current question"),
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content="t" * 800, tool_call_id="call-async"),
    ]

    result = asyncio.run(
        middleware.abefore_model({"messages": messages}, None)
    )

    assert result is not None
    diagnostics = reporter.last()
    assert diagnostics["fallback_used"] is True
    assert diagnostics["tokens_after"] <= diagnostics["input_budget"]
    assert diagnostics["current_user_message_preserved"] is True


def test_compaction_with_real_qwen_tokenizer():
    settings = get_settings()
    config = ContextConfig(
        max_tokens=4096,
        response_reserve=512,
        strategy="summarize",
        keep_recent_messages=2,
    )
    reporter = ContextReporter(thread_id="long-dialog")
    middleware = BudgetedSummarizationMiddleware(
        model=SummaryModel(),
        context_config=config,
        reporter=reporter,
        tokenizer_name=settings.model.tokenizer_for("apollo-qwen:16k"),
        tools=AGENT_TOOLS,
        system_prompt="You are a research assistant.",
    )

    messages = [
        HumanMessage(content="filler word " * 400)
        for _ in range(10)
    ]
    messages.append(HumanMessage(content="current question"))

    result = middleware.before_model({"messages": messages}, None)

    diagnostics = reporter.last()
    assert result is not None
    assert diagnostics["compacted"] is True
    assert diagnostics["tokens_before"] > diagnostics["input_budget"]
    assert diagnostics["tokens_after"] <= diagnostics["input_budget"]
    assert diagnostics["current_user_message_preserved"] is True
    assert diagnostics["strategy"] == "summarize"
