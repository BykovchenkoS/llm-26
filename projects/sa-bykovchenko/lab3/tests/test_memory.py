from pathlib import Path
import sys

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

LAB3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB3_ROOT / "source"))

from agent import create_apollo_agent  # noqa: E402


class FakeFinalModel(BaseChatModel):
    answer: str = "ok"

    @property
    def _llm_type(self) -> str:
        return "fake-final"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.answer))]
        )


def _human_contents(agent, config):
    state = agent.get_state(config)
    return [
        message.content
        for message in state.values.get("messages", [])
        if isinstance(message, HumanMessage)
    ]


def test_two_sequential_messages_preserve_history():
    agent = create_apollo_agent(
        model=FakeFinalModel(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "sequential"}}

    agent.invoke(
        {"messages": [{"role": "user", "content": "first question"}]},
        config=config,
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "second question"}]},
        config=config,
    )

    contents = _human_contents(agent, config)
    assert "first question" in contents
    assert "second question" in contents


def test_threads_are_isolated_in_memory():
    agent = create_apollo_agent(
        model=FakeFinalModel(),
        checkpointer=InMemorySaver(),
    )
    first = {"configurable": {"thread_id": "iso-1"}}
    second = {"configurable": {"thread_id": "iso-2"}}

    agent.invoke(
        {"messages": [{"role": "user", "content": "hello one"}]},
        config=first,
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "hello two"}]},
        config=second,
    )

    assert "hello one" in _human_contents(agent, first)
    assert "hello two" not in _human_contents(agent, first)
    assert "hello two" in _human_contents(agent, second)
    assert "hello one" not in _human_contents(agent, second)


def test_sqlite_history_survives_reconnect(tmp_path):
    db_path = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "persist-thread"}}

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        agent = create_apollo_agent(
            model=FakeFinalModel(answer="first answer"),
            checkpointer=checkpointer,
        )
        agent.invoke(
            {"messages": [{"role": "user", "content": "first question"}]},
            config=config,
        )

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        agent = create_apollo_agent(
            model=FakeFinalModel(answer="second answer"),
            checkpointer=checkpointer,
        )
        assert "first question" in _human_contents(agent, config)

        agent.invoke(
            {"messages": [{"role": "user", "content": "second question"}]},
            config=config,
        )
        contents = _human_contents(agent, config)
        assert "first question" in contents
        assert "second question" in contents
