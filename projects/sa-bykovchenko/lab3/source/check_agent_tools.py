# Быстрая проверка, что у LangGraph зарегистрированы нужные инструменты

from langchain_ollama import ChatOllama

from tools import AGENT_TOOLS


# Проверяем регистрацию и имена tools без полного бенчмарка
def main():
    model = ChatOllama(
        model="apollo-qwen:16k",
        temperature=0,
    )
    # Эта проверка привязывает схемы tools к модели без сборки графа
    # Так видно, какие инструменты Qwen может выбрать
    llm = model.bind_tools(AGENT_TOOLS)

    questions = [
        "Найди информацию о подготовке к выходу на Луну.",
        "Вычисли разницу между 125 и 90 с помощью инструмента.",
    ]

    for question in questions:
        response = llm.invoke(question)
        print("\nQUESTION:", question)
        print("TOOL CALLS:", response.tool_calls)
        print("CONTENT:", response.content)


if __name__ == "__main__":
    main()
