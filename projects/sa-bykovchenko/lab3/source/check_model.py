from __future__ import annotations
# Небольшая проверка, что локальная модель поддерживает tool calling


import argparse

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integer numbers."""
    return a * b


# Отдельно проверяем, умеет ли локальная модель возвращать tool_calls
def check_model(model_name: str) -> bool:
    model = ChatOllama(
        model=model_name,
        temperature=0,
    )
    model_with_tools = model.bind_tools([multiply])
    # Проверяем, что модель умеет вернуть структурированный tool call, а не только свободный текст
    response = model_with_tools.invoke(
        "Вызови инструмент multiply, чтобы умножить 17 на 23."
    )

    print(f"Модель: {model_name}")
    print("Ответ модели:", response.content)
    print("Вызовы инструментов:", response.tool_calls)

    if not response.tool_calls:
        print("Tool calling НЕ работает: модель не вызвала инструмент.")
        return False
    if response.tool_calls[0]["name"] != "multiply":
        print("Tool calling НЕ работает: вызван неверный инструмент.")
        return False

    print("Tool calling работает.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка нативного tool calling у Ollama-модели.")
    parser.add_argument(
        "--model",
        default="apollo-qwen:16k",
        help="Имя модели в Ollama (по умолчанию apollo-qwen:16k).",
    )
    args = parser.parse_args()

    return 0 if check_model(args.model) else 1


if __name__ == "__main__":
    raise SystemExit(main())
