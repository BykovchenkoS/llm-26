# Отдельная проверка Python-инструментов без полного агентного цикла

from tools import AGENT_TOOLS


# Вызываем Python-инструменты напрямую, чтобы отделить их ошибки от LLM
def main():
    for tool in AGENT_TOOLS:
        print(f"{tool.name}: {tool.args}")

    search, context, calculate = AGENT_TOOLS

    # Здесь напрямую проверяем работу поискового инструмента. Это еще не самостоятельное решение агента вызвать его
    result = search.invoke(
        {
            "query": "extravehicular preparation",
            "k": 3,
        }
    )
    print("Search:", result)

    chunk_id = result["results"][0]["chunk_id"]
    details = context.invoke(
        {
            "chunk_id": chunk_id,
        }
    )
    print("Context:", details)

    calculation = calculate.invoke(
        {
            "expression": "(125 - 90) / 90 * 100",
        }
    )
    print("Calculation:", calculation)


if __name__ == "__main__":
    main()
