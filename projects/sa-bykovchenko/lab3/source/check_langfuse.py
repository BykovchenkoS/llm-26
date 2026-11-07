# Проверка соединения с локальным Langfuse через маленький тестовый trace
# Она не вызывает LLM и не является полноценным запросом к агенту


from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

langfuse = get_client()

# Проверяем ключи Langfuse до отправки событий
if not langfuse.auth_check():
    raise RuntimeError("Langfuse authentication failed")

print("Langfuse authentication: OK")

# Это тестовый span без LLM. Он проверяет отправку данных, а не работу агента
with langfuse.start_as_current_observation(
    name="lab3-first-trace",
    as_type="span",
    input={"question": "Apollo 11 test"},
) as span:
    span.update(
        output={"status": "ok"},
        metadata={"agent": "custom-langgraph"},
    )

langfuse.flush()

print("Trace sent successfully")
