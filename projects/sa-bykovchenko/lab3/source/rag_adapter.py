from __future__ import annotations
# Подключение готового RAG-индекса из Лабы 2
# В Лабе 3 не переиндексируем PDF, используем уже сохраненный retriever


from functools import lru_cache
from pathlib import Path
import sys
from typing import Literal

import yaml
from pydantic import BaseModel, Field

LAB3_ROOT = Path(__file__).resolve().parents[1]
LAB2_ROOT = LAB3_ROOT.parent / "lab2"
LAB2_SOURCE = LAB2_ROOT / "source"
CONFIG_PATH = LAB3_ROOT / "config" / "agent.yaml"

sys.path.insert(0, str(LAB2_SOURCE))

from source_loader import load_source


class ProjectConfig(BaseModel):
    name: str


class RetrievalConfig(BaseModel):
    store_id: str
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    backend: Literal["flat_ip", "hnsw"] = "hnsw"
    top_k: int = Field(default=5, ge=1, le=10)
    candidate_k: int = Field(default=50, ge=1)


class ModelConfig(BaseModel):
    provider: str = "ollama"
    name: str
    base_url: str = "http://127.0.0.1:11434/v1"
    temperature: float = 0.0
    timeout_seconds: int = Field(default=60, ge=1)

    # Токенизатор по умолчанию используется только для модели, указанной в конфигурации

    # Для других моделей токенизатор должен быть задан в словаре tokenizers,
    # чтобы не считать размер контекста токенизатором от другой модели
    tokenizer: str = "Qwen/Qwen2.5-3B-Instruct"
    tokenizers: dict[str, str] = Field(default_factory=dict)

    # Токенизатор выбирается по реальной модели запуска, чтобы не считать Qwen как Llama
    def tokenizer_for(self, model_name: str | None = None) -> str:
        name = (model_name or self.name).strip()
        # Для выбранной через --model Qwen берём явное соответствие в YAML
        if name in self.tokenizers:
            return self.tokenizers[name]
        if name == self.name:
            return self.tokenizer
        configured = ", ".join(sorted(self.tokenizers)) or "<none>"
        raise ValueError(
            f"No tokenizer is configured for model {name!r}. "
            f"Add model.tokenizers.{name} to config/agent.yaml. "
            f"Configured models: {configured}."
        )


class AgentConfig(BaseModel):
    max_model_calls: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=12, ge=1)
    max_tool_retries: int = Field(default=1, ge=0, le=5)
    timeout_seconds: int = Field(default=120, ge=1)


class MemoryConfig(BaseModel):
    backend: Literal["sqlite", "memory"] = "sqlite"
    db_path: str = "data/agent_checkpoints.sqlite"


class ContextConfig(BaseModel):
    max_tokens: int = Field(default=4096, ge=1)
    response_reserve: int = Field(default=512, ge=0)
    strategy: str = "summarize"
    keep_recent_messages: int = Field(default=6, ge=0)


class AgentSettings(BaseModel):
    project: ProjectConfig
    retrieval: RetrievalConfig
    model: ModelConfig
    agent: AgentConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig


# Читаем YAML один раз и валидируем параметры с помощью Pydantic
# Не загружаем FAISS/эмбеддинги заново для каждой реплики
@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Не найден файл конфигурации: {CONFIG_PATH}")
    # YAML хранит настройки, а Pydantic ниже проверит типы и диапазоны
    # Так ошибки конфигурации выявляются до запуска агента
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return AgentSettings.model_validate(data)


# Составляем путь именно к готовому хранилищу нужной конфигурации Лабы 2
def get_store_path() -> Path:
    settings = get_settings()
    return (
        LAB2_ROOT
        / "data"
        / "vector_stores"
        / "apollo11"
        / settings.retrieval.store_id
    )


STORE_PATH = get_store_path()


# Загрузка индекса дорогая, поэтому кешируем retriever в пределах процесса
@lru_cache(maxsize=1)
def get_retriever():
    """Один раз загружает сохранённый индекс и E5."""
    if not STORE_PATH.is_dir():
        raise FileNotFoundError(f"Не найдено хранилище: {STORE_PATH}")

    # Загружаем уже построенные индекс и retriever
    # PDF на этом шаге не режем и эмбеддинги заново не считаем
    source = load_source(
        store=STORE_PATH,
        offline=True,
    )
    return source["retriever"]


# Это адаптер над уже существующим поиском, а не новая RAG-система
def search_documents(query: str, k: int | None = None) -> dict:
    """Ищет фрагменты в Apollo 11 Mission Report.

    Параметры поиска (mode, backend, top_k, candidate_k) берутся из
    lab3/config/agent.yaml. Если k не задан, используется retrieval.top_k.
    """
    if not query.strip():
        raise ValueError("Поисковый запрос пуст.")

    settings = get_settings()
    retrieval = settings.retrieval
    if k is None:
        k = retrieval.top_k
    if not 1 <= k <= 10:
        raise ValueError("k должно находиться в диапазоне 1–10.")

    retriever = get_retriever()
    # Здесь вызывается поиск Lab 2 с параметрами из agent.yaml
    # timings сохраняем, чтобы понимать стоимость retrieval-шага
    results, timings = retriever.search(
        query=query,
        k=k,
        mode=retrieval.mode,
        backend=retrieval.backend,
        candidate_k=retrieval.candidate_k,
    )

    return {
        "query": query,
        "store_id": STORE_PATH.name,
        "mode": retrieval.mode,
        "backend": retrieval.backend,
        # Метаданные о типе score нужны для интерпретации ранжирования
        # Нельзя сравнивать числа разных поисковых алгоритмов как одну шкалу
        "score_type": results[0]["score_type"] if results else None,
        "k": k,
        "results": results,
        "timings": timings,
    }
