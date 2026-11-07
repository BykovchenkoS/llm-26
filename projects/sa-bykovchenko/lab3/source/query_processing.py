"""Подготовка запросов и проверка найденных фрагментов для агента Apollo 11.

Документ на английском, а вопросы могут быть на русском. Поэтому перед поиском
создаём несколько формулировок вопроса, затем объединяем и уточняем выдачу.
Оригинальный вопрос всегда сохраняем: дополнительные варианты могут оказаться
неточными, но не должны полностью заменять исходную формулировку.

Здесь нет вызовов LLM: перевод по словарю, RRF и проверка найденного текста
работают по фиксированным правилам. Это делает шаг воспроизводимым, но не
гарантирует смысловую правильность найденных доказательств. Словарь покрывает
не все выражения, а лексический rerank иногда может поставить выше не тот чанк.

"""

from __future__ import annotations

ALGORITHM_VERSION = "query-processing-v1"

import re
from dataclasses import dataclass
from typing import Iterable

# В этом модуле только правила обработки текста, без генерации моделью
# Его задача не отвечать на вопрос, а подготовить и оценить кандидатов для RAG

# Сохраняем в токенах числа и составные обозначения вроде «4.12» или «south-southwest»
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[.\-][A-Za-zА-Яа-яЁё0-9]+)*")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

# Эти выражения убирают просьбы «укажи раздел», «приведи источник» и т.п.
# Сам факт, который нужно найти, по возможности оставляем в вопросе
_META_PATTERNS = [
    re.compile(r"\bв\s+ответе\s+укажи\b.*$", re.IGNORECASE),
    re.compile(r"\bукажи\s+(?:конкретный\s+)?(?:раздел|pdf[\s-]*страниц\w*|источник\w*|ссылк\w*)\b.*$", re.IGNORECASE),
    re.compile(r"\bприведи\s+(?:pdf[\s-]*страниц\w*|раздел|источник\w*|ссылк\w*)\b.*$", re.IGNORECASE),
    re.compile(r"\bсошлись\s+на\s+отч[её]т\b.*$", re.IGNORECASE),
    re.compile(r"\bответь\s+только\s+на\s+основе\b.*$", re.IGNORECASE),
    re.compile(r"\bесли\s+отч[её]т\s+этого\s+не\s+сообщает\b.*$", re.IGNORECASE),
    re.compile(r"\bиспользуй\s+арифметическ\w*\s+инструмент\b.*$", re.IGNORECASE),
    re.compile(r"\bнайди(?:те)?\s+в\s+отч[её]те[, ]*", re.IGNORECASE),
    re.compile(r"\bin\s+your\s+answer[, ]*(?:cite|include|provide|give)\b.*$", re.IGNORECASE),
    re.compile(r"\bcite\s+(?:the\s+)?(?:section|page|source)\b.*$", re.IGNORECASE),
    re.compile(r"\banswer\s+only\s+from\s+the\s+report\b.*$", re.IGNORECASE),
]

_STOPWORDS = {
    # Служебные русские слова обычно не помогают искать тему документа.
    # Некоторые слова времени тоже исключаются здесь, но отдельно проверяются по исходному вопросу в assess_evidence().
    "и", "в", "во", "на", "с", "со", "по", "к", "ко", "о", "об", "от", "до",
    "после", "для", "из", "у", "за", "при", "про", "что", "как", "какой",
    "какая", "какие", "каких", "каков", "какова", "где", "когда", "почему",
    "зачем", "это", "этот", "эта", "эти", "тот", "та", "те", "но", "или",
    "а", "же", "ли", "бы", "чтобы", "если", "то", "так", "также", "только",
    "ещё", "еще", "уже", "был", "была", "были", "быть", "есть", "может",
    "можно", "нужно", "надо", "отчёт", "отчет", "документ", "страница",
    "раздел", "pdf", "информация", "информации", "вопрос", "ответ", "укажи",
    "приведи", "сообщает", "сообщить", "согласно", "основе", "прямо", "скажи",
    "назови", "перечисли", "кратко", "ответь", "найди", "найдите", "между",
    "членами", "членов", "была", "было", "все", "весь", "вся", "них", "него",
    "неё", "нее", "свой", "свои", "своего", "который", "которая", "которые",
    "которых", "конкретный", "конкретно", "оба", "обе", "этом", "этого",
    "данном", "данного", "сообщает", "отсутствии", "данных",
    # Английские служебные слова. Отрицания дополнительно проверяем отдельно
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "with", "from", "by",
    "at", "as", "that", "this", "these", "those", "it", "its", "what",
    "which", "who", "whom", "whose", "when", "where", "why", "how", "did",
    "do", "does", "done", "has", "have", "had", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must", "nor", "if", "then",
    "than", "so", "such", "into", "about", "over", "under", "between",
    "during", "according", "report", "document", "page", "section", "pdf",
    "answer", "provide", "cite", "tell", "list", "briefly", "exactly",
    "only", "based", "used", "use", "using", "does", "mention", "mentioned",
}

# «До» и «после» способны менять смысл ответа. Проверим их отдельно
# при оценке найденных фрагментов, даже если это не ключевые слова темы
_TEMPORAL_GROUPS = {
    "before": {"before", "prior", "earlier", "до", "перед", "раньше"},
    "after": {"after", "later", "после", "позже"},
    "first": {"first", "initial", "первый", "первого", "первым"},
    "last": {"last", "final", "последний", "последнего"},
}

_NEGATION_TERMS = {
    "not", "never", "without", "no", "none", "neither", "nor",
    "не", "никогда", "без", "нет", "ни", "никакой", "никаких",
}

_META_TERMS = {
    "calculate", "calculation", "find", "cite", "provide", "list", "name",
    "explain", "describe", "answer", "report", "mention", "include", "give",
    "tell", "briefly", "вычисли", "вычислить", "рассчитай", "найди", "укажи",
    "приведи", "перечисли", "назови", "объясни", "опиши", "ответь", "кратко",
}

# Словарь терминов Apollo 11 для дополнительного англоязычного запроса
# Это не полноценный переводчик, он заменяет известные слова и сочетания
# Исходный вопрос всё равно остаётся отдельным вариантом поиска
_RU_EN_PHRASES = {
    "лунная поверхность": "lunar surface",
    "выход на поверхность": "extravehicular activity surface",
    "внекорабельная деятельность": "extravehicular activity",
    "точка приводнения": "splashdown point",
    "точку приводнения": "splashdown point",
    "граница входа": "entry interface",
    "границы входа": "entry interface",
    "вход в атмосферу": "entry interface",
    "входа в атмосферу": "entry interface",
    "лунная пыль": "lunar dust",
    "лунные образцы": "lunar samples",
    "лунный модуль": "lunar module",
    "командный модуль": "command module",
    "пилот командного модуля": "command module pilot",
    "пилот лунного модуля": "lunar module pilot",
    "лазерный отражатель": "laser reflector",
    "лазерного отражателя": "laser reflector",
    "сейсмический эксперимент": "seismic experiment",
    "солнечный ветер": "solar wind",
    "научные цели": "scientific objectives",
    "научных целей": "scientific objectives",
    "интенсивность энергозатрат": "energy expenditure rate",
    "энергозатраты": "energy consumption",
    "телевизионный кабель": "television cable",
    "во время": "during",
    "согласно": "according to",
    "отчет": "report",
    "отчета": "report",
    "отчёте": "report",
    "отчёт": "report",
    "документ": "document",
    "страница": "page",
    "страницы": "page",
    "раздел": "section",
    "раздела": "section",
    "источник": "source",
    "источники": "sources",
    "миссии": "mission",
    "миссия": "mission",
    "программы": "program",
    "программа": "program",
    "полета": "flight",
    "полет": "flight",
    "посадки": "landing",
    "посадка": "landing",
    "приводнения": "splashdown",
    "приводнение": "splashdown",
    "расстояния": "distance",
    "расстояние": "distance",
    "разницу": "difference",
    "разница": "difference",
    "разность": "difference",
    "милях": "miles",
    "миль": "miles",
    "мили": "miles",
    "перенесли": "moved",
    "перенес": "moved",
    "изменили": "changed",
    "района": "area",
    "район": "area",
    "точку": "point",
    "точка": "point",
    "атмосферу": "atmosphere",
    "атмосфера": "atmosphere",
    "граница": "boundary",
    "границы": "boundary",
    "входа": "entry",
    "вход": "entry",
    "смартфона": "smartphone",
    "смартфон": "smartphone",
    "модели": "model",
    "модель": "model",
    "использовала": "used",
    "использовал": "used",
    "армстронг": "Armstrong",
    "нил": "Neil",
    "фольги": "foil",
    "фольгу": "foil",
    "фольга": "foil",
    "экспонирования": "exposure",
    "экспонирование": "exposure",
    "экспозиция": "exposure",
    "отражатель": "reflector",
    "точность": "accuracy",
    "измерения": "measurements",
    "гравитация": "gravity",
    "карантин": "quarantine",
    "образцов": "samples",
    "образцы": "samples",
    "очищали": "cleaned",
    "очистка": "cleaning",
    "предполетная": "preflight",
    "послеполетная": "postflight",
    "оцененная": "estimated",
    "процентов": "percent",
    "процент": "percent",
    "вычислить": "calculate",
    "вычисли": "calculate",
    "рассчитай": "calculate",
    "температура": "temperature",
    "частицы": "particles",
    "эксперименты": "experiments",
    "эксперимент": "experiment",
    "астронавты": "astronauts",
    "астронавт": "astronaut",
    "экипажа": "crew",
    "экипаж": "crew",
    "ролей": "roles",
    "роли": "roles",
    "роль": "role",
    "командир": "commander",
    "цели": "objectives",
    "цель": "objective",
    "поверхности": "surface",
    "поверхность": "surface",
    "выхода": "egress",
    "выход": "egress",
    "отдыха": "rest",
    "отдых": "rest",
    "сон": "sleep",
    "подготовку": "preparation",
    "подготовки": "preparation",
    "подготовка": "preparation",
    "заняла": "took",
    "заняло": "took",
    "задержки": "delays",
    "задержка": "delay",
    "причины": "reasons",
    "причина": "reason",
    "факторы": "factors",
    "мешали": "prevented",
    "мешал": "prevented",
    "проблемы": "problems",
    "проблема": "problem",
    "пыль": "dust",
    "загрязнение": "contamination",
    "оборудование": "equipment",
    "кабель": "cable",
    "проблем": "problems",
    "начал": "began",
    "начала": "began",
    "почему": "why",
    "зачем": "why",
    "какую": "which",
    "какие": "which",
    "каких": "which",
    "какова": "what",
    "каков": "what",
    "какой": "which",
    "какая": "which",
    "насколько": "how much",
    "сколько": "how many",
    "когда": "when",
    "где": "where",
    "как": "how",
    "что": "what",
    "кто": "who",
    "чем": "than",
    "перечисли": "list",
    "кратко": "briefly",
    "назови": "name",
    "опиши": "describe",
    "объясни": "explain",
    "найди": "find",
    "приведи": "provide",
    "укажи": "cite",
    "сообщает": "reports",
    "до": "before",
    "после": "after",
    "перед": "before",
    "раньше": "earlier",
    "позже": "later",
    "первого": "first",
    "первым": "first",
    "первый": "first",
    "запланированного": "planned",
    "запланированный": "planned",
    "решили": "decided",
    "решил": "decided",
    "провести": "conduct",
    "провели": "conducted",
    "и": "and",
    "или": "or",
    "не": "not",
    "без": "without",
    "на": "on",
    "во": "in",
    "в": "in",
    "со": "from",
    "с": "from",
    "от": "from",
    "ко": "to",
    "к": "to",
    "по": "on",
    "для": "for",
    "основная": "primary",
    "главная": "main",
    "распределение": "distribution",
    "распределялись": "distributed",
    "членами": "members",
    "членов": "members",
    "члены": "members",
    "больше": "more",
    "времени": "time",
    "ожидалось": "expected",
    "при": "during",
    "между": "between",
    "посадке": "landing",
    "посадку": "landing",
    "отказался": "refused",
    "отказались": "refused",
    "отказалась": "refused",
    "первоначальной": "original",
    "первоначальную": "original",
    "площадки": "landing site",
    "площадку": "landing site",
    "площадка": "landing site",
    "ручному": "manual",
    "ручное": "manual",
    "управлению": "control",
    "управление": "control",
    "перешел": "switched",
    "перешли": "switched",
    "электрический разъем": "electrical connector",
    "электрического разъема": "electrical connector",
    "задержка": "delay",
    "задержала": "delayed",
    "задержало": "delayed",
    "лунной пыли": "lunar dust",
    "пыли": "dust",
    "холод": "cold",
    "шум": "noise",
    "освещение": "lighting",
    "насосы": "pumps",
    "насосов": "pumps",
    "выспаться": "sleep",
    "сейсмометры": "seismometers",
    "сейсмический": "seismic",
    "лазерное измерение": "laser ranging",
    "лазерного": "laser",
    "относительность": "relativity",
    "относительности": "relativity",
    "сантиметров": "centimeters",
    "сантиметра": "centimeters",
    "метров": "meters",
    "метрах": "meters",
    "юго-юго-запад": "south-southwest",
    "предполетные": "preflight",
    "послеполетные": "postflight",
    "оцененные": "estimated",
    "биосфера": "biosphere",
    "биосферу": "biosphere",
    "инфекционные": "infectious",
    "кабины": "cabins",
    "кабину": "cabin",
    "пылесос": "vacuum",
    "фильтры": "filters",
    "мешки": "bags",
    "перегрузка": "overload",
    "компьютера": "computer",
    "сигналы": "alarms",
    "наведение": "guidance",
    "радар": "radar",
    "сближение": "rendezvous",
    "площадку": "landing site",
    "посадочную": "landing",
    "посадочной": "landing",
}

_SYNONYMS = {
    # Близкие английские слова и формы, которые могут встретиться в отчёте
    "eva": ("extravehicular", "egress", "moonwalk", "surface"),
    "egress": ("extravehicular", "exit", "surface"),
    "extravehicular": ("eva", "egress", "surface"),
    "splashdown": ("splashdown", "landing", "recovery", "splash"),
    "landing": ("landing", "splashdown", "touchdown"),
    "recovery": ("recovery", "splashdown", "landing"),
    "smartphone": ("smartphone", "phone", "mobile", "telephone"),
    "phone": ("phone", "telephone", "smartphone"),
    "objective": ("objective", "purpose", "mission", "goal"),
    "purpose": ("purpose", "objective", "mission", "goal"),
    "primary": ("primary", "main", "purpose", "objective"),
    "role": ("role", "position", "assignment", "function", "commander", "pilot"),
    "roles": ("role", "position", "assignment", "function", "commander", "pilot"),
    "crew": ("crew", "astronaut", "commander", "pilot"),
    "astronaut": ("astronaut", "crew", "commander", "pilot"),
    "moved": ("move", "moved", "relocate", "change", "shift"),
    "move": ("move", "moved", "relocate", "change", "shift"),
    "distance": ("distance", "range", "miles", "mile"),
    "range": ("range", "distance", "miles", "mile"),
    "entry": ("entry", "interface", "boundary"),
    "interface": ("interface", "entry", "boundary"),
    "boundary": ("boundary", "interface", "entry"),
    "rest": ("rest", "sleep", "rested"),
    "longer": ("longer", "more", "exceeded", "extra", "optimistic"),
    "expected": ("expected", "anticipated", "estimated", "planned", "optimistic"),
    "preparation": ("preparation", "preparing", "preparations"),
    "took": ("took", "taken", "required", "exceeded"),
    "began": ("began", "begin", "beginning", "started"),
    "conduct": ("conduct", "conducted", "performed", "began"),
    "decided": ("decided", "thought", "chose", "considered", "begin"),
    "sleep": ("sleep", "rest", "rested"),
    "foil": ("foil", "film", "sheet"),
    "exposure": ("exposure", "exposed", "exposition"),
    "laser": ("laser", "reflector", "retroreflector"),
    "reflector": ("reflector", "retroreflector", "laser"),
    "dust": ("dust", "lunar", "soil"),
    "pressure": ("pressure", "depressurize", "depressurization"),
    # Русские термины используем как запасной вариант расширения.
    # В основном сценарии английский query может сформировать сам агент.
    "выход": ("egress", "extravehicular", "exit", "surface"),
    "поверхность": ("surface", "lunar"),
    "отдых": ("rest", "sleep"),
    "отдыха": ("rest", "sleep"),
    "цель": ("objective", "purpose", "goal"),
    "экипаж": ("crew", "astronaut"),
    "экипажа": ("crew", "astronaut"),
    "роли": ("role", "position", "assignment"),
    "роль": ("role", "position", "assignment"),
    "приводнение": ("splashdown", "landing", "recovery"),
    "приводнения": ("splashdown", "landing", "recovery"),
    "расстояние": ("distance", "range", "miles"),
    "расстояния": ("distance", "range", "miles"),
    "граница": ("boundary", "interface"),
    "границы": ("boundary", "interface"),
    "вход": ("entry", "interface"),
    "входа": ("entry", "interface"),
    "атмосфера": ("atmosphere", "entry"),
    "атмосферу": ("atmosphere", "entry"),
    "фольга": ("foil", "film"),
    "фольги": ("foil", "film"),
    "экспонирование": ("exposure", "exposed"),
    "экспонирования": ("exposure", "exposed"),
    "смартфон": ("smartphone", "phone", "mobile"),
    "смартфона": ("smartphone", "phone", "mobile"),
    "пыль": ("dust", "lunar"),
    "давление": ("pressure", "depressurization"),
    "образцы": ("sample", "samples"),
    "образцов": ("sample", "samples"),
    "карантин": ("quarantine", "isolation"),
    "энергозатраты": ("energy", "btu", "consumption"),
    "фольгу": ("foil", "film"),
    "первого": ("first", "initial"),
    "первый": ("first", "initial"),
    "запланированного": ("planned", "scheduled"),
    "запланированный": ("planned", "scheduled"),
    "решили": ("decided", "chose", "thought", "considered", "begin"),
    "решил": ("decided", "chose", "thought", "considered", "begin"),
    "провести": ("conduct", "perform", "begin", "carry"),
    "провели": ("conducted", "performed", "began", "carried"),
}


# Небольшой набор терминов для расширения английского запроса.
# Добавляем не любые синонимы, а только заранее выбранные слова по теме.
_EXPANSION_HINTS = {
    "roles": ("commander", "pilot"),
    "role": ("commander", "pilot"),
    "crew": ("commander", "pilot"),
    "astronaut": ("crew", "commander", "pilot"),
    "splashdown": ("landing", "recovery"),
    "landing": ("splashdown", "recovery"),
    "entry": ("interface", "boundary"),
    "boundary": ("interface",),
    "distance": ("range",),
    "rest": ("sleep",),
    "foil": ("film",),
    "reflector": ("retroreflector",),
    "exposure": ("exposed",),
    "dust": ("lunar",),
    "quarantine": ("isolation",),
}


# Имя варианта нужно, чтобы после поиска понять, какая формулировка привела к конкретному чанку
# frozen=True защищает объект от случайной правки
@dataclass(frozen=True)
class QueryVariant:
    name: str
    text: str


# Приводим текст к нижнему регистру и выделяем слова регулярным выражением
def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


# Убираем несколько распространённых английских окончаний для грубого сопоставления словоформ
def _stem(token: str) -> str:
    if len(token) > 4:
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
    return token


def _has_cyrillic(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text or ""))


def has_cyrillic(text: str) -> bool:
    return _has_cyrillic(text)


def normalize_question(question: str) -> str:
    """Убирает инструкции по оформлению ответа, оставляя вопрос для поиска"""
    text = (question or "").strip()
    # Каждый шаблон вырезает инструкцию к ответу
    # пробелы после этого нормализуем, чтобы не передавать в поиск дырявую строку
    for pattern in _META_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n,;:.-")
    return text


def translate_ru_to_en(text: str) -> str:
    """Составляет англоязычный запрос по словарю терминов Apollo.

    Сначала заменяются длинные фразы, затем отдельные слова. Так сочетания
    вроде «первого запланированного отдыха» не разбиваются раньше времени.
    Полного машинного перевода здесь нет.
    """
    # Для английского ввода отдельный словарный перевод не требуется
    if not _has_cyrillic(text or ""):
        return ""
    result = normalize_question(text)
    # Сортировка по длине защищает устойчивые фразы от преждевременной замены их отдельных слов
    for phrase in sorted(_RU_EN_PHRASES, key=len, reverse=True):
        replacement = _RU_EN_PHRASES[phrase]
        result = re.sub(
            rf"(?<!\w){re.escape(phrase)}(?!\w)",
            f" {replacement} ",
            result,
            flags=re.IGNORECASE,
        )
    # Незнакомые русские слова убираем
    # этот вариант нужен только как дополнительная английская подсказка, а не замена исходного вопроса
    tokens = [token for token in _tokenize(result) if not _has_cyrillic(token)]
    return " ".join(content_terms(" ".join(tokens)))


def expand_english_query(text: str, max_extra: int = 6) -> str:
    """Добавляет к английскому запросу несколько тематически близких терминов."""
    terms = content_terms(text)
    extras: list[str] = []
    # Синонимы добавляем один раз
    # повторы не дают новой информации, а слишком длинный запрос может ухудшить поиск
    for term in terms:
        for synonym in _EXPANSION_HINTS.get(term, ()):
            if synonym not in terms and synonym not in extras:
                extras.append(synonym)
    if not extras:
        return " ".join(terms)
    return " ".join(terms + extras[:max_extra])


def content_terms(text: str) -> list[str]:
    """Выделяет значимые слова без повторов для поиска и проверки покрытия."""
    terms: list[str] = []
    seen: set[str] = set()
    # Стоп-слова и слишком короткие элементы отбрасываются, числа сохраняются
    # Важные смысловые условия проверяются отдельно в assess_evidence()
    for token in _tokenize(text):
        if token in _STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


def _expand_term(term: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((term, *_SYNONYMS.get(term, ()))))


def build_query_variants(
    question: str,
    query: str | None = None,
    *,
    max_variants: int = 6,
) -> list[QueryVariant]:
    """Собирает несколько вариантов вопроса без дополнительных вызовов LLM.

    Первым всегда идёт оригинал. Остальные варианты добавляются только если
    они не пустые и отличаются от уже добавленных. Число вариантов ограничено.
    """
    original = (question or "").strip()
    if not original:
        raise ValueError("Вопрос не может быть пустым.")

    normalized = normalize_question(original)
    # Исходная формулировка идёт первой - даже если словарь или сама модель
    # предложили неудачный альтернативный запрос
    variants: list[QueryVariant] = [QueryVariant("original", original)]

    def add(name: str, text: str) -> None:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not cleaned:
            return
        # Не повторяем одинаковые запросы
        # иначе они получили бы неоправданно большой вес при последующем объединении по RRF
        key = cleaned.casefold()
        if any(item.text.casefold() == key for item in variants):
            return
        variants.append(QueryVariant(name, cleaned))

    if normalized and normalized.casefold() != original.casefold():
        add("normalized", normalized)

    # Словарный перевод используем только при наличии хотя бы двух значимых английских слов
    # слишком короткая подсказка обычно бесполезна
    glossary_query = translate_ru_to_en(original)
    if len(content_terms(glossary_query)) >= 2:
        add("glossary_en", glossary_query)
        add("expanded_en", expand_english_query(glossary_query))

    # Это запрос, который составила сама Qwen при вызове инструмента
    # Он полезен, потому что модель может учесть смысл всей формулировки
    if query and query.strip():
        add("agent_query", query.strip())

    # Последний резервный вариант - только значимые слова вопроса
    # Он может быть менее точным, поэтому оригинал мы не удаляем
    keyword_source = normalized or original
    keywords = " ".join(content_terms(keyword_source))
    if keywords:
        add("keywords", keywords)

    # Не делаем слишком много отдельных RAG-поисков для одного вопроса
    return variants[:max_variants]


def fuse_results(
    result_sets: Iterable[tuple[str, list[dict]]],
    *,
    k: int = 5,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Объединяет выдачи разных запросов методом взвешенного RRF."""
    weights = weights or {}
    merged: dict[str, dict] = {}
    # Каждый вариант поиска возвращает ранжированный список словарей.
    # Собираем одинаковые чанки по chunk_id, сохраняя их метаданные.
    for variant_name, results in result_sets:
        for item in results:
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue
            rank = int(item.get("rank", 0) or 0)
            if rank <= 0:
                continue
            entry = merged.get(chunk_id)
            if entry is None:
                # Копируем запись, чтобы не перезаписывать результат отдельного поискового прохода
                entry = dict(item)
                entry["variant_ranks"] = {}
                entry["fused_score"] = 0.0
                merged[chunk_id] = entry
            entry["variant_ranks"][variant_name] = rank
            # RRF складывает вклады от позиций в выдаче
            # Чем выше чанки чем больше разных запросов его нашли, тем больше общий балл
            # Здесь не смешиваются напрямую несопоставимые BM25/FAISS scores
            entry["fused_score"] += weights.get(variant_name, 1.0) / (rrf_k + rank)

    # Отбираем первые k после объединения
    # при равной оценке используем позицию чанка в исходной выдаче.
    ordered = sorted(
        merged.values(),
        key=lambda item: (-item["fused_score"], item.get("rank", 0)),
    )[:k]
    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank
        item["score"] = round(item["fused_score"], 6)
        item["score_type"] = "multi_rrf"
    return ordered


def _term_coverage(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    tokens = set(_tokenize(text))
    stems = {_stem(token) for token in tokens}
    # Считаем долю терминов запроса, найденных в чанке
    # Учитываем некоторые словоформы и заранее заданные синонимы
    matched = 0
    for term in terms:
        if any(
            _stem(candidate) in stems or _stem(candidate) in tokens
            for candidate in _expand_term(term)
        ):
            matched += 1
    return matched / len(terms)


def rerank_by_coverage(
    results: list[dict],
    english_query: str | None,
    *,
    original_results: list[dict] | None = None,
    anchor_threshold: float = 0.3,
    anchor_margin: float = 0.3,
) -> list[dict]:
    terms = content_terms(english_query or "")
    if not terms or not results:
        return results

    candidates = list(results)
    covered_ids = {item.get("chunk_id") for item in candidates}
    # Если сильный результат исходного запроса пропал после RRF,
    # возвращаем его в пул кандидатов перед перестановкой
    if original_results:
        top_original = original_results[0]
        if top_original.get("chunk_id") not in covered_ids:
            candidate = dict(top_original)
            candidate["variant_ranks"] = {"original": top_original.get("rank", 1)}
            candidate["fused_score"] = 0.0
            candidates.append(candidate)

    for item in candidates:
        item["evidence_coverage"] = round(
            _term_coverage(item.get("text", ""), terms),
            3,
        )

    anchored_id = None
    if original_results:
        top_original = original_results[0]
        top_candidate = next(
            (item for item in candidates if item.get("chunk_id") == top_original.get("chunk_id")),
            None,
        )
        best_coverage = max(
            (item.get("evidence_coverage", 0.0) for item in candidates),
            default=0.0,
        )
        # Закрепляем исходный top-1, только если он не сильно хуже лучшего кандидата по покрытию и текст не слишком зашумлен
        effective_threshold = max(
            anchor_threshold,
            best_coverage - anchor_margin,
        )
        if (
            top_candidate is not None
            and top_candidate.get("evidence_coverage", 0.0) >= effective_threshold
            and _alnum_ratio(top_candidate.get("text", "")) >= 0.55
        ):
            anchored_id = top_candidate.get("chunk_id")

    # При закреплённом первом чанке сортируем только остальные
    # Иначе полностью переставляем выдачу по покрытию терминов
    if anchored_id is not None:
        anchor = next(item for item in candidates if item.get("chunk_id") == anchored_id)
        rest = [item for item in candidates if item.get("chunk_id") != anchored_id]
        rest.sort(
            key=lambda item: (
                -item.get("evidence_coverage", 0.0),
                -item.get("fused_score", 0.0),
            )
        )
        ordered = [anchor, *rest]
    else:
        ordered = sorted(
            candidates,
            key=lambda item: (
                -item.get("evidence_coverage", 0.0),
                -item.get("fused_score", 0.0),
            ),
        )

    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank
    return ordered


# Признаки вычислений и составных вопросов. Это подсказки для агента
_CALCULATION_CUES = (
    "вычисл", "разниц", "насколько", "процент", "во сколько", "больше", "меньше",
    "сумм", "отношени", "calculate", "difference", "percent", "ratio",
    "how many", "how much",
)

_MULTI_PART_CUES = (
    "и почему", "и как", "и каков", "обе", "оба", "both", "two parts",
    "first", "second", "and why", "and how", "и вычисли",
)


# Если вопрос содержит несколько требований, для ответа может понадобиться больше одного фрагмента или дополнительное расширение контекста
def is_multi_part(question: str) -> bool:
    lower = (question or "").lower()
    if any(cue in lower for cue in _MULTI_PART_CUES):
        return True
    return lower.count("?") >= 2


def looks_like_calculation(question: str) -> bool:
    lower = (question or "").lower()
    return any(cue in lower for cue in _CALCULATION_CUES)


# Ищем в найденном тексте числовые значения вместе с единицами
# Например, «77 minutes» или «14 meters», но не произвольные факты
_UNIT_PATTERN = re.compile(
    r"(\d[\d\s.,]*)\s*"
    r"(nautical\s+miles|miles|mile|feet|foot|ft|minutes|minute|min|"
    r"seconds|second|sec|percent|%|btu/hr|btu/?|btu|degrees|degree|"
    r"centimeters|centimeter|cm|meters|meter|m)\b",
    re.IGNORECASE,
)


def extract_unit_values(results: list[dict], limit: int = 15) -> list[dict]:
    values: list[dict] = []
    seen: set[tuple[str, str]] = set()
    # Берем только три первых чанка
    # так не накапливаем слишком много случайных чисел из менее релевантных частей отчета
    for item in results[:3]:
        for match in _UNIT_PATTERN.finditer(item.get("text", "")):
            value = re.sub(r"\s+", " ", match.group(1)).strip(" .,")
            unit = match.group(2).lower()
            key = (value, unit)
            if value and key not in seen:
                seen.add(key)
                values.append({"value": value, "unit": unit})
                if len(values) >= limit:
                    return values
    return values


def extract_numeric_values(results: list[dict], limit: int = 20) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    # Числа без единиц сохраняются отдельно для диагностики
    # Мы не делаем вывод, что каждое найденное число относится к вопросу
    pattern = re.compile(r"\d[\d\s.,]*\d|\d")
    for item in results[:3]:
        for match in pattern.finditer(item.get("text", "")):
            value = re.sub(r"\s+", " ", match.group(0)).strip(" .,")
            if value and value not in seen:
                seen.add(value)
                values.append(value)
                if len(values) >= limit:
                    return values
    return values


_UNIT_HINTS = {
    "miles": ("мил", "mile"),
    "feet": ("фут", "feet", "foot", "ft"),
    "minutes": ("минут", "minute"),
    "percent": ("процент", "percent"),
    "btu": ("btu",),
    "centimeters": ("сантиметр", "centimeter"),
    "meters": ("метр", "meter"),
    "seconds": ("секунд", "second"),
    "degrees": ("градус", "degree"),
}


def requested_unit(question: str) -> str | None:
    lower = (question or "").lower()
    for unit, aliases in _UNIT_HINTS.items():
        if any(alias in lower for alias in aliases):
            return unit
    return None


def requested_operation(question: str) -> str | None:
    lower = (question or "").lower()
    if "процент" in lower or "percent" in lower:
        return "percentage"
    if "во сколько" in lower or "ratio" in lower or "отношени" in lower:
        return "ratio"
    if "сумм" in lower or "sum" in lower:
        return "sum"
    if (
        "разниц" in lower
        or "difference" in lower
        or "насколько" in lower
        or "больше" in lower
        or "меньше" in lower
    ):
        return "difference"
    return None


def build_calculation_hint(
    question: str,
    unit_values: list[dict],
) -> str | None:
    # Подсказку строим, только если в вопросе явно распознано
    # вычисление и в выдаче нашлись числовые значения
    operation = requested_operation(question)
    if not operation or not unit_values:
        return None
    unit = requested_unit(question)
    values = [
        item
        for item in unit_values
        if unit is None
        or item.get("unit") == unit
        or item.get("unit", "").startswith(unit)
    ]
    if not values:
        # Если единицы не совпали, оставляем найденные значения,
        # но их релевантность всё равно должен проверить агент
        values = unit_values
    chosen_unit = unit or values[0].get("unit", "matching")
    # Здесь создаётся текстовая подсказка для LLM, а не вычисление
    # Первые два числа могут оказаться посторонними - это ограничение подхода
    if operation == "difference" and len(values) >= 2:
        first, second = values[0], values[1]
        return (
            f"Use calculate with the two {chosen_unit} values "
            f"({first['value']} and {second['value']}): larger - smaller. "
            "Do not compute a percentage unless the question asks for one."
        )
    if operation == "percentage" and len(values) >= 2:
        return (
            "Use calculate as (new - old) / old * 100 with the two "
            f"{chosen_unit} values."
        )
    if operation == "ratio" and len(values) >= 2:
        return f"Use calculate with the two {chosen_unit} values as a ratio."
    if operation == "sum":
        return f"Use calculate to sum the {chosen_unit} values."
    return None


# Доля букв и цифр в тексте - грубый индикатор OCR-шума
# Очень низкое значение может указывать на испорченный PDF-фрагмент
def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    alnum = sum(1 for char in text if char.isalnum())
    return alnum / len(text)


def _corpus_tokens(results: list[dict]) -> set[str]:
    tokens: set[str] = set()
    for item in results:
        for token in _tokenize(item.get("text", "")):
            tokens.add(token)
            tokens.add(_stem(token))
    return tokens


def assess_evidence(
    question: str,
    query: str | None,
    results: list[dict],
    *,
    coverage_threshold: float = 0.75,
) -> dict:
    """Оценивает, похожи ли найденные чанки на достаточные доказательства.

    Проверяем совпадение ключевых слов, признаки шума в тексте, временные
    условия и отрицания. Результат нужен как подсказка для дальнейшего поиска:
    высокая оценка ещё не доказывает, что чанк правильно отвечает на вопрос.
    """
    normalized = normalize_question(question)
    question_terms = content_terms(normalized)
    # Если агент передал английский query, используем его для проверки покрытия
    # иначе берём словарный английский вариант или исходные слова.
    english_query = (
        query
        if query and not _has_cyrillic(query)
        else translate_ru_to_en(normalized)
    )
    query_terms = content_terms(english_query or "")
    focus_terms = query_terms or question_terms

    # При пустой выдаче доказательств нет
    # Возвращаем тот же набор полей, чтобы вызывающему коду не приходилось отдельно их угадывать
    if not results:
        return {
            "heuristic_sufficient": False,
            "coverage": 0.0,
            "missing_terms": focus_terms,
            "matched_terms": [],
            "ocr_noise": True,
            "temporal_required": [],
            "temporal_found": [],
            "negation_required": [],
            "negation_found": [],
            "reason": "no_results",
            "critical_missing_terms": focus_terms,
            "numeric_values": [],
            "unit_values": [],
            "calculation_requested": looks_like_calculation(question),
            "calculation_hint": None,
            "multi_part": is_multi_part(question),
            "note": "Heuristic lexical check; not a semantic entailment judge.",
        }

    # Сравниваем значимые слова вопроса с токенами найденных чанков
    # Синонимы помогают найти разные формулировки одной темы, но способны давать ложные совпадения
    corpus_tokens = _corpus_tokens(results)
    matched: list[str] = []
    missing: list[str] = []
    for term in focus_terms:
        expanded = _expand_term(term)
        if any(_stem(candidate) in corpus_tokens for candidate in expanded):
            matched.append(term)
        else:
            missing.append(term)

    # Coverage - доля совпавших терминов

    # Отдельно отмечаем длинные отсутствующие слова, которые могут менять смысл вопроса
    coverage = len(matched) / len(focus_terms) if focus_terms else 0.0
    critical_missing = [
        term
        for term in missing
        if len(term) >= 6 and term not in _META_TERMS
    ]

    # Проверка OCR приблизительная
    # просто оцениваем долю букв и цифр в первых трёх текстах, а не распознаём ошибки по существу
    top_texts = [item.get("text", "") for item in results[:3]]
    avg_alnum = sum(_alnum_ratio(text) for text in top_texts) / max(len(top_texts), 1)
    ocr_noise = avg_alnum < 0.55

    # Для «до/после» и отрицаний нельзя ориентироваться только на
    # совпадение основных существительных, смысл вопроса будет другим.
    source_tokens = set(_tokenize(f"{normalized} {english_query or query or ''}"))
    result_tokens = set(_tokenize(" ".join(top_texts)))
    temporal_required = sorted(
        group
        for group, terms in _TEMPORAL_GROUPS.items()
        if terms & source_tokens
    )
    temporal_found = sorted(
        group
        for group in temporal_required
        if _TEMPORAL_GROUPS[group] & result_tokens
    )
    temporal_missing = sorted(set(temporal_required) - set(temporal_found))
    negation_required = sorted(term for term in _NEGATION_TERMS if term in source_tokens)
    negation_found = sorted(term for term in negation_required if term in result_tokens)

    # Собираем причины сомнения в найденных фрагментах
    # Любая причина делает heuristic_sufficient=False, но даже True
    # не является доказательством достоверности итогового ответа
    reasons: list[str] = []
    if ocr_noise:
        reasons.append("ocr_noise")
    if coverage < coverage_threshold:
        reasons.append("low_term_coverage")
    if critical_missing:
        reasons.append("missing_focus_term")
    if temporal_missing:
        reasons.append("missing_temporal_constraint")
    if negation_required and not negation_found:
        reasons.append("missing_negation_constraint")

    unit_values = extract_unit_values(results)
    # Передаём агенту и оценку покрытия, и конкретные недостающие слова,
    # и извлечённые числа - это помогает решить, уточнять ли поиск
    return {
        "heuristic_sufficient": not reasons,
        "coverage": round(coverage, 3),
        "missing_terms": missing,
        "matched_terms": matched,
        "ocr_noise": ocr_noise,
        "temporal_required": temporal_required,
        "temporal_found": temporal_found,
        "temporal_missing": temporal_missing,
        "negation_required": negation_required,
        "negation_found": negation_found,
        "critical_missing_terms": critical_missing,
        "numeric_values": extract_numeric_values(results),
        "unit_values": unit_values,
        "calculation_requested": looks_like_calculation(question),
        "calculation_hint": build_calculation_hint(question, unit_values),
        "multi_part": is_multi_part(question),
        "reason": ",".join(reasons) if reasons else "ok",
        "note": "Heuristic lexical check; not a semantic entailment judge.",
    }
