from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Callable

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS_TO_SPLIT_ON = [('#', 'chapter'), ('##', 'section'), ('###', 'subsection')]
_WS = re.compile(r'\s+')
DEFAULT_SPECIAL_RESERVE = 4


def digest24(value) -> str:
    return hashlib.sha256(value if isinstance(value, bytes)
                          else str(value).encode('utf-8')).hexdigest()[:24]


_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


def _strip_control(text: str) -> str:
    return _CONTROL_RE.sub('', text)


def _squash(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', text)


def heading_context_string(heading_path: list[str]) -> str:
    return ' / '.join(heading_path)


def _make_recursive_splitter(tokenizer, chunk_tokens: int, overlap_tokens: int):
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=chunk_tokens, chunk_overlap=overlap_tokens,
        separators=['\n\n', '\n', ' ', ''], keep_separator=True)


def _chapter_metadata(page: dict) -> dict:
    return {'chapter': f"{page['chapter_id']}. {page['chapter_title']}"}


def _split_page_markdown_header(page: dict, tokenizer, config: dict,
                                heading_state: dict | None = None) -> list[dict]:
    # Заголовки предыдущей страницы или текущей главы
    state = heading_state if heading_state is not None else _chapter_metadata(page)

    # Собираем заголовки в Markdown
    prefix = '\n'.join(
        f'{marker} {state[key]}'
        for marker, key in HEADERS_TO_SPLIT_ON
        if state.get(key)
    )

    # Добавляем их перед текстом страницы
    markdown = prefix + '\n\n' + page['text']

    # Один проход нужен, чтобы запомнить раздел для следующей страницы. Второй — чтобы получить текст для чанков.

    # Первый проход - определяем последний раздел страницы
    with_headers = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
        return_each_line=True
    ).split_text(markdown)

    # Второй проход - выделяем текст разделов, заголовки оставляем в metadata
    pieces = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True
    ).split_text(markdown)

    # Запоминаем последний раздел для следующей страницы
    if with_headers:
        state.clear()
        state.update(with_headers[-1].metadata)

    entries = []
    for doc in pieces:
        text = doc.page_content.strip()

        # Пропускаем пустой текст
        if not text:
            continue

        # Собираем путь: глава → раздел → подраздел
        heading_path = [
            doc.metadata[key]
            for _, key in HEADERS_TO_SPLIT_ON
            if doc.metadata.get(key)
        ]

        # Сохраняем текст и его заголовки
        entries.append({
            'text': text,
            'heading_path': heading_path
        })

    # Разбиение по размеру будет дальше, в build_chunks
    return entries


def iter_page_entries(pages: list[dict], tokenizer, config: dict):
    state = {}
    previous = None
    for page in sorted(pages, key=lambda p: (p['source_id'], p['pdf_page'])):
        identity = (page['source_id'], page['pdf_page'])
        if previous is not None and identity == previous[:2]:
            raise ValueError(f'Повтор страницы: {identity}')
        if (previous is None or page['source_id'] != previous[0]
                or page['pdf_page'] != previous[1] + 1
                or page['chapter_id'] != previous[2]):
            state = _chapter_metadata(page)
        if config['splitter'] == 'markdown_header':
            entries = _split_page_markdown_header(page, tokenizer, config, state)
        else:
            entries = _split_page_recursive(page, tokenizer, config)
        previous = (*identity, page['chapter_id'])
        if page['retrieval_eligible']:
            yield page, entries


def _split_page_recursive(page: dict, tokenizer, config: dict) -> list[dict]:
    text = page['text'].strip()
    return [{'text': text,
             'heading_path': [f"{page['chapter_id']} {page['chapter_title']}"]}]


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, verbose=False)['input_ids'])


def build_chunks(pages: list[dict], tokenizer, config: dict, build_id: str) -> list[dict]:
    # Читаем настройки разбиения
    mode = config['splitter']
    if mode not in ('markdown_header', 'page_recursive'):
        raise ValueError("splitter must be 'markdown_header' or 'page_recursive'")

    include_headings = config['include_headings']
    special_reserve = config.get('special_tokens_reserve', DEFAULT_SPECIAL_RESERVE)
    overlap = config['overlap']
    passage_prefix = config['passage_prefix']

    chunks: list[dict] = []

    # Обрабатываем страницы и выделенные на них текстовые части
    for page, entries in iter_page_entries(pages, tokenizer, config):
        for entry_index, entry in enumerate(entries):
            heading_path = entry['heading_path']

            # Добавляем заголовки, если это включено в настройках
            header = heading_context_string(heading_path) + '\n\n' if include_headings else ''

            # Собираем passage: и заголовки
            prefix = passage_prefix + header

            # Считаем их длину вместе со специальными токенами
            prefix_tokens = len(
                tokenizer(prefix, add_special_tokens=True, verbose=False)['input_ids']
            )

            # Оставляем место для основного текста и вычитаем запас
            budget = config['chunk_size'] - prefix_tokens - special_reserve

            # Бюджет тела должен быть больше перекрытия
            if budget <= overlap:
                raise ValueError('Заголовок съедает бюджет тела. Увеличить chunk_size')

            body = entry['text']
            body_tokens = token_len(tokenizer, body)

            if body_tokens <= budget:
                # Короткий текст оставляем целиком
                pieces = [body]
            else:
                # Длинный текст делим по бюджету тела с перекрытием
                splitter = _make_recursive_splitter(tokenizer, budget, overlap)
                pieces = [p.strip() for p in splitter.split_text(body) if p.strip()]

            # Определяем раздел для полученных фрагментов
            section_id = _last_section_id(heading_path, page)

            for piece_index, piece in enumerate(pieces):
                # Собираем полный текст, который получит E5
                embedding_text = prefix + piece

                # Повторно считаем длину всего входа со спецтокенами
                count = len(
                    tokenizer(
                        embedding_text,
                        add_special_tokens=True,
                        verbose=False
                    )['input_ids']
                )

                # Не допускаем превышения общего размера чанка
                if count > config['chunk_size']:
                    raise ValueError(
                        f'Чанк превышает бюджет: {count} > {config["chunk_size"]} '
                        f'(стр. {page["pdf_page"]}, кусок {piece_index}).'
                    )

                # Сохраняем чанк, его длину и связь с источником
                chunks.append({
                    'chunk_id': digest24((
                        build_id, page['source_id'], page['pdf_page'],
                        entry_index, piece_index
                    )),

                    # Страница и раздел документа
                    'pdf_page': page['pdf_page'],
                    'printed_page': page['printed_page'],
                    'chapter_id': page['chapter_id'],
                    'chapter_title': page['chapter_title'],
                    'section_id': section_id,
                    'heading_path': heading_path,
                    'citation_url': page['citation_url'],

                    # Тело отдельно, полный вход E5 отдельно
                    'text': piece,
                    'embedding_text': embedding_text,
                    'embedding_token_count': count,
                    'body_token_count': token_len(tokenizer, piece),
                    'is_last_piece': piece_index == len(pieces) - 1,

                    # Данные исходного документа
                    'source_id': page['source_id'],
                    'source_title': page['source_title'],
                    'source_sha256': page['source_sha256'],
                    'language': page['language'],
                    'retrieval_eligible': True,
                })

    # Проверяем, что удалось создать хотя бы один чанк
    if not chunks:
        raise ValueError('Нет eligible-контента для чанков.')

    return chunks


def _last_section_id(heading_path: list[str], page: dict) -> str:
    if heading_path:
        match = re.match(r'^([0-9]{1,2}(?:\.[0-9]{1,2}){0,2})\s', heading_path[-1])
        if match:
            return match.group(1)
    return page['chapter_id']


def check_chunks(chunks: list[dict], pages: list[dict], config: dict) -> None:
    by_page = {(p['source_id'], p['pdf_page']): p for p in pages}
    page_words: dict[tuple[str, int], Counter] = {}
    chunk_words: dict[tuple[str, int], Counter] = {}

    seen_ids = set()
    for chunk in chunks:
        if chunk['embedding_token_count'] > config['chunk_size']:
            raise ValueError(f"Oversized chunk: {chunk['chunk_id']}")
        if chunk['chunk_id'] in seen_ids:
            raise ValueError('Дубликат chunk_id')
        seen_ids.add(chunk['chunk_id'])

        page_key = (chunk['source_id'], chunk['pdf_page'])
        page = by_page[page_key]
        if not page['retrieval_eligible']:
            raise ValueError(f"Чанк из исключённой страницы: {chunk['pdf_page']}")
        page_text = _strip_control(page['text'])
        page_squash = _squash(page_text)
        body_squash = _squash('\n'.join(
            line for line in page_text.split('\n') if not line.startswith('#')))
        chunk_squash = _squash(_strip_control(chunk['text']))
        if chunk_squash not in page_squash and chunk_squash not in body_squash:
            raise ValueError(
                f"Текст чанка не найден на своей странице: {chunk['chunk_id']} "
                f"(стр. {chunk['pdf_page']})")

        if page_key not in page_words:
            body_lines = [ln for ln in _strip_control(page['text']).split('\n')
                          if not ln.startswith('#')]
            page_words[page_key] = Counter(_words('\n'.join(body_lines)))
        chunk_words.setdefault(page_key, Counter()).update(
            _words(_strip_control(chunk['text'])))

    for pdf_page, need in page_words.items():
        have = chunk_words.get(pdf_page)
        if have is None:
            raise ValueError(f'Страница {pdf_page} без чанков')
        lost = {w: c for w, c in need.items() if have[w] < c}
        serious = {w: c for w, c in lost.items() if len(w) >= 4}
        if serious:
            raise ValueError(f'Страница {pdf_page}: потеряны слова {sorted(serious)[:10]}')


def _words(text: str) -> list[str]:
    return re.findall(r'[A-Za-z0-9]+', text)
