from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf
import pymupdf4llm

PARSER_VERSION = '3.2.0'


# ---------------------------------------------------------------- утилиты

def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- нормализация

_SOFT_HYPHEN = chr(0xAD)  # мягкий перенос из OCR-слоя
_TAG_SUP = re.compile(r'</?sup>')
_TAG_BR = re.compile(r'<br\s*/?>', re.IGNORECASE)
_TAG_U = re.compile(r'</?u>')


def normalize_page_text(text: str) -> str:
    text = _TAG_SUP.sub('', text)
    text = _TAG_BR.sub(' ', text)
    text = _TAG_U.sub('', text)
    text = text.replace('**', '').replace('~~', '')
    # мягкий перенос на границе строки склеивается вместе с переносом строки
    text = re.sub(_SOFT_HYPHEN + r'\s*\n\s*', '', text)
    text = text.replace(_SOFT_HYPHEN, '')
    text = re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[a-z])', '-', text)
    lines = [re.sub(r'[ \t]+', ' ', line).rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip('\n')


# ---------------------------------------------------------------- заголовки разделов

# Номер раздела отчёта: «4.10.3», «11.6» — с OCR-пробелами вокруг точек или без.
_SECTION_RE = re.compile(
    r'^([0-9]{1,2})\s*\.\s*([0-9]{1,2})(?:\s*\.\s*([0-9]{1,2}))?\s+([A-Z].{2,})$')


def chapter_at(pdf_page: int, chapters: list) -> tuple[str, str]:
    result = ('FRONT', 'FRONT MATTER')
    for start, identifier, title in chapters:
        if start > pdf_page:
            break
        result = (str(identifier), str(title))
    return result


def inject_section_headers(text: str, chapter_id: str, max_title_len: int) -> tuple[str, list[dict]]:
    found: list[dict] = []
    out_lines: list[str] = []
    for line in text.split('\n'):
        match = _SECTION_RE.match(line.strip())
        if match:
            major, minor, sub, title = match.groups()
            if major == chapter_id and len(title) <= max_title_len and title == title.strip():
                ident = f'{major}.{minor}' + (f'.{sub}' if sub else '')
                level = '###' if sub else '##'
                out_lines.append(f'{level} {ident} {title}')
                found.append({'section_id': ident, 'title': title, 'level': level})
                continue
        out_lines.append(line)
    return '\n'.join(out_lines), found


def chapter_header_line(chapter_id: str, chapter_title: str) -> str:
    return f'# {chapter_id}. {chapter_title}'


# ---------------------------------------------------------------- печатный номер страницы

_PRINTED_RE = re.compile(r'^([A-E]|[0-9lI]{1,2})-([0-9lI]{1,3})$')


def printed_page_label(text: str, pdf_page: int, overrides: dict) -> tuple[str | None, str]:
    if str(pdf_page) in overrides:
        return overrides[str(pdf_page)], 'reviewed_override'
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        value = line.replace(' ', '')
        match = _PRINTED_RE.match(value)
        if match:
            value = '-'.join(part.replace('l', '1').replace('I', '1') for part in match.groups())
            return value, 'header_ocr'
        return None, 'unknown'
    return None, 'empty_page'


# ---------------------------------------------------------------- правило включения страниц

_MARKUP_CHARS = re.compile(r'[#|>*_`\-=\[\]()]')
_WS = re.compile(r'\s+')


def plain_text_stats(markdown_text: str) -> tuple[int, float]:
    plain = _MARKUP_CHARS.sub(' ', markdown_text)
    letters = [c for c in plain if c.isalpha()]
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    n_chars = len(_WS.sub('', plain))
    ratio = non_ascii / len(letters) if letters else 0.0
    return n_chars, ratio


def page_eligibility(markdown_text: str, rule: dict) -> tuple[bool, str | None]:
    n_chars, nonascii_ratio = plain_text_stats(markdown_text)
    if n_chars < rule['min_chars']:
        return False, f'insufficient_text({n_chars}<{rule["min_chars"]})'
    if nonascii_ratio > rule['max_nonascii_alpha_ratio']:
        return False, f'suspected_corrupt_text_layer(nonascii={nonascii_ratio:.3f})'
    return True, None


# ---------------------------------------------------------------- патч флага извлечения

class _InhibitSpacesPatch:
    def __init__(self, enabled: bool, drop_collect_styles: bool = True):
        self.enabled = enabled
        self.drop_collect_styles = drop_collect_styles
        self._original = None

    def __enter__(self):
        if not self.enabled:
            return self
        if not pymupdf4llm.__version__.startswith('1.28.'):
            raise RuntimeError(
                'Патч флагов извлечения проверен только для pymupdf4llm 1.28.x; '
                f'найдена {pymupdf4llm.__version__}. Обновите requirements или '
                'выключите extraction.inhibit_spaces в config/parser.json.')
        self._original = pymupdf.Page.get_textpage

        def patched(page, flags=1, **kwargs):
            flags = flags | pymupdf.TEXT_INHIBIT_SPACES
            if self.drop_collect_styles:
                flags &= ~pymupdf.TEXT_COLLECT_STYLES
            return self._original(page, flags=flags, **kwargs)

        pymupdf.Page.get_textpage = patched
        return self

    def __exit__(self, *exc):
        if self._original is not None:
            pymupdf.Page.get_textpage = self._original
        return False


# ---------------------------------------------------------------- порядок строк (fallback)

def _page_lines(page, flags: int) -> list[dict]:
    """Текстовые строки страницы с координатами, в порядке блоков MuPDF"""
    data = page.get_text('dict', flags=flags, sort=False)
    lines = []
    for block_no, block in enumerate(data.get('blocks', [])):
        if block.get('type') != 0:
            continue
        for line in block.get('lines', []):
            spans = line.get('spans', [])
            text = ''.join(span.get('text', '') for span in spans).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line['bbox']
            origin = next((span.get('origin') for span in spans
                           if span.get('text', '').strip()), None)
            baseline = float(origin[1]) if origin else y1
            size = max((float(span.get('size', 0.0)) for span in spans), default=y1 - y0)
            lines.append({'block': block_no, 'x0': x0, 'y0': y0, 'baseline': baseline,
                          'x1': x1, 'y1': y1, 'size': size, 'text': text})
    return lines


def _prose_like(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 8 and letters / max(1, len(text)) > 0.6 and len(text.split()) >= 2


def _strong_continuation(left: str, right: str) -> bool:
    """Пара фрагментов выглядит как разрыв одной фразы, а не две колонки"""
    left = left.rstrip()
    right = right.lstrip()
    if not (_prose_like(left) and _prose_like(right)):
        return False
    if re.search(r'[.!?:;]$', left):
        return False
    return bool(right[:1].islower() or re.match(r'[\(\[]', right))


_TABLE_HEADING_RE = re.compile(r'(?m)^\s*(?:TABLE\s+[0-9A-Z]|\|)')


def looks_tabular(markdown_text: str) -> bool:
    """Страница содержит markdown-таблицу или заголовок TABLE — fallback не нужен"""
    return bool(_TABLE_HEADING_RE.search(markdown_text[:1500]))


def line_order_fallback(page, max_gap_pt: float = 12.0) -> tuple[str, dict]:
    """Восстановление порядка строк для страниц с разорванными строками.

    MuPDF иногда делит одну визуальную строку на блоки (например, левая и
    правая половины предложения) и возвращает блоки не в порядке чтения.
    Здесь строки группируются по базовой линии, близкие по горизонтали
    фрагменты склеиваются, группы читаются сверху вниз. Fallback применяется
    только если он реально склеил разорванную prose-строку и в исходном
    порядке был возврат вверх по странице; таблицы обрабатываются библиотекой.
    """
    flags = pymupdf.TEXT_MEDIABOX_CLIP | pymupdf.TEXT_INHIBIT_SPACES
    lines = _page_lines(page, flags)
    if not lines:
        return '', {'applied': False, 'reason': 'empty_page', 'merged_pairs': 0}

    max_baseline = -1.0
    for line in lines:
        tolerance = max(2.0, 0.25 * line['size'])
        line['backward'] = line['baseline'] < max_baseline - tolerance
        max_baseline = max(max_baseline, line['baseline'])

    rows: list[dict] = []
    for line in lines:
        tolerance = max(2.0, 0.25 * line['size'])
        target = None
        for row in rows:
            if abs(line['baseline'] - row['baseline']) <= tolerance:
                target = row
                break
        if target is None:
            rows.append({'baseline': line['baseline'], 'lines': [line]})
        else:
            target['lines'].append(line)

    merged_pairs = 0
    strong_merges = 0
    for row in rows:
        row['lines'].sort(key=lambda item: item['x0'])
        runs = []
        for line in row['lines']:
            if runs and line['x0'] - runs[-1][-1]['x1'] <= max_gap_pt:
                previous = runs[-1][-1]
                if previous['block'] != line['block']:
                    merged_pairs += 1
                    if _strong_continuation(previous['text'], line['text']) and (
                            previous['backward'] or line['backward']):
                        strong_merges += 1
                    if previous['backward'] or line['backward']:
                        previous['backward'] = True
                runs[-1].append(line)
            else:
                runs.append([line])
        row['runs'] = [' '.join(item['text'].strip() for item in run) for run in runs]
        row['run_bounds'] = [(run[0]['x0'], run[-1]['x1']) for run in runs]
        row['backward'] = any(item['backward'] for item in row['lines'])

    rows.sort(key=lambda row: (row['baseline'], row['lines'][0]['x0']))

    def _ambiguous_columns() -> bool:
        for row in rows:
            if len(row['runs']) < 2 or not all(_prose_like(text) for text in row['runs']):
                continue
            for left, right in zip(row['run_bounds'], row['run_bounds'][1:]):
                if right[0] - left[1] > max_gap_pt:
                    return True
        return False

    if _ambiguous_columns():
        return '', {'applied': False, 'reason': 'ambiguous_multi_column_layout',
                    'merged_pairs': merged_pairs}

    applied = strong_merges > 0
    if not applied:
        return '', {'applied': False, 'reason': 'no_line_fragment_order_issue',
                    'merged_pairs': merged_pairs}

    gaps = [rows[i]['baseline'] - rows[i - 1]['baseline'] for i in range(1, len(rows))
            if rows[i]['baseline'] > rows[i - 1]['baseline']]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    paragraph_gap = max(median_gap * 1.6, median_gap + 4) if median_gap else 1e9

    out = []
    previous = None
    for row in rows:
        if previous is not None and row['baseline'] - previous['baseline'] > paragraph_gap:
            out.append('')
        out.append('    '.join(row['runs']))
        previous = row
    text = '\n'.join(out)
    return text, {'applied': True, 'reason': 'line_fragment_merge',
                  'merged_pairs': merged_pairs, 'strong_merges': strong_merges,
                  'rows': len(rows)}


# ---------------------------------------------------------------- основной проход

def parse_pdf(pdf_path: Path, config: dict) -> list[dict]:
    extraction = config['extraction']
    if extraction['path'] != 'legacy':
        raise ValueError('Поддерживается только legacy-путь pymupdf4llm (без Tesseract).')

    pymupdf4llm.use_layout(False)
    with _InhibitSpacesPatch(enabled=extraction.get('inhibit_spaces', True),
                             drop_collect_styles=extraction.get('drop_collect_styles', True)):
        pages = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            show_progress=True,
            margins=extraction.get('margins', 0),
            table_strategy=extraction.get('table_strategy', 'lines_strict'),
        )

    rule = config['page_eligibility']
    chapters = config['chapters']
    overrides = config.get('printed_page_overrides', {})
    headings_cfg = config.get('headings', {})
    max_title_len = headings_cfg.get('max_title_len', 115)
    fallback_cfg = extraction.get('line_order_fallback', {}) or {}

    records: list[dict] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for chunk in pages:
            pdf_page = int(chunk['metadata']['page'])  # 1-based в pymupdf4llm
            raw = chunk.get('text') or ''

            extraction_mode = 'pymupdf4llm'
            fallback_info = None
            if (fallback_cfg.get('enabled', True) and not (chunk.get('tables') or [])
                    and not looks_tabular(raw)):
                fallback_text, fallback_info = line_order_fallback(
                    doc[pdf_page - 1], float(fallback_cfg.get('max_gap_pt', 12.0)))
                if fallback_info['applied']:
                    raw = fallback_text
                    extraction_mode = 'line_fragment_merge'

            chapter_id, chapter_title = chapter_at(pdf_page, chapters)
            text = normalize_page_text(raw)

            # Заголовки разделов внедряются до фильтрации: правило отбора страниц
            # не зависит от наличия заголовков, но тексты «шапок» остаются в корпусе.
            text, sections = inject_section_headers(text, chapter_id, max_title_len)
            if headings_cfg.get('inject_chapter_headers', True):
                starts = {int(start): (str(i), str(t)) for start, i, t in chapters}
                if pdf_page in starts:
                    line = chapter_header_line(*starts[pdf_page])
                    text = line + '\n\n' + text.lstrip('\n')

            label, label_method = printed_page_label(text, pdf_page, overrides)
            eligible, reason = page_eligibility(text, rule)

            records.append({
                'pdf_page': pdf_page,
                'chapter_id': chapter_id,
                'chapter_title': chapter_title,
                'printed_page': label,
                'printed_page_method': label_method,
                'text': text,
                'n_tables': len(chunk.get('tables') or []),
                'extraction_mode': extraction_mode,
                'line_order_fallback': fallback_info or {'applied': False},
                'section_headers': [s['section_id'] for s in sections],
                'retrieval_eligible': eligible,
                'exclusion_reason': reason,
            })
    return records
