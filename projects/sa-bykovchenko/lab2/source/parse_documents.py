"""PDF Apollo 11 -> Markdown по страницам и блоки с привязкой к исходнику"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
VERSION = '2.0.0'
FLAGS = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_INHIBIT_SPACES


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def normalize(text):
    # Удаляем только мягкий перенос, обычный дефис сохраняем
    text = re.sub(r'\u00ad\s*\n\s*', '', text)
    text = text.replace('\u00ad', '').replace('\u00a0', ' ')
    text = re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[a-z])', '-', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([,;:!?])', r'\1', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    return text


def page_lines(page, tolerance):
    """Сначала объединить фрагменты по базовой линии, затем соединить переносы строк

    В скане одно предложение может быть разделено на левый и правый PDF-блоки,
    сортировка целых блоков нарушила бы порядок чтения"""
    fragments = []
    for block in page.get_text('dict', flags=FLAGS)['blocks']:
        for line in block.get('lines', []):
            text = ''.join(span['text'] for span in line['spans']).strip()
            if text:
                bbox = list(line['bbox'])
                # Границы ocr шрифтов могут быть завышены. Используем реальные базовые линии текста,
                # иначе фрагменты сдвинутся на строку вверх
                baseline = statistics.median(span['origin'][1] for span in line['spans'])
                fragments.append({'bbox': bbox, 'text': text, 'cy': baseline})

    rows = []
    for fragment in sorted(fragments, key=lambda f: (f['cy'], f['bbox'][0])):
        if rows and abs(fragment['cy'] - rows[-1]['cy']) <= tolerance:
            rows[-1]['parts'].append(fragment)
        else:
            rows.append({'cy': fragment['cy'], 'parts': [fragment]})

    result = []
    for row in rows:
        parts = sorted(row['parts'], key=lambda f: f['bbox'][0])
        result.append({
            'text': ' '.join(part['text'] for part in parts),
            'baseline': statistics.median(part['cy'] for part in parts),
            'bbox': [min(p['bbox'][0] for p in parts), min(p['bbox'][1] for p in parts),
                     max(p['bbox'][2] for p in parts), max(p['bbox'][3] for p in parts)],
        })

    return result


def chapter_at(number, config):
    result = ('FRONT', 'FRONT MATTER')

    for start, identifier, title in config['chapters']:
        if start > number:
            break
        result = (identifier, title)

    return result


def printed_label(lines, number, config):
    if str(number) in config['printed_page_overrides']:
        return config['printed_page_overrides'][str(number)], 'reviewed_override'

    for line in lines:
        if line['bbox'][1] > 85:
            continue
        value = re.sub(r'\s+', '', line['text'])
        match = re.fullmatch(r'([A-E]|[0-9lI]{1,2})-([0-9lI]{1,3})', value)
        if match:
            value = '-'.join(v.replace('l', '1').replace('I', '1') for v in match.groups())
            return value, 'header_ocr'

    return None, 'unknown'


def heading(text, chapter):
    # Нормализуем пробелы в номерах разделов, не затрагивая произвольные десятичные числа
    match = re.match(r'^([A-E]|\d{1,2})\s*\.\s*(\d+)(.*)$', text)
    if not match or match[1] != chapter:
        return None

    ident = match[1] + '.' + match[2]
    rest = match[3]

    sub = re.match(r'^\s*\.\s*(\d+)(.*)$', rest)
    if sub:
        ident += '.' + sub[1]
        rest = sub[2]

    title = rest.strip(' .')
    if not title or len(title) > 115 or not re.match(r'[A-Z]', title):
        return None

    if ident.endswith('.0'):
        return chapter, title

    return ident, title


def prose_blocks(lines, chapter, state, config):
    body = [line.copy() for line in lines
            if line['bbox'][1] >= 80 and re.search(r'[A-Za-z0-9]', line['text'])]
    # Номер раздела и его заголовок могут находиться на соседних строках
    combined = []
    i = 0

    while i < len(body):
        line = body[i]
        if (re.fullmatch(r'(?:[A-E]|\d{1,2})(?:\s*\.\s*\d+){1,2}', line['text'].strip())
                and i + 1 < len(body) and re.match(r'[A-Z]', body[i + 1]['text'])
                and body[i + 1]['bbox'][1] - line['bbox'][3] < 20):
            following = body[i + 1]
            line = {'text': line['text'] + ' ' + following['text'], 'baseline': following['baseline'],
                    'bbox': [min(line['bbox'][0], following['bbox'][0]), line['bbox'][1],
                             max(line['bbox'][2], following['bbox'][2]), following['bbox'][3]]}
            i += 1
        combined.append(line)
        i += 1

    blocks, pending = [], []

    def flush():
        if not pending:
            return

        text = normalize('\n'.join(line['text'] for line in pending))
        kind = 'list_item' if re.match(r'^[a-z]\.\s+', text) else 'paragraph'
        md = '- ' + text if kind == 'list_item' else text

        blocks.append({'kind': kind, 'text': text, 'markdown': md,
                       'section_id': state['id'], 'section_title': state['title'],
                       'bbox': [round(min(l['bbox'][0] for l in pending), 2), round(pending[0]['bbox'][1], 2),
                                round(max(l['bbox'][2] for l in pending), 2), round(pending[-1]['bbox'][3], 2)]})

        pending.clear()

    for line in combined:
        text = normalize(line['text'])
        found = heading(text, chapter)

        if found:
            flush()
            state['id'], state['title'] = found
            level = min(3, found[0].count('.') + 1)
            blocks.append({'kind': 'heading', 'text': f'{found[0]} {found[1]}',
                           'markdown': '#' * level + ' ' + f'{found[0]} {found[1]}',
                           'section_id': state['id'], 'section_title': state['title'],
                           'bbox': [round(x, 2) for x in line['bbox']]})
        else:
            if pending and (line['bbox'][1] - pending[-1]['bbox'][3] > config['paragraph_gap']
                            or re.match(r'^[a-z]\.\s+', text)):
                flush()
            pending.append(line)

    flush()

    return blocks


def table_blocks(override, chapter, chapter_title):
    result = []
    for table in override['tables']:
        def row(cells):
            return '| ' + ' | '.join(c.replace('|', '\\|') for c in cells) + ' |'

        md = '\n'.join(['### ' + table['title'], '', row(table['headers']),
                        row(['---'] * len(table['headers'])), *[row(r) for r in table['rows']]])
        result.append({'kind': 'table', 'text': md, 'markdown': md, 'section_id': chapter,
                       'section_title': chapter_title, 'bbox': None, 'table': table,
                       'transcription_method': override['method']})
    return result


def make_export(args):
    config_path = args.config.resolve()
    config = read_json(config_path)
    source = read_json(ROOT / 'data/sources.json')['sources'][0]
    tables_path = ROOT / 'data/verified_tables.json'
    tables = read_json(tables_path)
    pdf = ROOT / source['path']
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()

    if digest != source['sha256']:
        raise ValueError('Source checksum mismatch. This parser profile is for the supplied NASA PDF only.')

    for item in tables.values():
        if not digest.startswith(item['source_sha256_prefix']):
            raise ValueError('Table transcription does not match source PDF.')

    fingerprint = {'source_sha256': digest, 'config': config, 'tables': tables,
                   'source_metadata': source, 'parser_version': VERSION,
                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'pymupdf': pymupdf.VersionBind, 'render_review': args.render_review}

    build_id = hashlib.sha256(canonical(fingerprint).encode()).hexdigest()[:16]
    base = args.output.resolve()
    target = base / build_id

    if target.exists():
        try:
            complete = read_json(target / 'parse_report.json')
            if complete['build_id'] == build_id and complete['status'] == 'completed':
                if not args.rebuild:
                    write_json(base / 'latest.json', {'build_id': build_id, 'path': build_id})
                    print(f'Already built: {target}\nUse --rebuild to regenerate this version.')
                    return
        except (OSError, ValueError, KeyError):
            if not args.rebuild:
                raise ValueError('Incomplete existing build. Run with --rebuild.')

    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.parse-', dir=base))

    try:
        (work / 'pages').mkdir()
        if args.render_review:
            (work / 'review_images').mkdir()

        pages, blocks, review = [], [], []
        chapter_previous, state = None, {}

        with pymupdf.open(pdf) as document:
            if document.is_repaired or len(document) != source['pdf_pages']:
                raise ValueError('Invalid PDF or page count.')
            for number, page in enumerate(document, 1):
                chapter, chapter_title = chapter_at(number, config)

                if chapter != chapter_previous:
                    state = {'id': chapter, 'title': chapter_title}
                    chapter_previous = chapter

                raw = page.get_text('text', flags=FLAGS)
                lines = page_lines(page, config['line_y_tolerance'])
                label, label_method = printed_label(lines, number, config)
                eligible = any(start <= number <= end for start, end in config['prose_page_ranges'])
                override = tables.get(str(number))
                warnings = []

                if override:
                    page_blocks = table_blocks(override, chapter, chapter_title)
                    mode, eligible = 'verified_table_transcription', True

                elif eligible:
                    page_blocks = prose_blocks(lines, chapter, state, config)
                    mode = 'prose_geometry'
                    unusual = sum(not c.isascii() and c.isalpha() for c in raw)
                    if unusual > max(15, len(raw) * 0.02):
                        eligible = False
                        warnings.append('suspected_corrupt_text_layer')
                    if re.search(r'\bTABLE\s+[0-9A-E]', raw):
                        eligible = False
                        warnings.append('table_layout_needs_review')
                    warnings.append('ocr_not_fully_proofread')

                else:
                    mode = 'source_fallback'
                    page_blocks = []
                    warnings.append('layout_not_validated_for_retrieval')

                if label is None:
                    warnings.append('printed_page_not_identified_use_pdf_page')

                page_id = f"{source['id']}:p{number:04d}"
                meta = {'page_id': page_id, 'source_id': source['id'], 'source_title': source['title'],
                        'pdf_page': number, 'printed_page': label, 'printed_page_method': label_method,
                        'source_path': source['path'], 'source_url': source['url'],
                        'citation_url': source['url'] + f'#page={number}',
                        'publication_month': source['publication_month'], 'retrieved_date': source['retrieved_date'],
                        'source_sha256': digest, 'language': 'en', 'chapter_id': chapter,
                        'chapter_title': chapter_title,
                        'section_at_page_start': page_blocks[0]['section_id'] if page_blocks else state['id'],
                        'extraction_method': mode, 'retrieval_eligible': eligible, 'warnings': warnings}

                for index, block in enumerate(page_blocks):
                    block.update({**meta, 'block_id': f'{page_id}:b{index:03d}', 'build_id': build_id})
                    block['section_path'] = [f'{chapter} {chapter_title}']
                    if block['section_id'] != chapter:
                        block['section_path'].append(f"{block['section_id']} {block['section_title']}")
                    blocks.append(block)

                content = '\n\n'.join(block['markdown'] for block in page_blocks)
                if not content:
                    content = ('> This page has not been normalized into reliable text. '
                               'Use the source PDF; raw OCR is retained separately.')

                if not eligible and page_blocks:
                    content = '> REVIEW REQUIRED: excluded from retrieval.\n\n' + content

                front = '\n'.join(key + ': ' + json.dumps(value, ensure_ascii=False) for key, value in meta.items())
                md = '---\n' + front + '\n---\n\n' + f'# PDF page {number} — {chapter_title}\n\n'
                md += f"[Original PDF, page {number}]({meta['citation_url']})\n\n" + content + '\n'
                (work / 'pages' / f'p{number:04d}.md').write_text(md, encoding='utf-8')

                pages.append({**meta, 'markdown_path': f'pages/p{number:04d}.md', 'raw_text': raw,
                              'normalized_text': '\n\n'.join(b['text'] for b in page_blocks),
                              'block_ids': [b['block_id'] for b in page_blocks]})

                if not eligible:
                    review.append({'pdf_page': number, 'reasons': warnings, 'citation_url': meta['citation_url']})

                if args.render_review and number in config['review_preview_pages']:
                    page.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(work / 'review_images' / f'p{number:04d}.png')

        questions = read_json(ROOT / 'data/questions_draft.json')
        question_pages = {p for q in questions for p in q['verified_seed_pdf_pages']}

        missing = sorted(p for p in question_pages if not pages[p - 1]['retrieval_eligible'])
        if missing:
            raise ValueError(f'Question evidence pages not eligible: {missing}')

        text_pages = sum(p['retrieval_eligible'] and len(''.join(p['normalized_text'].split())) >= 300 for p in pages)
        if text_pages < 50:
            raise ValueError(f'Only {text_pages} eligible text-rich pages; at least 50 needed.')

        for filename, records in [('pages.jsonl', pages), ('blocks.jsonl', blocks)]:
            (work / filename).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records),
                                         encoding='utf-8')

        document_md = ['# Apollo 11 Mission Report — normalized export', '',
                       'Page-level source metadata and raw text: pages.jsonl. Only retrieval_eligible blocks may be indexed.',
                       '']
        for p in pages:
            document_md += [f"## PDF page {p['pdf_page']} | printed {p['printed_page'] or 'unknown'}", '',
                            f"[Page Markdown]({p['markdown_path']}) | [Source]({p['citation_url']})", '',
                            (work / p['markdown_path']).read_text(encoding='utf-8').split('---\n', 2)[-1], '']

        (work / 'corpus.md').write_text('\n'.join(document_md), encoding='utf-8')
        write_json(work / 'review_queue.json', review)
        write_json(work / 'build_config.json', fingerprint)

        report = {'status': 'completed', 'build_id': build_id, 'parser_version': VERSION,
                  'created_at_utc': datetime.now(timezone.utc).isoformat(), 'pdf_pages': len(pages),
                  'retrieval_eligible_pages': sum(p['retrieval_eligible'] for p in pages),
                  'eligible_text_rich_pages': text_pages, 'blocks': len(blocks),
                  'verified_markdown_tables': sum(b['kind'] == 'table' for b in blocks),
                  'headings': sum(b['kind'] == 'heading' for b in blocks),
                  'list_items': sum(b['kind'] == 'list_item' for b in blocks),
                  'source_fallback_or_review_pages': len(review),
                  'all_question_seed_pages_eligible': not missing,
                  'warning_counts': dict(Counter(w for p in pages for w in p['warnings'])),
                  'limitation': 'Eligible prose has not been exhaustively proofread; diagrams and unverified tables are excluded from retrieval.'}

        write_json(work / 'parse_report.json', report)
        if target.exists():
            # Удаляем только эту сборку, созданную скриптом и адресуемую по хешу
            shutil.rmtree(target)
        work.rename(target)
        write_json(base / 'latest.json', {'build_id': build_id, 'path': build_id})
        print(
            f"PDF pages: {len(pages)} | eligible pages: {report['retrieval_eligible_pages']} | text-rich: {text_pages}")
        print(
            f"Headings: {report['headings']} | list items: {report['list_items']} | verified tables: {report['verified_markdown_tables']}")
        print(f'All question seed pages: OK\nSaved: {target}')
        print('Parsing finished. Review exclusions and OCR limitations in README_STEP2.md.')

    finally:
        if work.exists():
            shutil.rmtree(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/parser.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--render-review', action='store_true',
                        help='Render selected original pages as PNG for visual QA.')

    args = parser.parse_args()

    try:
        make_export(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
