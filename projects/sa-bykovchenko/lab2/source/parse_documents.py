from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import pymupdf4llm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parsing 

ROOT = Path(__file__).resolve().parents[1]


def build_page_markdown(record: dict, meta: dict) -> str:
    front = '\n'.join(f'{key}: {json.dumps(value, ensure_ascii=False)}' for key, value in meta.items())
    body = record['text'] if record['text'].strip() else (
        '> Страница не содержит извлекаемого текста (графика/скан без слоя). '
        'Обращайтесь к исходному PDF.')
    note = ''
    if not record['retrieval_eligible']:
        note = f"> Исключена из retrieval: {record['exclusion_reason']}\n\n"
    return f'---\n{front}\n---\n\n{note}{body}\n'


def make_export(args) -> Path:
    config = parsing.read_json(args.config.resolve())
    source = parsing.read_json(ROOT / 'data/sources.json')['sources'][0]
    pdf = ROOT / source['path']
    digest = parsing.file_hash(pdf)
    if digest != source['sha256']:
        raise ValueError('Source checksum mismatch: парсер рассчитан на приложенный NASA PDF.')

    fingerprint = {
        'parser_version': parsing.PARSER_VERSION,
        'source_sha256': digest,
        'source_metadata': source,
        'config': config,
        'parsing_sha256': parsing.file_hash(ROOT / 'source/parsing.py'),
        'pymupdf': pymupdf.__version__,
        'pymupdf4llm': pymupdf4llm.__version__,
        'pymupdf4llm_path': 'legacy(use_layout=False)',
    }
    build_id = hashlib16(fingerprint)
    base = args.output.resolve()
    target = base / build_id

    if target.exists():
        try:
            done = parsing.read_json(target / 'parse_report.json')
            if done.get('build_id') == build_id and done.get('status') == 'completed':
                if not args.rebuild:
                    parsing.write_json(base / 'latest.json', {'build_id': build_id, 'path': build_id})
                    print(f'Already built: {target}\nUse --rebuild to regenerate.')
                    return target
        except (OSError, ValueError, KeyError):
            if not args.rebuild:
                raise ValueError('Incomplete existing build. Run with --rebuild.')

    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.parse-', dir=base))
    try:
        (work / 'pages').mkdir()
        records = parsing.parse_pdf(pdf, config)

        if len(records) != source['pdf_pages']:
            raise ValueError(f'Ожидалось {source["pdf_pages"]} страниц, получено {len(records)}.')

        eligible = [r for r in records if r['retrieval_eligible']]
        text_rich = [r for r in eligible if parsing.plain_text_stats(r['text'])[0] >= 300]
        if len(text_rich) < 50:
            raise ValueError(f'Только {len(text_rich)} текстонасыщенных страниц; нужно >= 50.')

        rows = []
        for record in records:
            meta = {
                'source_id': source['id'], 'source_title': source['title'],
                'source_url': source['url'],
                'citation_url': source['url'] + f'#page={record["pdf_page"]}',
                'source_sha256': digest, 'language': 'en',
                'publication_month': source['publication_month'],
                'retrieved_date': source['retrieved_date'],
                'parser_build_id': build_id,
                'extraction': f'pymupdf4llm=={pymupdf4llm.__version__}/legacy',
                **{k: v for k, v in record.items() if k != 'text'},
            }
            md = build_page_markdown(record, meta)
            (work / 'pages' / f'p{record["pdf_page"]:04d}.md').write_text(md, encoding='utf-8')
            rows.append({
                'page_id': f"{source['id']}:p{record['pdf_page']:04d}",
                'markdown_path': f"pages/p{record['pdf_page']:04d}.md",
                'text': record['text'],
                **{k: v for k, v in meta.items() if k != 'citation_url'},
                'citation_url': meta['citation_url'],
            })

        (work / 'pages.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')

        corpus = ['# Apollo 11 Mission Report — pymupdf4llm export', '',
                  'Извлечено pymupdf4llm (legacy-путь, встроенный OCR-слой).',
                  'Индексируются только страницы с retrieval_eligible=true; причины исключений — в pages.jsonl.', '']
        for row in rows:
            corpus += [f"## PDF page {row['pdf_page']} | printed {row['printed_page'] or 'unknown'} "
                       f"| chapter {row['chapter_id']}", '',
                       f"[Page Markdown]({row['markdown_path']}) | [Source]({row['citation_url']})", '',
                       row['text'], '']
        (work / 'corpus.md').write_text('\n'.join(corpus), encoding='utf-8')

        report = {
            'status': 'completed', 'build_id': build_id,
            'parser_version': parsing.PARSER_VERSION,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'pdf_pages': len(records),
            'retrieval_eligible_pages': len(eligible),
            'eligible_text_rich_pages': len(text_rich),
            'pages_with_tables': sum(1 for r in records if r['n_tables']),
            'pages_with_section_headers': sum(1 for r in records if r['section_headers']),
            'section_headers_total': sum(len(r['section_headers']) for r in records),
            'printed_page_identified': sum(1 for r in records if r['printed_page']),
            'excluded_pages': len(records) - len(eligible),
            'pages_with_line_order_fallback': sum(
                1 for r in records if r.get('extraction_mode') == 'line_fragment_merge'),
            'line_order_fallback_rule': config['extraction'].get('line_order_fallback', {}),
            'exclusion_rule': config['page_eligibility'],
            'limitation': (
                'Извлечение идёт по встроенному OCR-слою исходного скана: сохраняются его '
                'ошибки (1/l, отдельные неверные символы); таблицы реконструируются '
                'pymupdf4llm lines_strict и наследуют шум слоя. Списки не преобразуются '
                'в markdown-списки. Fallback порядка строк применяется только к prose-'
                'страницам с разорванными блоками и фиксируется постранично.'),
        }
        reasons = Counter(r['exclusion_reason'].split('(')[0]
                          for r in records if not r['retrieval_eligible'])
        report['exclusion_reason_counts'] = dict(reasons)
        parsing.write_json(work / 'parse_report.json', report)
        parsing.write_json(work / 'build_config.json', fingerprint)

        if target.exists():
            shutil.rmtree(target)  # только сборка, созданная этим скриптом (адресуется хешем)
        work.rename(target)
        parsing.write_json(base / 'latest.json', {'build_id': build_id, 'path': build_id})

        print(f"PDF pages: {len(records)} | eligible: {len(eligible)} | text-rich(>=300 chars): {len(text_rich)}")
        print(f"Tables on pages: {report['pages_with_tables']} | section headers: {report['section_headers_total']}")
        print(f"Excluded: {report['excluded_pages']} | reasons: {dict(reasons)}")
        print(f'Saved: {target}')
        return target

    finally:
        if work.exists():
            shutil.rmtree(work)


def hashlib16(value) -> str:
    return hashlib.sha256(parsing.canonical(value).encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/parser.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()

    try:
        make_export(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
