"""Проверка исходного пдф apollo и выгрузка текстового слоя для просмотра"""
import hashlib
import json
from pathlib import Path
import sys

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'data/sources.json').read_text(encoding='utf-8'))
    source = manifest['sources'][0]
    pdf = ROOT / source['path']

    if not pdf.is_file():
        raise ValueError(f'PDF missing: {pdf}. Extract the entire archive.')

    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if digest != source['sha256']:
        raise ValueError('PDF checksum mismatch: file is damaged or is a different edition.')

    rows = []

    with pymupdf.open(pdf) as document:
        if document.is_repaired or len(document) != source['pdf_pages']:
            raise ValueError('Invalid PDF or unexpected page count.')
        for number, page in enumerate(document, 1):
            text = page.get_text('text', sort=True)
            rows.append({
                'page_id': f"{source['id']}:p{number:04d}",
                'source_id': source['id'], 'pdf_page': number,
                'source_path': source['path'], 'source_url': source['url'],
                'publication_month': source['publication_month'],
                'retrieved_date': source['retrieved_date'],
                'text': text, 'non_whitespace_characters': len(''.join(text.split())),
            })

    substantial = sum(row['non_whitespace_characters'] >= 300 for row in rows)

    if substantial < 50:
        raise ValueError(f'Only {substantial} pages have at least 300 non-whitespace characters.')

    questions = json.loads((ROOT / 'data/questions_draft.json').read_text(encoding='utf-8'))

    for question in questions:
        if not all(1 <= p <= len(rows) for p in question['verified_seed_pdf_pages']):
            raise ValueError(f"Invalid page reference: {question['id']}")

    output = ROOT / 'outputs' / f"audit_{digest[:12]}_pymupdf_{pymupdf.VersionBind}"
    output.mkdir(parents=True, exist_ok=True)

    report = {
        'source_id': source['id'], 'sha256': digest, 'pdf_pages': len(rows),
        'pages_with_at_least_300_non_whitespace_characters': substantial,
        'low_text_pdf_pages': [r['pdf_page'] for r in rows if r['non_whitespace_characters'] < 300],
        'question_count': len(questions), 'extractor': 'PyMuPDF',
        'extractor_version': pymupdf.VersionBind,
        'status': 'corpus_inventory_passed_not_markdown_quality_certification',
        'notes': 'Text layer contains OCR errors; tables and headings require separate parsing and visual QA.',
    }

    (output / 'corpus_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'pages_raw.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')

    print(f"PDF: {len(rows)} pages; text-rich pages (heuristic): {substantial}")
    print(f"Questions: {len(questions)}; checksum: OK")
    print(f"Saved: {output}")
    print('Stage 1 passed. Markdown parsing, gold relevance review and RAG are the next stages.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
