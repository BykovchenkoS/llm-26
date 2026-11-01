import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def squash(text):
    return re.sub(r'[^a-z0-9]', '', text.casefold())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(data):
    ids = set()
    for q in data['questions']:
        if q['id'] in ids:
            raise ValueError('Duplicate question ID: ' + q['id'])
        ids.add(q['id'])
        evidence = q.get('evidence', [])
        facts = q.get('facts', [])
        if q['answerable'] and (not facts or not evidence):
            raise ValueError('Answerable question without facts/evidence: ' + q['id'])
        if not q['answerable'] and (facts or evidence):
            raise ValueError('Unanswerable question has positive labels: ' + q['id'])
        used = set()
        fact_ids = set()
        for fact in facts:
            if fact['id'] in fact_ids:
                raise ValueError('Duplicate fact ID')
            fact_ids.add(fact['id'])
            refs = fact.get('evidence', [])
            if not refs or any(type(r) is not int or r < 0 or r >= len(evidence) for r in refs):
                raise ValueError('Invalid or missing evidence: ' + fact['id'])
            used.update(refs)
        if used != set(range(len(evidence))):
            raise ValueError('Unreferenced evidence: ' + q['id'])
        for e in evidence:
            if not squash(e.get('quote', '')) or type(e.get('page')) is not int or e['page'] < 1:
                raise ValueError('Empty/invalid quote: ' + q['id'])


def matches(chunk, evidence, default_source):
    quote = squash(evidence['quote'])
    return bool(quote) and chunk.get('source_id') == evidence.get('source_id', default_source) \
        and chunk.get('pdf_page') == evidence['page'] and quote in squash(chunk['text'])


def compare(data, chunks, pages):
    """Доступность доказательств во всём индексе, НЕ top-k retrieval."""
    source = data['_meta']['source_id']
    details = []
    for q in data['questions']:
        if not q['answerable']:
            continue
        mapped = []
        for i, e in enumerate(q['evidence']):
            ids = [c['chunk_id'] for c in chunks if matches(c, e, source)]
            page = pages.get((e.get('source_id', source), e['page']))
            in_page = page is not None and squash(e['quote']) in squash(page['text'])
            mapped.append({'index': i, 'page': e['page'], 'chunk_ids': ids,
                           'status': 'covered' if ids else (
                               'split_or_chunking_loss' if in_page else 'parser_text_mismatch')})
        available = sum(any(mapped[i]['chunk_ids'] for i in f['evidence']) for f in q['facts'])
        details.append({'id': q['id'], 'split': q['split'], 'facts_available': available,
                        'facts_total': len(q['facts']), 'evidence': mapped})
    aggregate = {}
    for split in ('dev', 'test', 'all'):
        subset = [q for q in details if split == 'all' or q['split'] == split]
        ev = [e for q in subset for e in q['evidence']]
        aggregate[split] = {
            'questions': len(subset), 'facts_total': sum(q['facts_total'] for q in subset),
            'facts_available': sum(q['facts_available'] for q in subset),
            'fact_availability_macro': sum(q['facts_available'] / q['facts_total'] for q in subset) / len(
                subset) if subset else None,
            'fully_covered_questions': sum(q['facts_available'] == q['facts_total'] for q in subset),
            'evidence_total': len(ev),
            **{status: sum(e['status'] == status for e in ev) for status in
               ('covered', 'split_or_chunking_loss', 'parser_text_mismatch')}}
    return {'aggregate': aggregate, 'questions': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qrels', type=Path, default=ROOT / 'data/qrels_evidence.json')
    parser.add_argument('--before', type=Path, default=ROOT / 'data/history/qrels_evidence_v3.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/step1/evidence_comparison.json')
    args = parser.parse_args()
    after, before = read(args.qrels), read(args.before)
    validate(after)
    if [(q['id'], q['split'], q['question_ru'], q['question_en']) for q in before['questions']] != \
            [(q['id'], q['split'], q['question_ru'], q['question_en']) for q in after['questions']]:
        raise ValueError('Questions or DEV/TEST changed')
    import pymupdf
    source = after['_meta']['source_id']
    sources = {s['id']: s for s in read(ROOT / 'data/sources.json')['sources']}
    pdf_path = ROOT / sources[source]['path']
    expected_hash = sources[source].get('sha256')
    if expected_hash and sha(pdf_path) != expected_hash:
        raise ValueError('PDF checksum mismatch')
    with pymupdf.open(pdf_path) as pdf:
        for q in after['questions']:
            for e in q['evidence']:
                if e.get('source_id', source) != source or e['page'] > len(pdf) or \
                        squash(e['quote']) not in squash(pdf[e['page'] - 1].get_text(sort=True)):
                    raise ValueError('Evidence not present in source PDF: ' + q['id'])
    output = {'stage': 1, 'qrels_before_sha256': sha(args.before), 'qrels_after_sha256': sha(args.qrels),
              'pdf_sha256': sha(pdf_path), 'pdf_quote_check': 'all passed',
              'note': 'Corpus-wide evidence availability, not retrieval@k. TEST is diagnostic, not used to choose a configuration.',
              'builds': []}
    for config_path in sorted((ROOT / 'data/indexes').glob('*/build_config.json')):
        config = read(config_path)
        if config['config']['splitter'] not in ('markdown_header', 'page_recursive'):
            continue
        chunks_path = config_path.parent / 'chunks.jsonl'
        pages_path = ROOT / 'data/processed' / config['parser_build_id'] / 'pages.jsonl'
        if sha(pages_path) != config['pages_sha256']:
            raise ValueError('Changed parser artifact')
        chunk_rows = rows(chunks_path)
        pages = {(p['source_id'], p['pdf_page']): p for p in rows(pages_path)}
        output['builds'].append({'build_id': config_path.parent.name,
                                 'chunk_size': config['config']['chunk_size'], 'overlap': config['config']['overlap'],
                                 'splitter': config['config']['splitter'], 'chunk_count': len(chunk_rows),
                                 'chunks_sha256': sha(chunks_path), 'pages_sha256': sha(pages_path),
                                 'before': compare(before, chunk_rows, pages),
                                 'after': compare(after, chunk_rows, pages)})
    if len(output['builds']) < 3:
        raise ValueError('Need at least three saved chunk configurations')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(args.out)
    for b in output['builds']:
        print(b['chunk_size'], b['splitter'], b['before']['aggregate']['all'], b['after']['aggregate']['all'])


if __name__ == '__main__':
    main()
