from pathlib import Path
import json
import chunking
import index_core as core
import audit_evidence as evidence

ROOT = core.ROOT


def main():
    code_sha = core.file_hash(ROOT / 'source/chunking.py')
    qrels = core.read_json(ROOT / 'data/qrels_evidence.json')
    evidence.validate(qrels)
    results = []

    for config_path in sorted((ROOT / 'data/chunks').glob('*/build_config.json')):
        fingerprint = core.read_json(config_path)
        if fingerprint.get('chunking_sha256') != code_sha:
            continue
        cfg = fingerprint['config']
        pages_path = ROOT / 'data/processed' / fingerprint['parser_build_id'] / 'pages.jsonl'
        if not pages_path.exists() or core.file_hash(pages_path) != fingerprint['pages_sha256']:
            print(f'Пропуск {config_path.parent.name}: страницы изменились, пересоберите чанки')
            continue
        pages = core.read_jsonl(pages_path)
        path = config_path.parent / 'chunks.jsonl'
        chunks = core.read_jsonl(path)
        chunking.check_chunks(chunks, pages, cfg)
        entries = {(p['source_id'], p['pdf_page']): ee
                   for p, ee in chunking.iter_page_entries(pages, None, cfg)}
        for c in chunks:
            matching = [e for e in entries[(c['source_id'], c['pdf_page'])]
                        if e['heading_path'] == c['heading_path']
                        and chunking._squash(c['text']) in chunking._squash(e['text'])]
            if not matching:
                raise ValueError('Неверный раздел/merge: ' + c['chunk_id'])
        page_map = {(p['source_id'], p['pdf_page']): p for p in pages}
        old = []
        for old_path in sorted((ROOT / 'data/indexes').glob('*/build_config.json')):
            old_fp = core.read_json(old_path)
            if old_fp['config'] == cfg and old_fp['pages_sha256'] == fingerprint['pages_sha256']:
                old_chunks = core.read_jsonl(old_path.parent / 'chunks.jsonl')
                old.append({'build_id': old_path.parent.name, 'chunk_count': len(old_chunks),
                            'evidence': evidence.compare(qrels, old_chunks, page_map)['aggregate']})
        results.append({'build_id': config_path.parent.name, 'splitter': cfg['splitter'],
                        'chunk_size': cfg['chunk_size'], 'overlap': cfg['overlap'],
                        'chunk_count': len(chunks), 'token_max': max(c['embedding_token_count'] for c in chunks),
                        'chunks_sha256': core.file_hash(path), 'pages_sha256': fingerprint['pages_sha256'],
                        'hierarchy_and_provenance_check': 'passed', 'before': old,
                        'after': evidence.compare(qrels, chunks, page_map)})

    if not results:
        raise ValueError('Сначала запустите build_index.py --chunks-only для нового кода')
    out = ROOT / 'reports/step2/hierarchy_comparison.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'stage': 2, 'chunking_sha256': code_sha,
                              'qrels_sha256': core.file_hash(ROOT / 'data/qrels_evidence.json'),
                              'note': 'Evidence availability in corpus, not retrieval@k or QA. No new embeddings.',
                              'builds': results}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for row in results:
        print(row['splitter'], row['chunk_size'], 'chunks=', row['chunk_count'],
              'max_tokens=', row['token_max'], 'evidence=', row['after']['aggregate']['all']['covered'])
    print(out)


if __name__ == '__main__':
    main()
