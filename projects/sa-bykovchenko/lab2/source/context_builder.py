from __future__ import annotations

DEFAULT_MAX_CONTEXT_BYTES = 24_000


def build_context(results: list[dict], max_bytes: int = DEFAULT_MAX_CONTEXT_BYTES) -> dict:
    if max_bytes < 1000:
        raise ValueError('max_bytes слишком мал для осмысленного контекста.')

    seen_texts: set[str] = set()
    sources, dropped = [], []
    blocks = []
    total = 0

    for item in results:
        text = item['text'].strip()
        key = ' '.join(text.split())
        if key in seen_texts:
            dropped.append({'chunk_id': item['chunk_id'], 'rank': item['rank'],
                            'reason': 'duplicate_text'})
            continue
        seen_texts.add(key)

        pages = str(item['pdf_page'])
        section = item.get('section_id') or '?'
        header = f"[{len(sources) + 1}] (PDF стр. {pages}; раздел {section}): "
        block = header + text
        block_bytes = len(block.encode('utf-8'))

        # Если чанк превышает оставшийся бюджет — пропускаем его
        if total + block_bytes + 2 > max_bytes:
            dropped.append({'chunk_id': item['chunk_id'], 'rank': item['rank'],
                            'reason': f'byte_budget({total + block_bytes} > {max_bytes})'})
            continue

        # Если помещается — добавляем целиком
        blocks.append(block)
        total += block_bytes + 2
        sources.append({
            'citation_id': len(sources) + 1,
            'chunk_id': item['chunk_id'],
            'rank': item['rank'],
            'pdf_page': item['pdf_page'],
            'printed_page': item.get('printed_page'),
            'section_id': item.get('section_id'),
            'heading_path': item.get('heading_path'),
            'score': item.get('score'),
            'score_type': item.get('score_type'),
            'citation_url': item.get('citation_url'),
            'text': text,
        })

    if not sources:
        raise ValueError('Ни один фрагмент не вошёл в бюджет контекста.')

    context = '\n\n'.join(blocks)
    return {'context': context, 'sources': sources, 'dropped': dropped,
            'context_bytes': len(context.encode('utf-8'))}
