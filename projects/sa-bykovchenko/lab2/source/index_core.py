"""Разбиение с учётом токенов и символьных диапазонов исходника, эмбеддинги E5 и индекс BM25."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    if config['splitter'] not in ('section', 'token_window'):
        raise ValueError('splitter must be section or token_window')

    if config['vectorization'] not in ('dense', 'sparse', 'hybrid'):
        raise ValueError('vectorization must be dense, sparse or hybrid')

    if not 100 <= config['chunk_size'] <= 1000:
        raise ValueError('chunk_size must be between 100 and 1000')

    if not 0 <= config['overlap'] < config['chunk_size']:
        raise ValueError('Require 0 <= overlap < chunk_size')

    if not 0 < config['max_input_tokens'] or config['chunk_size'] > config['max_input_tokens']:
        raise ValueError('chunk_size exceeds max_input_tokens; E5-small supports at most 512.')

    if config['batch_size'] < 1 or config['threads'] < 1:
        raise ValueError('batch_size and threads must be positive')

    if not config['bm25_k1'] > 0 or not 0 <= config['bm25_b'] <= 1 or config['rrf_k'] < 1:
        raise ValueError('Invalid BM25/RRF parameters')

    if not isinstance(config['include_headings'], bool):
        raise ValueError('include_headings must be a JSON boolean')


def load_tokenizer(config, offline=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config['embedding_model'],
                                              revision=config['model_revision'], cache_dir=ROOT / 'data/model_cache',
                                              use_fast=True, trust_remote_code=False, local_files_only=offline)
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer with offset mapping is required.')

    limit = tokenizer.model_max_length
    if limit < 1_000_000 and config['max_input_tokens'] > limit:
        raise ValueError(f'Configured input limit exceeds tokenizer limit {limit}')

    return tokenizer


def token_count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=True, truncation=False)['input_ids'])


def make_groups(blocks, splitter):
    """Не объединять текст через исключённый блок/страницу, границу источника или таблицу"""
    groups, current, key_previous, page_previous = [], [], None, None

    def flush():
        if current:
            groups.append(current.copy())
            current.clear()

    for block in blocks:
        if not block['retrieval_eligible']:
            flush();
            key_previous = None;
            page_previous = None
            continue
        if block['kind'] == 'heading':
            continue
        if block['kind'] == 'table':
            flush();
            groups.append([block]);
            key_previous = None;
            page_previous = None
            continue

        key = (block['source_id'], block['section_id'] if splitter == 'section' else block['chapter_id'])
        if key != key_previous or (page_previous is not None and block['pdf_page'] > page_previous + 1):
            flush()

        current.append(block)
        key_previous, page_previous = key, block['pdf_page']

    flush()

    return groups


def group_text(group):
    pieces, spans, offset = [], [], 0

    for block in group:
        if pieces:
            pieces.append('\n\n');
            offset += 2
        text = block['text']
        spans.append((offset, offset + len(text), block))
        pieces.append(text);
        offset += len(text)

    return ''.join(pieces), spans


def make_chunks(blocks, tokenizer, config, identity):
    chunks = []
    for group_number, group in enumerate(make_groups(blocks, config['splitter'])):
        first = group[0]
        header = ''
        if config['include_headings']:
            header = f"{first['chapter_id']} {first['chapter_title']}"
            if config['splitter'] == 'section' and first['section_id'] != first['chapter_id']:
                header += f" / {first['section_id']} {first['section_title']}"
            header += '\n\n'
        prefix = config['passage_prefix'] + header
        text, spans = group_text(group)

        # Группа может быть длиннее входа энкодера - здесь получаем только позиции токенов.
        # Размер каждого реального входа энкодера проверяется отдельно ниже
        encoded = tokenizer(text, add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True, verbose=False)
        offsets = encoded['offset_mapping']
        if not offsets:
            continue
        budget = config['chunk_size'] - token_count(tokenizer, prefix) - 4
        if budget <= config['overlap']:
            raise ValueError(
                'Header leaves too little space for body and overlap. Increase chunk_size or reduce overlap.')

        start = 0
        while start < len(offsets):
            end = min(len(offsets), start + budget)
            # Предпочитаем границу абзаца, если она заполняет не менее 70% окна
            if config['splitter'] == 'section' and end < len(offsets):
                boundaries = {stop for _, stop, _ in spans}
                candidates = [j + 1 for j in range(start + int(budget * 0.7), end)
                              if offsets[j][1] in boundaries]
                if candidates:
                    end = candidates[-1]

            # Проверенные таблицы сохраняем целиком, при превышении лимита выдаём ошибку
            if first['kind'] == 'table':
                if token_count(tokenizer, prefix + text) > config['chunk_size']:
                    raise ValueError(
                        f"Table on PDF page {first['pdf_page']} exceeds chunk_size; use a larger size or disable headings.")
                end = len(offsets)

            while True:
                char_start = 0 if start == 0 else offsets[start][0]
                char_end = len(text) if end == len(offsets) else offsets[end - 1][1]
                body = text[char_start:char_end]
                embedding_text = prefix + body
                count = token_count(tokenizer, embedding_text)
                if count <= config['chunk_size']:
                    break
                end -= max(1, count - config['chunk_size'])
                if end <= start:
                    raise ValueError('Could not construct a chunk within token budget.')

            evidence = []
            for lo, hi, block in spans:
                left, right = max(lo, char_start), min(hi, char_end)
                if left < right:
                    evidence.append({
                        'block_id': block['block_id'], 'page_id': block['page_id'],
                        'pdf_page': block['pdf_page'], 'printed_page': block['printed_page'],
                        'section_id': block['section_id'], 'section_title': block['section_title'],
                        'block_char_start': left - lo, 'block_char_end': right - lo,
                        'chunk_char_start': left - char_start, 'chunk_char_end': right - char_start,
                        'citation_url': block['citation_url'],
                    })

            next_start = end if end == len(offsets) else max(start + 1, end - config['overlap'])

            chunks.append({
                'chunk_id': digest([identity, group_number, start, end])[:24],
                'text': body, 'heading_context': header.strip(), 'embedding_text': embedding_text,
                'embedding_token_count': count,
                'body_token_count': len(tokenizer(body, add_special_tokens=False)['input_ids']),
                'group_index': group_number, 'group_token_start': start, 'group_token_end': end,
                'overlap_to_next_tokens': end - next_start,
                'source_id': first['source_id'], 'source_title': first['source_title'],
                'source_sha256': first['source_sha256'], 'source_path': first['source_path'],
                'source_url': first['source_url'], 'language': first['language'],
                'publication_month': first['publication_month'], 'retrieved_date': first['retrieved_date'],
                'pdf_pages': sorted({e['pdf_page'] for e in evidence}),
                'section_ids': list(dict.fromkeys(e['section_id'] for e in evidence)),
                'kind': 'table' if first['kind'] == 'table' else 'text',
                'source_spans': evidence, 'retrieval_eligible': True,
            })
            start = next_start

    if not chunks:
        raise ValueError('No eligible content.')

    return chunks


def check_chunks(chunks, blocks, config):
    """Проверить диапазоны исходника и полноту покрытия независимо от алгоритма разбиения"""
    by_id = {b['block_id']: b for b in blocks}
    coverage = defaultdict(list)

    for chunk in chunks:
        if chunk['embedding_token_count'] > config['chunk_size']:
            raise ValueError('Oversized chunk')
        for s in chunk['source_spans']:
            block = by_id[s['block_id']]
            if not block['retrieval_eligible']:
                raise ValueError('Excluded source leaked into chunks')
            if chunk['text'][s['chunk_char_start']:s['chunk_char_end']] != block['text'][
                s['block_char_start']:s['block_char_end']]:
                raise ValueError('Source span mismatch')
            coverage[s['block_id']].append((s['block_char_start'], s['block_char_end']))

    for block in blocks:
        if not block['retrieval_eligible'] or block['kind'] == 'heading':
            continue
        end = 0
        for lo, hi in sorted(coverage[block['block_id']]):
            if lo > end and block['text'][end:lo].strip():
                raise ValueError(f"Text lost before offset {lo}: {block['block_id']}")
            end = max(end, hi)
        if block['text'][end:].strip():
            raise ValueError(f"Text lost at end: {block['block_id']}")

    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Duplicate chunk IDs')


def lexical_tokens(text):
    return re.findall(r'[^\W_]+', text.casefold(), flags=re.UNICODE)


def build_bm25(chunks, config):
    lengths, postings = [], defaultdict(list)
    for i, chunk in enumerate(chunks):
        # Префиксы — инструкции embedding-модели, а не слова документа для BM25.
        terms = lexical_tokens(chunk['heading_context'] + '\n' + chunk['text'])
        lengths.append(len(terms))
        for term, count in Counter(terms).items():
            postings[term].append([i, count])

    return {'algorithm': 'BM25 positive Robertson IDF', 'k1': config['bm25_k1'],
            'b': config['bm25_b'], 'document_lengths': lengths, 'postings': dict(postings),
            'average_length': sum(lengths) / len(lengths), 'document_count': len(lengths)}


def bm25_scores(index, query):
    import numpy as np

    scores = np.zeros(index['document_count'], dtype=np.float32)
    n, avg = index['document_count'], index['average_length']

    for term in set(lexical_tokens(query)):
        postings = index['postings'].get(term, [])
        df = len(postings)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, tf in postings:
            length = index['document_lengths'][i]
            denom = tf + index['k1'] * (1 - index['b'] + index['b'] * length / max(avg, 1e-9))
            scores[i] += idf * tf * (index['k1'] + 1) / denom

    return scores


def encode_dense(texts, tokenizer, config, offline=False):
    import numpy as np
    import torch
    from transformers import AutoModel

    torch.set_num_threads(config['threads'])
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(config['embedding_model'], revision=config['model_revision'],
                                      cache_dir=ROOT / 'data/model_cache', trust_remote_code=False,
                                      local_files_only=offline,
                                      use_safetensors=True).to('cpu').eval()

    if config['max_input_tokens'] > getattr(model.config, 'max_position_embeddings', 10 ** 9):
        raise ValueError('Configured token limit exceeds encoder position capacity.')

    vectors = []

    with torch.inference_mode():
        for start in range(0, len(texts), config['batch_size']):
            batch = tokenizer(texts[start:start + config['batch_size']], padding=True,
                              truncation=False, return_tensors='pt')
            if batch['input_ids'].shape[1] > config['max_input_tokens']:
                raise ValueError('Input exceeds model limit. Truncation is intentionally disabled.')
            output = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).bool()
            pooled = output.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
            if len(texts) > config['batch_size']:
                print(f'Encoded {min(start + config["batch_size"], len(texts))}/{len(texts)}', flush=True)

    matrix = np.concatenate(vectors, axis=0)

    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
        raise ValueError('Invalid dense embeddings')

    return matrix
