"""Проверка, что все evidence-цитаты реально существуют в нормализованном корпусе на указанных страницах"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_core as core  # noqa: E402

ROOT = core.ROOT


def squash(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', text.casefold())


def verify(qrels_path: Path, pages_path: Path) -> tuple[int, int, list[str]]:
    data = core.read_json(qrels_path)
    pages = {p['pdf_page']: p for p in core.read_jsonl(pages_path)}
    total = passed = 0
    problems: list[str] = []

    for question in data['questions']:
        if not question.get('evidence'):
            continue
        for index, ev in enumerate(question['evidence']):
            total += 1
            page = pages.get(ev['page'])
            if page is None:
                problems.append(f"{question['id']}: страницы {ev['page']} нет в корпусе")
                continue
            if squash(ev['quote']) in squash(page['text']):
                passed += 1
            else:
                problems.append(f"{question['id']}.evidence[{index}]: цитата не найдена "
                                f"на стр. {ev['page']}: {ev['quote'][:80]!r}...")

    # структурные проверки
    ids = [q['id'] for q in data['questions']]
    if len(set(ids)) != len(ids):
        problems.append('дублирующиеся id вопросов')
    splits = [q['split'] for q in data['questions']]
    for split in ('dev', 'test'):
        if split not in splits:
            problems.append(f'нет вопросов со split={split}')
    for question in data['questions']:
        for fact in question.get('facts', []):
            for ref in fact.get('evidence', []):
                if not 0 <= ref < len(question.get('evidence', [])):
                    problems.append(f"{question['id']}: fact {fact['id']} ссылается на "
                                    f"несуществующий evidence[{ref}]")
    return total, passed, problems


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qrels', type=Path, default=ROOT / 'data/qrels_evidence.json')
    p.add_argument('--processed', type=Path,
                   help='Каталог сборки парсера; по умолчанию latest.json')
    args = p.parse_args()

    processed = args.processed or (
        ROOT / 'data/processed' / core.read_json(ROOT / 'data/processed/latest.json')['path'])
    pages_path = processed / 'pages.jsonl'

    total, passed, problems = verify(args.qrels, pages_path)
    print(f'Цитат проверено: {total}; найдено на своих страницах: {passed}')
    for problem in problems:
        print(f'  ПРОБЛЕМА: {problem}')
    if problems:
        print('РЕЗУЛЬТАТ: провал, исправьте qrels или проверьте сборку парсера.')
        return 1
    print('РЕЗУЛЬТАТ: все цитаты подтверждены корпусом.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
