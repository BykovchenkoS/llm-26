from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_core as core  # noqa: E402

ROOT = core.ROOT

RUBRIC = {
    'correctness': ('0: неверно/не подтверждено источником, либо отказ на вопрос, '
                    'на который в корпусе есть ответ; 1: частично верно; '
                    '2: все фактические утверждения верны. Для unanswerable '
                    '(answerable=false): 2 = корректный отказ.'),
    'completeness': ('0: ни одного запрошенного факта; 1: часть; 2: все факты. '
                     'Для unanswerable: 2 при корректном отказе, 0 при выдуманном ответе. '
                     'Сверяйте с expected_facts в qrels_evidence.json и исходным PDF, '
                     'а не только с кратким expected_answer.'),
    'citations_supported': ('true, только если ВСЕ фактические утверждения ответа '
                           'подтверждаются текстом цитируемых фрагментов (проверьте '
                           'context в qa.json и оригинал). Корректный отказ может быть true '
                           'без цитат. Страницы сами по себе — не доказательство.'),
    'comment': 'Объясните ошибки: этап (retrieval/контекст/генерация/цитирование), '
               'страницы источника, недостающие факты.',
}


def validate_review(data, qa, qa_path):
    """Не допускаем пустую оценку, пропуски строк и подмену исходных ответов"""
    if not isinstance(data.get('reviewer'), str) or not data['reviewer'].strip():
        raise ValueError('Заполните reviewer.')
    if data.get('qa_sha256') != core.file_hash(qa_path):
        raise ValueError('qa.json изменён или выбран другой прогон.')
    expected = {(model, a['id']): a for model, result in qa['models'].items()
                for a in result['answers']}
    rows = data.get('assessments', [])
    keys = [(r['model'], r['id']) for r in rows]
    if not expected or len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError('Неполный/дублирующийся набор ручных оценок.')
    for row in rows:
        original = expected[(row['model'], row['id'])]
        if row['answer'] != original['answer'] or row['question'] != original['question']:
            raise ValueError('Изменён оцениваемый ответ/вопрос.')
        if row.get('answerable') is not original.get('answerable'):
            raise ValueError('Изменена метка answerable.')
        if (type(row.get('correctness')) is not int or row['correctness'] not in (0, 1, 2)
                or type(row.get('completeness')) is not int or row['completeness'] not in (0, 1, 2)
                or type(row.get('citations_supported')) is not bool
                or not isinstance(row.get('comment'), str) or not row['comment'].strip()):
            raise ValueError(f'Незавершённая оценка: {row["model"]}/{row["id"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qa', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--summarize', type=Path)
    a = p.parse_args()

    if a.summarize:
        d = core.read_json(a.summarize)
        qa_path = a.qa or Path(d['qa_path'])
        try:
            validate_review(d, core.read_json(qa_path), qa_path)
        except (ValueError, OSError) as exc:
            p.error(str(exc))
        for model in sorted({x['model'] for x in d['assessments']}):
            rows = [x for x in d['assessments'] if x['model'] == model]
            answerable = [x for x in rows if x.get('answerable')]
            unanswerable = [x for x in rows if not x.get('answerable')]
            n = len(answerable) or 1
            print(model, {
                'n_total': len(rows),
                'answerable': len(answerable),
                'fully_correct_rate': round(sum(x['correctness'] == 2 for x in answerable) / n, 4),
                'complete_rate': round(sum(x['completeness'] == 2 for x in answerable) / n, 4),
                'supported_citations_rate': round(
                    sum(x['citations_supported'] for x in answerable) / n, 4),
                'correct_refusal_rate': round(
                    sum(x['correctness'] == 2 for x in unanswerable) / len(unanswerable), 4)
                if unanswerable else None,
            })
        return

    if not a.qa or not a.output:
        p.error('Укажите --qa и --output, либо --summarize')
    if a.output.exists():
        p.error('Файл оценки уже существует; выберите другой путь, чтобы сохранить суждения.')

    qrels = {q['id']: q for q in core.read_json(ROOT / 'data/qrels_evidence.json')['questions']}
    data = core.read_json(a.qa)
    if data.get('status') != 'completed':
        p.error('Нужен завершённый qa.json, не partial.')
    if data.get('qrels_sha256') != core.file_hash(ROOT / 'data/qrels_evidence.json'):
        p.error('QA и текущая разметка qrels имеют разные версии.')
    rows = []
    for model, result in data['models'].items():
        for answer in result['answers']:
            question = qrels[answer['id']]
            rows.append({
                'model': model,
                'id': answer['id'],
                'answerable': question.get('answerable'),
                'question': answer['question'],
                'answer': answer['answer'],
                'status': answer.get('status'),
                'cited_pages': answer.get('cited_pages'),
                'invalid_citation_ids': answer.get('invalid_citation_ids'),
                'context': answer.get('context', ''),
                'sources': answer.get('sources', []),
                'parse_error': answer.get('parse_error'),
                'llm_finish_reason': answer.get('llm_finish_reason'),
                'evidence': question.get('evidence', []),
                'expected_answer_ru': question.get('expected_answer_ru', ''),
                'expected_facts': [f['text'] for f in question.get('facts', [])],
                'correctness': None, 'completeness': None,
                'citations_supported': None, 'comment': '',
            })
    form = {'reviewer': '', 'qa_path': str(a.qa.resolve()), 'qa_sha256': core.file_hash(a.qa),
            'qrels_sha256': data['qrels_sha256'], 'rubric': RUBRIC, 'assessments': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    core.write_json(a.output, form)
    print(a.output)


if __name__ == '__main__':
    main()
