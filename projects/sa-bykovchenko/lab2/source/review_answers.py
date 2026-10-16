"""Создание формы оценки QA и сводки по заполненным оценкам"""
import argparse
import json
from pathlib import Path
import index_core as core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qa', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--summarize', type=Path)
    a = p.parse_args()
    if a.summarize:
        d = core.read_json(a.summarize)
        if not d.get('reviewer'):
            p.error('Fill reviewer (human or model-assisted, honestly) before summarizing')
        for row in d['assessments']:
            if row['correctness'] not in [0, 1, 2] or row['completeness'] not in [0, 1, 2] or row[
                'citations_supported'] not in [True, False] or not row['comment']:
                p.error('Finish every assessment and explain judgments before summarizing')
        for model in sorted({x['model'] for x in d['assessments']}):
            rows = [x for x in d['assessments'] if x['model'] == model and x['id'] != 'REFUSAL']
            print(model, {'n': len(rows), 'fully_correct_rate': sum(x['correctness'] == 2 for x in rows) / len(rows),
                          'complete_rate': sum(x['completeness'] == 2 for x in rows) / len(rows),
                          'supported_citations_rate': sum(x['citations_supported'] for x in rows) / len(rows)})
        return
    if not a.qa or not a.output:
        p.error('Provide --qa and --output, or --summarize')
    if a.output.exists():
        p.error('Review file already exists; choose another path to preserve judgments')
    qrels = {q['id']: q for q in core.read_json(core.ROOT / 'data/qrels.json')['questions']}
    data = core.read_json(a.qa)
    rows = []
    for model, result in data['models'].items():
        for answer in result['answers']:
            rows.append({'model': model, 'id': answer['id'], 'question': answer['question'], 'answer': answer['answer'],
                         'expected_answer': qrels.get(answer['id'], {}).get('expected_answer_ru',
                                                                            'Отказ: вопрос вне корпуса'),
                         'correctness': None, 'completeness': None, 'citations_supported': None, 'comment': ''})
    form = {'reviewer': '', 'qa_path': str(a.qa), 'qa_sha256': core.file_hash(a.qa), 'rubric': {
        'correctness': '0: incorrect/unsupported or refusal to answer an answerable corpus question; 1: partly correct; 2: all factual claims correct. Probe: 2 for appropriate refusal.',
        'completeness': '0: no requested facts; 1: some; 2: all. Assess against source and question, not only short reference answer.',
        'citations_supported': 'true only if all factual claims have valid citations whose actual provided text supports them; correct refusal can be true. Page overlap alone is insufficient.',
        'comment': 'Explain errors, missing facts, source PDF pages and whether failure came from retrieval or generation. Inspect saved context and original source.'},
            'assessments': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    core.write_json(a.output, form)
    print(a.output)


if __name__ == '__main__':
    main()
