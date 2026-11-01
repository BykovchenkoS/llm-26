import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from audit_evidence import ROOT, read, validate, matches, compare

class EvidenceStage1Tests(unittest.TestCase):
    def setUp(self):
        self.data={'_meta':{'source_id':'s'},'questions':[{'id':'q','split':'dev','answerable':True,
            'facts':[{'id':'f','text':'foil exposure','evidence':[0]}],
            'evidence':[{'page':1,'quote':'The foil was retrieved after 77 minutes'}]}]}
    def test_new_annotation_valid(self):
        validate(read(ROOT/'data/qrels_evidence.json'))
    def test_questions_and_splits_unchanged(self):
        a=read(ROOT/'data/qrels_evidence.json')['questions']
        b=read(ROOT/'data/history/qrels_evidence_v3.json')['questions']
        self.assertEqual([(q['id'],q['split'],q['question_ru'],q['question_en']) for q in a],
                         [(q['id'],q['split'],q['question_ru'],q['question_en']) for q in b])
    def test_missing_evidence_rejected(self):
        self.data['questions'][0]['facts'][0]['evidence']=[]
        with self.assertRaises(ValueError):validate(self.data)
    def test_negative_reference_rejected(self):
        self.data['questions'][0]['facts'][0]['evidence']=[-1]
        with self.assertRaises(ValueError):validate(self.data)
    def test_empty_quote_rejected(self):
        self.data['questions'][0]['evidence'][0]['quote']='---'
        with self.assertRaises(ValueError):validate(self.data)
    def test_page_and_source_required(self):
        e=self.data['questions'][0]['evidence'][0]
        c={'chunk_id':'x','text':e['quote'],'pdf_page':2,'source_id':'s'}
        self.assertFalse(matches(c,e,'s'))
        c.update(pdf_page=1,source_id='other');self.assertFalse(matches(c,e,'s'))
    def test_split_quote_not_counted_as_complete(self):
        chunks=[{'chunk_id':'1','source_id':'s','pdf_page':1,'text':'The foil was retrieved'},
                {'chunk_id':'2','source_id':'s','pdf_page':1,'text':'after 77 minutes'}]
        pages={('s',1):{'text':'The foil was retrieved after 77 minutes'}}
        result=compare(self.data,chunks,pages)['aggregate']['all']
        self.assertEqual(result['facts_available'],0)
        self.assertEqual(result['split_or_chunking_loss'],1)
    def test_parser_loss_is_distinct(self):
        result=compare(self.data,[],{('s',1):{'text':'garbled OCR'}})['aggregate']['all']
        self.assertEqual(result['parser_text_mismatch'],1)
        self.assertEqual(result['split_or_chunking_loss'],0)

if __name__=='__main__':unittest.main()
