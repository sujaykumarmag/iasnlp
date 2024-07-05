'''Acc Evaluator'''
from src.openicl.icl_evaluator.icl_base_evaluator import IclBaseEvaluator
from typing import List
import evaluate

class IclAccEvaluator(IclBaseEvaluator):
    def __init__(self, references: List) -> None:
        super().__init__(references)
        
        
    def score(self, predictions, references=None, src_lang=None, tgt_lang=None,  output_dir=None):
        if references == None:
            references = self.references
        assert len(predictions) == len(references)
        mapping_to_int_dict = {label: idx for idx, label in enumerate(set(references))}
        pred_set = set(predictions)
        for pred in pred_set:
            if pred not in mapping_to_int_dict.keys():
                mapping_to_int_dict[pred] = len(mapping_to_int_dict)
        golds = [mapping_to_int_dict[gold] for gold in references]
        preds = [mapping_to_int_dict[pred] for pred in predictions]
        metric = evaluate.load("accuracy")
        return metric.compute(references=golds, predictions=preds)
    