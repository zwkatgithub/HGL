from typing import Dict
from tqdm import tqdm
import abc
import json
from collections import defaultdict

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField, LabelField, SpanField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

from hyper_denosing.utils import lines_generator, spans_generator, convert_hat

class BaseDatasetReader(DatasetReader):

    def __init__(self, token_indexers, type_, 
                 start_tag='<E>', end_tag='</E>', max_span_length=6, 
                 lazy=False):
        super().__init__(lazy)
        self.token_indexers = token_indexers
        self.type_ = type_
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.max_span_length = max_span_length
        self.N, self.M = 0, 0
        self.golden_N = 0

    def _read(self, file_path: str):

        with open(file_path, "r", encoding="utf-8") as fin:
            for sent_id, lines in enumerate(lines_generator(fin)):
                words = lines[0].split()
                golden_spans = [(span[0], span[1]) for span in spans_generator(lines[2])
                                if span[2] == self.type_ and span[1]-span[0] < self.max_span_length]
                pred_spans = list(set([(span[0], span[1]) for span in spans_generator(lines[1])
                                       if span[2]==self.type_ and span[1]-span[0] < self.max_span_length]))
                
                self.golden_N += len(golden_spans)

                if self.__class__.__name__.startswith("Positive"):
                    for start, end in pred_spans:
                        self.N += 1
                        if (start, end) in golden_spans:
                            self.M += 1
                            yield self.text_to_instance(words, start, end-1, sent_id, 'true', 'true')
                        else:
                            yield self.text_to_instance(words, start, end-1, sent_id, 'true', 'false')
    
    def text_to_instance(self, words, start, end, sent_id, plabel, label=None) -> Instance:
        fields = {}

        words, start, end = convert_hat(
            words, start, end, self.start_tag, self.end_tag
        )
        text_field = TextField([Token(token) for token in words], self.token_indexers)

        fields['text'] = text_field
        fields['spans'] = SpanField(start, end, text_field)
        fields['plabels'] = LabelField(plabel, label_namespace='plabels')
        if label is not None:
            fields['labels'] = LabelField(label)
        fields['raw_words'] = MetadataField(words)
        fields['sent_ids'] = MetadataField(sent_id)

        return Instance(fields)

