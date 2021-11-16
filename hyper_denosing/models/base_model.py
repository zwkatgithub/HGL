import abc

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from hyper_denosing.metrics import PositiveMetric
from hyper_denosing.metrics.mil_metric import MILMetric

class BaseModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 hidden_size: int,
                 N: int,
                 M: int,
                 golden_N: int,
                 batch_size: int,
                 model_path: str,
                 T: float):
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.post_layer = torch.nn.Linear(
            self.text_field_embedder.get_output_dim(),
            self.text_field_embedder.get_output_dim()
        )
        self.batch_size = batch_size
        self.extractor = SelfAttentiveSpanExtractor(
            self.text_field_embedder.get_output_dim())
        self.score = torch.nn.Sequential(
            torch.nn.Linear(self.extractor.get_output_dim(), hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )
        self.T = T
        self.loss = self.build_loss()
        
        # self.metric = PositiveMetric(
        #     vocab, model_path, M, N, golden_N)
        self.metric = MILMetric(vocab, model_path, M, N)
        self.M = M
        self.N = N
        self.golden_N = golden_N

    @abc.abstractmethod
    def build_loss(self):
        raise NotImplementedError("This method is not been implemented.")

    def forward(self, text, spans, raw_words, sent_ids, plabels, labels=None):

        output = {}
        mask = get_text_field_mask(text)
        spans = spans.unsqueeze(-2)
        embedding = self.text_field_embedder(text)
        #ent_rep = self.extractor(embedding, spans, mask).squeeze(-2)
        hidden = self.post_layer(embedding)
        ent_rep = self.extractor(hidden, spans, mask).squeeze(-2)
        scores = self.score(ent_rep).squeeze(-1)
        output['scores'] = scores
        scores = torch.pow(scores, 1/self.T)
        loss_v = self.loss(scores, eps=1e-10)
        output['loss'] = loss_v

        if labels is not None:
            self.metric(scores.tolist(), labels.tolist(),
                         raw_words, sent_ids)
        return output

    def get_metrics(self, reset=False):
        ret = self.metric.get_metric(reset)
        return ret
