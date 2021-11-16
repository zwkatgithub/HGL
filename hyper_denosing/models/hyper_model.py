import torch
from itertools import accumulate
import scipy.stats as st

from allennlp.models import Model

from hyper_denosing.models.base_model import BaseModel

memory = dict()

@Model.register("hyper_model")
class HyperModel(BaseModel):

    def generate_hypergeo_dist(self, N, M, B):
        return [st.hypergeom.pmf(k, N, M, min(B, N-M)) for k in range(B+1)]

    def build_loss(self):

        self.dist = None
        self.q = None

        def loss(scores, **kwargs):
            eps = kwargs['eps']
            
            if self.dist is None:
                self.dist = self.generate_hypergeo_dist(self.N, self.M, self.batch_size)
                self.dist = torch.tensor(list(accumulate(self.dist[::-1]))[::-1]).to(scores.device)
                self.q = torch.tensor(
                        st.hypergeom.pmf(0, self.N, self.M, min(self.N-self.M, self.batch_size))).to(scores.device)
            num = scores.shape[0]
            scores_, _ = torch.sort(scores, dim=-1, descending=True)
            return -torch.sum(self.dist[:num]*torch.log(scores_+eps)+
                                 (1-self.dist[:num]-self.q)*torch.log(1-scores_+eps))
        
        return loss
