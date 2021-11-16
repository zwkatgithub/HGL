import os
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from hyper_denosing.metrics.base_metric import BaseMetric


class PositiveMetric(BaseMetric):

    def get_metric(self, reset=False):
        N = len(self.pred_labels)
        M = int(N*self.p)
        pred_labels = sorted(
            self.pred_labels, key=lambda x: x[1], reverse=True)
        preds = [1] * M + [0] * (N-M)
        labels = [d[2] if self.pos == 1 else 1-d[2] for d in pred_labels]

        precision, recall, f1 = self.prf(labels, preds)

        top_map = defaultdict(float)

        for _, top_k in enumerate([d/10 for d in range(1, 11)]):
            m = 0.0
            it = 1
            if len(labels) != 0:
                for idx, label in enumerate(labels[:int(N*top_k)]):
                    if (self.pos == 1 and label == self.pos) or (self.pos == 0 and (1-label) == self.pos):
                        m += it / (idx+1)
                        it += 1
            top_map["top_{}".format(int(top_k*100))] = m / \
                (it-1) if it > 1 else 0

        scores = [d[1] for d in pred_labels]
        if all(labels) or not any(labels):
            auc = 0
        else:
            auc = roc_auc_score(labels, scores)

        if reset:
            with open(os.path.join(self.path, "{}".format(self.ep)), "w", encoding="utf-8") as fout:
                for _, pred, label, _, raw_words, sent_id in pred_labels:
                    fout.write("{}\t{}\t{}\t{}\n".format(pred,
                                                         label if self.pos == 1 else 1-label, " ".join(raw_words), sent_id))
            self.pred_labels = []
            self.inpreds = set()
            self.ep += 1

        ret = {
            "auc": auc,
            "pre": precision,
            "rec": recall,
            "f1": f1
        }

        ret.update(top_map)
        return ret
