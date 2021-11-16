import os
from collections import defaultdict

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, roc_auc_score


class MILMetric:

   # def __init__(self, vocab, p: float, model_path):
    def __init__(self, vocab, model_path, M, N, golden_N=None):

        self.preds_labels = []
        self.p = M / N
        self.pos = vocab.get_token_index(
            "true", "labels")

        self.ep = 0
        self.path = os.path.join(model_path, "mid_result")
        os.makedirs(self.path, exist_ok=True)

    def __call__(self, preds, labels, raw_words, sent_ids):
        for idx, (pred, label, raw_word, sent_id) in \
            enumerate(zip(preds, labels, raw_words, sent_ids), len(self.preds_labels)):
            self.preds_labels.append((idx, pred, label, raw_word, sent_id))


    def calc_prf(self):
        N = len(self.preds_labels)
        M = int(self.p * N)
        sorted_data = sorted(self.preds_labels, key=lambda x: x[1], reverse=True)
        preds = [1] * M + [0] * (N-M)
        inv_preds = preds[::-1]
        if self.pos == 0:
            labels = [1-d[2] for d in sorted_data]
        else:
            labels = [d[2] for d in sorted_data]
        precision = precision_score(labels, preds, pos_label=1)
        recall = recall_score(labels, preds, pos_label=1)
        f1 = f1_score(labels, preds, pos_label=1)
        inv_precision = precision_score(labels, inv_preds, pos_label=1)
        inv_recall = recall_score(labels, inv_preds, pos_label=1)
        inv_f1 = f1_score(labels, inv_preds, pos_label=1)

        return precision, recall, f1, inv_precision, inv_recall, inv_f1

    def calc_map(self):
        sorted_data = sorted(self.preds_labels, key=lambda x: x[1], reverse=True)
        map_ = 0.0
        i = 1
        for idx, (_, _, label, _, _) in enumerate(sorted_data):
            if label==self.pos:
                map_ += i / (idx+1)
                i += 1

        inv_map = 0.0
        j = 1
        for idx, (_, _, label, _, _) in enumerate(sorted_data[::-1]):
            if label==self.pos:
                inv_map += j / (idx+1)
                j += 1
        return map_/(i-1) if (i-1) != 0 else 0, inv_map/(j-1) if (j-1) != 0 else 0

    def get_metric(self, reset=False):
        """
            应该传入sent_id用来挑选
        """
        precision, recall, f1, inv_precision, inv_recall, inv_f1 = self.calc_prf()
        #map_, inv_map = self.calc_map()
        map_ = self.map_metric()
        labels = [d[2] if self.pos==1 else 1-d[2] for d in self.preds_labels]
        scores = [d[1] for d in self.preds_labels]
        if all(labels) or not any(labels):
            auc = 0
        else:
            auc = roc_auc_score(labels, scores)
        if reset:
            with open(os.path.join(self.path, "ep{}.txt".format(self.ep)), "w", encoding="utf-8") as file:
                for _, pred, label, raw, sent_id in sorted(self.preds_labels, key=lambda x:x[1], reverse=True):
                    file.write("{}\t{}\t{}\t{}\n".format(
                            pred, label if self.pos == 1 else 1-label, " ".join(raw), sent_id
                        ))
            self.preds_labels = []
            self.ep += 1
        ret = {
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "inv_precision": inv_precision,
            "inv_recall": inv_recall,
            "inv_f1": inv_f1,
        }
        ret.update(map_)
        return ret

    def map_metric(self):
        sorted_data = sorted(
            self.preds_labels, key=lambda x: x[1], reverse=True)

        top_map = defaultdict(float)
        for k, top_k in enumerate([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
            map_ = 0.0
            i = 1
            if len(self.preds_labels) != 0 and len(self.preds_labels[0]) == 5:
                for idx, (_, p, l, r, sid) in enumerate(sorted_data[:int(len(self.preds_labels)*(top_k))]):
                    if l == self.pos:
                        map_ += i / (idx+1)
                        i += 1
            elif len(self.preds_labels) != 0:
                for idx, (_, p, l) in enumerate(sorted_data[:int(len(self.preds_labels)*(top_k))]):
                    if l == self.pos:
                        map_ += i/(idx+1)
                        i += 1
            top_map["top_{}".format(int(top_k * 100))
                    ] = map_/(i-1) if (i-1) != 0 else 0

        return top_map
