import os

class BaseMetric:

    def __init__(self, vocab, model_path, M, N, golden_N):
        self.M = M
        self.N = N
        self.p = self.M / self.N
        self.golden_N = golden_N

        self.pred_labels = []
        try:
            self.ppos = vocab.get_token_index("true", "plabels")
        except KeyError:
            self.ppos = 1
        
        self.pos = vocab.get_token_index("true", "labels")
        self.ep = 0
        self.path = os.path.join(model_path, "mid_result")
        self.inpreds = set()
        os.makedirs(self.path, exist_ok=True)

    def prf(self, labels, preds):
        tp, fp, tn, fn = 0, 0, 0, 0
        for idx in range(len(labels)):
            if labels[idx]==preds[idx]:
                if labels[idx]==self.pos: tp += 1
                else: tn += 1
            else:
                if labels[idx]==self.pos: fn += 1
                else: fp += 1
        p = tp / (tp+fp) if (tp+fp) > 0 else 0
        r = tp / self.golden_N
        f = 2*p*r / (p+r) if (p+r) > 0 else 0
        return p, r, f

    def __call__(self, preds, labels, plabels, raw_words, sent_ids):
        for idx, (pred, label, plabel, raw_word, sent_id) in enumerate(
            zip(preds, labels, plabels, raw_words, sent_ids)
            ):
            if (sent_id, " ".join(raw_word), plabel, label) not in self.inpreds:
                self.pred_labels.append((idx, pred, label, plabel, raw_word, sent_id))
                self.inpreds.add((sent_id, " ".join(raw_word), plabel, label))
    