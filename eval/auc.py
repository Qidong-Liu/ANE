import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def AUC(emb, positive, negative):

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def cosSim(x, y):
        vector_a = np.mat(x)
        vector_b = np.mat(y)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

    preds_pos = []
    for e in positive:
        preds_pos.append(sigmoid(np.dot(emb[str(int(e[0]))], emb[str(int(e[1]))])))
#        preds_pos.append(cosSim(emb[str(e[0])], emb[str(e[1])]))

    preds_neg = []
    for e in negative:
        preds_neg.append(sigmoid(np.dot(emb[str(int(e[0]))], emb[str(int(e[1]))])))
#        preds_neg.append(cosSim(emb[str(e[0])], emb[str(e[1])]))

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


