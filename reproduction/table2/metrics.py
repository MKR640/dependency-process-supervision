"""Numerical functions retained from the archived shared-label report."""
import numpy as np

def weighted_ap(y, scores, w):
    order = np.argsort(-scores, kind='stable')
    y, scores, w = y[order], scores[order], w[order]
    positives = np.sum(w*y)
    if positives == 0 or np.sum(w*(1-y)) == 0:
        return None
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores)-1]
    tp, total = np.cumsum(w*y)[ends], np.cumsum(w)[ends]
    precision = np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)
    return float(np.sum(np.diff(np.r_[0.,tp])/positives * precision))

def threshold(y, scores):
    if set(y) != {0,1}:
        raise ValueError('Validation must include both classes')
    order = np.argsort(-scores,kind='stable')
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    tp = np.cumsum(ys)[ends]
    ba = .5*(tp/y.sum() + 1-(ends+1-tp)/(len(y)-y.sum()))
    ts = ss[ends]
    half = .5*((scores[y==1]>=.5).mean()+(scores[y==0]<.5).mean())
    ts, ba = np.r_[ts,.5], np.r_[ba,half]
    best = np.flatnonzero(np.isclose(ba,ba.max(),rtol=0,atol=1e-12))
    i = min(best, key=lambda j:(abs(ts[j]-.5),ts[j]))
    return float(ts[i])

def metrics(y, score, t, w):
    pred = score>=t
    tp, fp, fn, tn = [float(w[m].sum()) for m in
                      ((y==1)&pred,(y==0)&pred,(y==1)&~pred,(y==0)&~pred)]
    ratio = lambda x,d: x/d if d else None
    return dict(error_ap=weighted_ap(y,score,w), error_f1=ratio(2*tp,2*tp+fp+fn),
        balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)) if (tp+fn)*(tn+fp) else None,
        error_recall=ratio(tp,tp+fn), correct_recall=ratio(tn,tn+fp))
