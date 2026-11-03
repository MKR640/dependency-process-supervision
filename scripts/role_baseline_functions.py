"""Role metrics for rows sorted by trajectory and step."""
import numpy as np

def score(y, p, classes=(0,1,2)):
    rows = []
    for c in classes:
        tp = sum(a == c and b == c for a,b in zip(y,p))
        fp = sum(a != c and b == c for a,b in zip(y,p))
        fn = sum(a == c and b != c for a,b in zip(y,p))
        rows.append({'class':c,'support':sum(a==c for a in y), 'f1':2*tp/max(1,2*tp+fp+fn)})
    return {'macro_f1':sum(r['f1'] for r in rows)/len(rows), 'classes':rows}

def sequential_roles(rows, threshold):
    errors = np.array([x['p_error'] >= threshold for x in rows])
    starts = np.array([i for i,r in enumerate(rows) if i==0 or r['trace_id']!=rows[i-1]['trace_id']])
    lens = np.diff(np.append(starts,len(rows)))
    cumulative = np.cumsum(errors)
    offset = np.repeat(cumulative[starts]-errors[starts],lens)
    return np.where(errors,np.where(cumulative-offset==1,1,2),0)

def confusion(y,p):
    return np.bincount(np.asarray(y)*3+np.asarray(p),minlength=9).reshape(3,3)

def macro_from_cm(c):
    tp=np.diagonal(c,axis1=-2,axis2=-1)
    den=c.sum(axis=-1)+c.sum(axis=-2)
    return np.mean(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0),axis=-1)
