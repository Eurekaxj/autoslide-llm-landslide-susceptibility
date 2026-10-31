import sys,json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import exp_lib as L
B=ROOT/'data'
d=pd.read_csv(B/'modeling_base.csv'); bp=pd.read_csv(ROOT/'results/03_最优超参数.csv')
oof=pd.read_csv(ROOT/'results/03_OOF预测.csv'); rows=[]
for _,r in bp[(bp.group=='G2')&(bp.model=='MLP')].iterrows():
    k=int(r.fold);tr=d.fold!=k;te=d.fold==k
    a,e=L.build_matrix(d[tr],L.GROUPS['G2'],fit=True)
    z,_=L.build_matrix(d[te],L.GROUPS['G2'],enc=e,fit=False)
    m=L.instantiate('MLP',json.loads(r.params),42,n_jobs=1)
    m.fit(a,d.loc[tr,'landslide'].astype(int))
    pred=m.predict_proba(z)[:,1]
    old=oof[(oof.group=='G2')&(oof.model=='MLP')&(oof.fold==k)].set_index('pointid').oof
    delta=float(np.max(np.abs(pred-old.reindex(d.loc[te,'pointid']).to_numpy())))
    v=np.array(m.validation_scores_)
    rows.append(dict(fold=k,n_iter=m.n_iter_,max_iter=m.max_iter,patience=m.n_iter_no_change,tol=m.tol,
                     internal_validation_fraction=m.validation_fraction,best_validation_score=m.best_validation_score_,
                     stopped_before_cap=bool(m.n_iter_<m.max_iter),test_auc=roc_auc_score(d.loc[te,'landslide'],pred),
                     max_abs_diff_archived=delta))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(ROOT/'results/mlp_stopping_audit.csv',index=False)
