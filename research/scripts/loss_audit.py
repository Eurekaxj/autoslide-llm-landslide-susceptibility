"""Diagnostic refits only; outer held-out records never select a model or a stopping rule."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import exp_lib as L
B=ROOT/'data'
OUT=ROOT/'results';OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(B/'modeling_base.csv');best=pd.read_csv(ROOT/'results/03_最优超参数.csv')
rows=[]
class RecordedMLP(MLPClassifier):
    def _update_no_improvement_count(self,*args,**kwargs):
        super()._update_no_improvement_count(*args,**kwargs)
        self.audit_history.append((log_loss(self.audit_ytr,self.predict_proba(self.audit_xtr)),
                                   log_loss(self.audit_yte,self.predict_proba(self.audit_xte))))
for model in ['XGBoost','CatBoost','MLP']:
    for _,r in best[(best.group=='G2')&(best.model==model)].iterrows():
        k=int(r.fold);tr=d.fold!=k;te=d.fold==k
        native=model=='CatBoost';build=L.matrix_for_categorical_model if native else L.build_matrix
        a,e=build(d[tr],L.GROUPS['G2'],fit=True);z,_=build(d[te],L.GROUPS['G2'],enc=e,fit=False)
        y=d.loc[tr,'landslide'].astype(int);yt=d.loc[te,'landslide'].astype(int)
        p=json.loads(r.params)
        if model=='XGBoost':
            p['n_estimators']=300;m=L.instantiate(model,p,42,n_jobs=1)
            m.fit(a,y,eval_set=[(a,y),(z,yt)],verbose=False)
            h=m.evals_result();train=h['validation_0']['logloss'];test=h['validation_1']['logloss']
        elif native:
            p['iterations']=300;m=L.instantiate(model,p,42,n_jobs=1)
            ci=[list(a.columns).index(c)for c in e['cat_cols']]
            m.fit(a,y,cat_features=ci,eval_set=(z,yt),use_best_model=False,verbose=False)
            h=m.get_evals_result();train=h['learn']['Logloss'];test=h['validation']['Logloss']
        else:
            m=RecordedMLP(**p,random_state=42,max_iter=300,early_stopping=False,n_iter_no_change=1000,tol=0)
            m.audit_xtr=a;m.audit_ytr=y;m.audit_xte=z;m.audit_yte=yt;m.audit_history=[]
            m.fit(a,y);train,test=zip(*m.audit_history)
        assert len(train)==len(test)==300
        rows.extend(dict(model=model,fold=k,step=i+1,train_loss=float(v),test_loss=float(w))for i,(v,w)in enumerate(zip(train,test)))
    print('done',model,flush=True)
q=pd.DataFrame(rows);q.to_csv(OUT/'loss_diagnostic_300.csv',index=False)
summary=[]
for model,g in q.groupby('model'):
    h=g.groupby('step')[['train_loss','test_loss']].mean();v=h.test_loss
    budget=200 if model!='MLP' else 300
    rel=float((v.loc[budget]-v.loc[int(.8*budget)])/v.loc[int(.8*budget)])
    best_value=float('inf');wait=0;stop=None
    for step,val in v.items():
        if val<best_value-1e-4:best_value=val;wait=0
        else:wait+=1
        if wait>=50:stop=int(step);break
    summary.append(dict(model=model,diagnostic_horizon=300,reference_step=budget,best_step=int(v.idxmin()),
                        best_loss=float(v.min()),loss_at_reference=float(v.loc[budget]),
                        tail_relative_change=rel,plateau_endpoint_lt_1pct=abs(rel)<.01,
                        diagnostic_patience_stop=stop,loss_at_300=float(v.loc[300])))
pd.DataFrame(summary).to_csv(OUT/'loss_diagnostic_summary.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
