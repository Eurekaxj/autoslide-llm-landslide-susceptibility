"""Spatial-block conditional uncertainty; no uncalibrated p/q or CV-fold t tests."""
import json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import exp_lib as L


def weighted_bootstrap(y,P,blocks,n_boot=2000,seed=42):
    """Exact block-multiplicity AUC/AP, retaining score ties and within-block records."""
    y=np.asarray(y,dtype=int);P=np.asarray(P,float)
    labels,block_idx=np.unique(np.asarray(blocks,dtype=str),return_inverse=True)
    draws=np.random.default_rng(seed).integers(0,len(labels),size=(n_boot,len(labels)))
    counts=np.zeros((n_boot,len(labels)),float)
    np.add.at(counts,(np.arange(n_boot)[:,None],draws),1.)
    aucs=np.empty((n_boot,P.shape[1]));aps=np.empty_like(aucs)
    for j in range(P.shape[1]):
        values,groups=np.unique(P[:,j],return_inverse=True)
        pos=np.zeros((len(labels),len(values)));neg=np.zeros_like(pos)
        np.add.at(pos,(block_idx[y==1],groups[y==1]),1)
        np.add.at(neg,(block_idx[y==0],groups[y==0]),1)
        wp=counts@pos;wn=counts@neg;np_=wp.sum(axis=1);nn=wn.sum(axis=1)
        aucs[:,j]=(wp*(np.cumsum(wn,axis=1)-.5*wn)).sum(axis=1)/(np_*nn)
        cp=np.cumsum(wp[:,::-1],axis=1);cn=np.cumsum(wn[:,::-1],axis=1)
        aps[:,j]=((wp[:,::-1]/np_[:,None])*np.divide(cp,cp+cn,out=np.zeros_like(cp),where=(cp+cn)>0)).sum(axis=1)
    return aucs,aps,draws,labels,counts,block_idx


def paired_table(pv,y,boot,reference=None):
    pairs=itertools.combinations(range(len(pv.columns)),2) if reference is None else [(list(pv.columns).index(reference),j) for j in range(len(pv.columns)) if pv.columns[j]!=reference]
    rows=[]
    for i,j in pairs:
        delta=boot[:,i]-boot[:,j];lo,hi=np.percentile(delta,[2.5,97.5])
        rows.append({'A':pv.columns[i],'B':pv.columns[j],
                     'OOF_AUC_A':roc_auc_score(y,pv.iloc[:,i]),'OOF_AUC_B':roc_auc_score(y,pv.iloc[:,j]),
                     'delta_A_minus_B':roc_auc_score(y,pv.iloc[:,i])-roc_auc_score(y,pv.iloc[:,j]),
                     'conditional_ci95_low':lo,'conditional_ci95_high':hi,
                     'conditional_interval_excludes_zero':bool(lo>0 or hi<0),'n_boot':len(delta),
                     'interpretation':'fixed-OOF spatial-block conditional interval; no null-hypothesis p/q or equivalence claim'})
    return pd.DataFrame(rows)


def main():
    R=Path(L.RES);D=Path(L.DATA);b=L.load_base().set_index('pointid')
    o=pd.read_csv(R/'03_OOF预测.csv');o=o[o.group=='G2']
    pv=o.pivot(index='pointid',columns='model',values='oof').sort_index()
    pv=pv[[m for m in L.MODELS if m in pv]];y=b.landslide.reindex(pv.index).to_numpy(dtype=int)
    blocks=b.spatial_block_id.reindex(pv.index).to_numpy()
    boot,ap,draws,labels,counts,block_idx=weighted_bootstrap(y,pv.values,blocks)
    pd.DataFrame(boot,columns=pv.columns).to_csv(R/'15_重采样AUC矩阵.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(ap,columns=pv.columns).to_csv(R/'15_重采样AP矩阵.csv',index=False,encoding='utf-8-sig')
    np.save(D/'15_bootstrap_AUC.npy',boot);np.save(D/'15_block_draw_indices.npy',draws)
    (D/'15_block_labels.json').write_text(json.dumps(labels.tolist()),encoding='utf-8')
    paired=paired_table(pv,y,boot);paired.to_csv(R/'15_bootstrap差值.csv',index=False,encoding='utf-8-sig')
    metrics=[];distribution=[]
    for j,m in enumerate(pv.columns):
        z=o[o.model==m].set_index('pointid').reindex(pv.index)
        sums=np.zeros((len(labels),5))
        obs=np.column_stack([((y==1)&(z.yhat.values==1)).astype(float),
                             ((y==0)&(z.yhat.values==1)).astype(float),
                             ((y==1)&(z.yhat.values==0)).astype(float),
                             (z.oof.values-y)**2,np.ones(len(y))])
        np.add.at(sums,block_idx,obs);v=counts@sums
        f1=2*v[:,0]/(2*v[:,0]+v[:,1]+v[:,2]);brier=v[:,3]/v[:,4]
        for key,vec,point in [('roc_auc',boot[:,j],roc_auc_score(y,pv[m])),
                             ('average_precision',ap[:,j],L.metrics_at(y,pv[m])['pr_auc']),
                             ('f1_fold_threshold',f1,2*obs[:,0].sum()/(2*obs[:,0].sum()+obs[:,1:3].sum())),
                             ('brier',brier,obs[:,3].mean())]:
            lo,hi=np.percentile(vec,[2.5,97.5]);metrics.append({'model':m,'metric':key,'pooled_estimate':point,
                'conditional_ci95_low':lo,'conditional_ci95_high':hi,'n_records':len(y),'n_spatial_blocks':len(labels),'n_boot':len(vec)})
            distribution.append(pd.DataFrame({'bootstrap_id':np.arange(len(vec)),'model':m,'metric':key,'value':vec}))
    pd.DataFrame(metrics).to_csv(R/'15_合并OOF条件区间.csv',index=False,encoding='utf-8-sig')
    pd.concat(distribution,ignore_index=True).to_csv(R/'15_指标重采样明细.csv',index=False,encoding='utf-8-sig')
    a=pd.read_csv(R/'05_消融OOF.csv');av=a.pivot(index='pointid',columns='variant',values='oof').reindex(pv.index)
    ab,_,_,_,_,_=weighted_bootstrap(y,av.values,blocks)
    pd.DataFrame(ab,columns=av.columns).to_csv(R/'05_消融重采样AUC矩阵.csv',index=False,encoding='utf-8-sig')
    at=pd.concat([paired_table(av,y,ab,ref).assign(baseline=ref) for ref in ['G2_full','G1_rx5day_cv','G0_rain2019','N0_norain']],ignore_index=True)
    at.to_csv(R/'05_相对基准差值.csv',index=False,encoding='utf-8-sig')
    at[at.baseline=='G2_full'].to_csv(R/'05_相对G2差值.csv',index=False,encoding='utf-8-sig')
    with __import__('rasterio').open(Path(L.RASTERS)/'dem.tif') as src:scale=20*abs(src.transform.a)/1000
    diag={'n_blocks':len(labels),'n_boot_requested':2000,'n_boot_valid':len(boot),'block_scale_km':scale,
          'bootstrap_seed':42,'resample_unit':'spatial_block_id; all records within a block retained',
          'scope':'conditional on fixed predictions, dataset and folds; excludes retraining uncertainty'}
    pd.DataFrame([diag]).to_csv(R/'15_块重采样诊断.csv',index=False,encoding='utf-8-sig')
    selected=pd.read_csv(R/'17_选择OOF.csv').set_index('pointid').reindex(pv.index)
    rv=pd.DataFrame({'fixed':pv.XGBoost,'selected':selected.oof_select},index=pv.index)
    rb,_,_,_,_,_=weighted_bootstrap(y,rv.values,blocks)
    paired_table(rv,y,rb).to_csv(R/'17_固定与选择条件差值.csv',index=False,encoding='utf-8-sig')
    (R/'15_统计解释.md').write_text('# 空间块条件不确定性\n\n'
        '正式结果报告折均值/SD和合并OOF，二者不混用。区间使用同一空间块重采样，块内记录整体进出，保存2000组抽样索引。\n\n'
        '这些区间条件于已训练模型的OOF，不涵盖重新训练/调参/选因子/重划区域的不确定性。\n\n'
        '不提供折级配对t/Wilcoxon p值，不对未校准尾比例做Holm，不以区间含零宣称等效。探索性多个比较不作家族显著性结论。\n',encoding='utf-8')
    historical=Path(L.EXP)/'provenance/deprecated_statistics';historical.mkdir(parents=True,exist_ok=True)
    for name in ['15_配对检验.csv','15_显著性报告.md']:
        p=R/name
        if p.exists():p.replace(historical/name)
    print(diag,flush=True)

if __name__=='__main__':main()
