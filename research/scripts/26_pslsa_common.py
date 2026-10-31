"""Prospective matched comparison: original-source adapter versus cached framework.

Identical data, features, outer/inner splits, estimator, grid, scoring, thresholds,
thread count and number of fits. No seed selection or outcome-dependent exclusions.
Timing measures this implementation's preprocessing cache, not an inherent LLM effect.
"""
import os,sys,json,time,hashlib,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid,StratifiedGroupKFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import exp_lib as L
from pslsa_source_adapter import training_method,export_adapter

OUT=Path(L.RES); DATA=Path(L.DATA); CONFIG=Path(L.CFG)
def h_ids(ids):return hashlib.sha256('|'.join(sorted(map(str,ids))).encode()).hexdigest()


def cached_search(d,y,splits,grid,base,factors):
    """Cache fitted preprocessing per split; all model fits use identical inputs."""
    cache=[]; prep_start=time.perf_counter()
    for a,z in splits:
        xa,enc=L.build_matrix(d.iloc[a],factors,fit=True)
        xz,_=L.build_matrix(d.iloc[z],factors,enc=enc,fit=False)
        cache.append((a,z,xa,xz))
    prep_s=time.perf_counter()-prep_start
    rows=[]; combos=list(ParameterGrid(grid))
    for i,params in enumerate(combos):
        scores=[];tic=time.perf_counter()
        for j,(a,z,xa,xz) in enumerate(cache):
            mdl=clone(base).set_params(**params).fit(xa,y[a])
            score=roc_auc_score(y[z],mdl.predict_proba(xz)[:,1]);scores.append(score)
            rows.append({'candidate':i,'inner_fold':j,'roc_auc':score,'params':json.dumps(params,sort_keys=True)})
        for r in rows[-len(splits):]:r['candidate_elapsed_s']=time.perf_counter()-tic
    means=np.array([np.mean([r['roc_auc'] for r in rows if r['candidate']==i]) for i in range(len(combos))])
    best_i=int(np.argmax(means));best=combos[best_i]
    xa,enc=L.build_matrix(d,factors,fit=True)
    fitted=clone(base).set_params(**best).fit(xa,y)
    # Both methods perform five additional threshold fits: equal computational budget.
    inner=np.full(len(d),np.nan)
    for a,z,xa,xz in cache:
        fitted_inner=clone(base).set_params(**best).fit(xa,y[a])
        inner[z]=fitted_inner.predict_proba(xz)[:,1]
    return fitted,enc,best,L.threshold_from_oof(inner,y),rows,inner,prep_s


def run_branch(branch,b,factors,grid,rep,split_cache):
    y=b.landslide.to_numpy(dtype=int);fold=b.fold.to_numpy();oof=np.full(len(b),np.nan)
    thresholds={};fold_rows=[];params_rows=[];search_rows=[];inner_rows=[];clocks=[]
    tic_all=time.perf_counter()
    base,_=L.get_model_and_grid('XGBoost',42,n_jobs=1)
    for k in range(5):
        tr=np.where(fold!=k)[0];te=np.where(fold==k)[0]
        d=b.iloc[tr].reset_index(drop=True);dte=b.iloc[te];yt=y[tr];splits=split_cache[k]
        tic=time.perf_counter()
        if branch=='PSLSA_source_adapter':
            method=training_method(True,tuple(factors),splits)
            search,_=method(d,yt,clone(base),grid)
            best={n.removeprefix('model__'):v for n,v in search.best_params_.items()}
            model=search.best_estimator_
            cv=search.cv_results_
            for i,params in enumerate(cv['params']):
                raw={n.removeprefix('model__'):v for n,v in params.items()}
                for j in range(5):
                    search_rows.append({'rep':rep,'branch':branch,'outer_fold':k,'candidate':i,'inner_fold':j,
                                        'roc_auc':cv[f'split{j}_test_score'][i],
                                        'params':json.dumps(raw,sort_keys=True),
                                        'candidate_elapsed_s':float(cv['mean_fit_time'][i]*5)})
            inner=np.full(len(d),np.nan)
            for a,z in splits:
                mm=clone(model).fit(d.iloc[a],yt[a]);inner[z]=mm.predict_proba(d.iloc[z])[:,1]
            thr=L.threshold_from_oof(inner,yt);p=model.predict_proba(dte)[:,1]
            prep_s=np.nan
            input_matrix_hash=model.named_steps['factors'].encoder_['matrix_sha256']
        else:
            model,enc,best,thr,rr,inner,prep_s=cached_search(d,yt,splits,grid,base,factors)
            xp,_=L.build_matrix(dte,factors,enc=enc,fit=False);p=model.predict_proba(xp)[:,1]
            input_matrix_hash=enc['matrix_sha256']
            for r in rr:r.update(rep=rep,branch=branch,outer_fold=k)
            search_rows.extend(rr)
        elapsed=time.perf_counter()-tic
        oof[te]=p;thresholds[k]=thr
        fm=L.metrics_at(y[te],p,thr);fm.update(rep=rep,branch=branch,fold=k,n_train=len(tr),n_test=len(te),elapsed_s=elapsed)
        fold_rows.append(fm)
        params_rows.append({'rep':rep,'branch':branch,'fold':k,'params':json.dumps(best,sort_keys=True),
                            'threshold':thr,'outer_train_ids_sha256':h_ids(d.pointid),
                            'matrix_sha256':input_matrix_hash,'search_fits':len(list(ParameterGrid(grid)))*len(splits),
                            'refit_and_threshold_fits':6,'preprocessing_cache_initial_s':prep_s})
        inner_rows.append(pd.DataFrame({'rep':rep,'branch':branch,'outer_fold':k,
                                        'pointid':d.pointid,'y':yt,'inner_oof':inner}))
        print(f'{branch} rep={rep} outer={k} AUC={fm["roc_auc"]:.6f} seconds={elapsed:.2f}',flush=True)
    total=time.perf_counter()-tic_all
    thrvec=pd.Series(fold).map(thresholds).to_numpy()
    record=pd.DataFrame({'rep':rep,'branch':branch,'pointid':b.pointid,'cell_id':b.cell_id,
                         'block_id':b.spatial_block_id,'fold':fold,'y':y,'oof':oof,
                         'threshold':thrvec,'yhat':(oof>=thrvec).astype(int)})
    pooled=L.metrics_with_fold_threshold(y,oof,fold,thresholds)
    pooled.update(rep=rep,branch=branch,elapsed_s=total,outer_folds=5,search_fits=16*5,refit_and_threshold_fits=6,
                  roc_auc_fold_mean=float(np.mean([r['roc_auc'] for r in fold_rows])))
    return record,fold_rows,params_rows,search_rows,pd.concat(inner_rows,ignore_index=True),pooled


def main():
    b=L.load_base().reset_index(drop=True);factors=L.GROUPS['G2']
    _,grid=L.get_model_and_grid('XGBoost',42,n_jobs=1)
    protocol={
        'run_id':'v4_pslsa_matched_001','source':export_adapter(),'n_records':len(b),'n_cells':b.cell_id.nunique(),
        'features':factors,'n_features_raw':len(factors),'grid':grid,'n_candidates':len(list(ParameterGrid(grid))),
        'model_seed':42,'outer':'frozen five spatial regions','inner':'StratifiedGroupKFold(5,shuffle=True,random_state=42),groups=cell_id',
        'selection_metric':'arithmetic mean of five inner-fold ROC-AUC values for BOTH branches',
        'selection_difference_from_official_main':'official main selects pooled inner OOF ROC-AUC; this controlled benchmark uses common mean-fold selection',
        'threshold':'F1-optimal threshold on selected-parameter inner OOF; same function and tie rule',
        'threads_per_model':1,'search_jobs':1,'repeats':3,'order':'alternate baseline/framework; all repeats retained',
        'difference_under_test':'training-only preprocessing cache in framework; all training/validation data and fit budgets equal',
        'data_sha256':hashlib.sha256((DATA/'modeling_base.csv').read_bytes()).hexdigest(),
        'fold_table_sha256':hashlib.sha256((CONFIG/'fold_table.csv').read_bytes()).hexdigest(),
        'not_claimed':['LLM improves the classifier algorithm','unmodified PSLSA GUI tested','general human development-time reduction'],
    }
    # Freeze protocol before executing either branch; do not choose seeds or exclusions from outcomes.
    (CONFIG/'pslsa_matched_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    split_cache={};split_records=[]
    for k in range(5):
        d=b[b.fold!=k].reset_index(drop=True)
        splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(d,d.landslide,d.cell_id))
        split_cache[k]=splits
        for j,(a,z) in enumerate(splits):
            if set(d.iloc[a].cell_id)&set(d.iloc[z].cell_id):raise ValueError('Cell leakage')
            split_records.append(pd.DataFrame({'outer_fold':k,'inner_fold':j,'pointid':d.pointid,
                                                'role':np.where(np.isin(np.arange(len(d)),z),'validation','training')}))
    pd.concat(split_records,ignore_index=True).to_csv(OUT/'26_共同内层划分.csv',index=False,encoding='utf-8-sig')
    oofs=[];folds=[];pars=[];scores=[];inners=[];sums=[];equiv=[]
    branches=['PSLSA_source_adapter','Framework_cached']
    REPS = int(os.environ.get("PSLSA_REPS", "3"))
    for rep in range(REPS):
        for branch in (branches if rep%2==0 else list(reversed(branches))):
            record,fr,pr,sr,ir,summary=run_branch(branch,b,factors,grid,rep,split_cache)
            oofs.append(record);folds+=fr;pars+=pr;scores+=sr;inners.append(ir);sums.append(summary)
        OO=pd.concat(oofs,ignore_index=True)
        a=OO[(OO.rep==rep)&(OO.branch==branches[0])].set_index('pointid').sort_index()
        z=OO[(OO.rep==rep)&(OO.branch==branches[1])].set_index('pointid').reindex(a.index)
        diff=float(np.abs(a.oof-z.oof).max())
        equiv.append({'rep':rep,'n':len(a),'max_probability_difference':diff,
                       'yhat_mismatches':int((a.yhat!=z.yhat).sum()),'auc_difference':roc_auc_score(a.y,a.oof)-roc_auc_score(z.y,z.oof)})
        if diff>1e-9:raise RuntimeError(f'Matched comparison predictions differ: {diff}')
        # Preserve complete results after each paired repeat, including slower/unfavourable runs.
        OO.to_csv(OUT/'26_共同条件OOF.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(folds).to_csv(OUT/'26_共同条件逐折.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(pars).to_csv(OUT/'26_共同条件参数.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(scores).to_csv(OUT/'26_共同条件搜索明细.csv',index=False,encoding='utf-8-sig')
        pd.concat(inners,ignore_index=True).to_csv(OUT/'26_共同条件内层OOF.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(sums).to_csv(OUT/'26_共同条件汇总.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(equiv).to_csv(OUT/'26_逐点等价性.csv',index=False,encoding='utf-8-sig')
    S=pd.DataFrame(sums);times=S.pivot(index='rep',columns='branch',values='elapsed_s')
    times['baseline_over_framework']=times[branches[0]]/times[branches[1]]
    times.reset_index().to_csv(OUT/'26_配对运行耗时.csv',index=False,encoding='utf-8-sig')
    # Run the actual unmodified source METHOD once as a separate diagnostic, not a fair ranking.
    X,enc=L.build_matrix(b,factors,fit=True);raw=X.copy()
    for col in enc['cont']:raw[col]=L.canon_numeric(b[col].values)
    native=training_method(False)
    base,_=L.get_model_and_grid('XGBoost',42,n_jobs=1)
    tic=time.perf_counter();search,best=native(raw.to_numpy(),b.landslide.to_numpy(),base,grid)
    _,test=train_test_split(np.arange(len(b)),test_size=.3,random_state=13)
    standardized=StandardScaler().fit_transform(raw.to_numpy());p=search.predict_proba(standardized[test])[:,1]
    native_oof=pd.DataFrame({'pointid':b.iloc[test].pointid,'y':b.iloc[test].landslide,'score':p,'role':'same_split_random_holdout'})
    native_oof.to_csv(OUT/'26_原函数诊断留出.csv',index=False,encoding='utf-8-sig')
    native_summary={'n_eval':len(test),'auc':roc_auc_score(b.iloc[test].landslide,p),'elapsed_s':time.perf_counter()-tic,
                    'best_params':best,'scope':'actual original model_train method, common frozen input/grid; random split diagnostic, not GUI or matched ranking'}
    (OUT/'26_原函数诊断.json').write_text(json.dumps(native_summary,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# PSLSA 与框架的共同条件对照\n',
        '原始源码、提取函数及最小协议补丁均已保存。主对照共用702记录、14因子、空间外折、cell分组内折、16参数组合、平均内折AUC选择、阈值规则和单线程预算。',
        '本对照的参数选择聚合为共同的内折AUC算术均值；不能与正式主实验的合并内层OOF选择口径混称。',
        '唯一性能实现差异是框架缓存每个内层训练子集的编码矩阵，基线Pipeline在每个参数组合重新拟合相同预处理。缓存不跨外折、不使用验证子集拟合。',
        '\n## 全部三次配对运行\n',S.to_markdown(index=False),
        '\n## 数值等价\n',pd.DataFrame(equiv).to_markdown(index=False),
        '\n## 实测运行时间\n',times.reset_index().to_markdown(index=False),
        f'\n耗时比（基线/框架）中位数：{times.baseline_over_framework.median():.3f}。三次均保留；仅描述本机本实现，不做显著性或普遍效率推断。',
        '\n## 结论范围\n',
        '相同条件下准确率和逐点结果相同是有效性证据，不宣称LLM改进了XGBoost。运行耗时差异可归于这里实际实现的预处理复用，不能归因为LLM本身。',
        '开发者人工耗时没有历史计时证据，故不报告百分比节约。原函数诊断单列，未经原版GUI运行，不能称原版GUI全面复现。',
        '用于绘图的搜索明细、702点OOF、逐折指标、内层划分/OOF、参数和三次配对计时均完整导出。']
    (OUT/'26_PSLSA共同条件报告.md').write_text('\n'.join(md),encoding='utf-8')
    print(times.to_string(),flush=True)

if __name__=='__main__':main()
