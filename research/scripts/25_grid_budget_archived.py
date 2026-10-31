"""Actual current-protocol wide/operational-grid diagnostic on outer-fold-0 training only."""
import ast,json,time,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid,StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel,delayed
import exp_lib as L


def main():
    if '--report-only' in sys.argv:
        R=Path(L.RES);S=pd.read_csv(R/'11_超参数预试算汇总.csv')
        gap=float(S.iloc[0].best_pooled_inner_auc-S.iloc[1].best_pooled_inner_auc)
        (R/'11_超参数预试算报告.md').write_text('# XGBoost 搜索范围与预算诊断\n\n'+S.to_markdown(index=False)+
            f'\n\n宽网格相对操作网格最佳内层OOF AUC差：{gap:.6f}。\n'
            '固定操作网格是预算限定示例，不宣称无损或全局最优；不据本单折诊断事后替换正式主方案。'
            '\n外层fold0测试数据未参与。完整组合、逐内折分数、内层OOF、参数和耗时已保存。',encoding='utf-8')
        return
    b=L.load_base();d=b[b.fold!=0].reset_index(drop=True);y=d.landslide.to_numpy(dtype=int)
    factors=L.GROUPS['G2'];base,narrow=L.get_model_and_grid('XGBoost',42,n_jobs=1)
    wide={'n_estimators':[200,400,800],'max_depth':[2,3,5,7,9],
          'learning_rate':[.01,.05,.1,.3],'subsample':[.6,.8,1.], 'colsample_bytree':[.6,.8,1.]}
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(d,y,d.cell_id))
    cache=[];trace=[]
    for j,(a,z) in enumerate(splits):
        xa,enc=L.build_matrix(d.iloc[a],factors,fit=True);xz,_=L.build_matrix(d.iloc[z],factors,enc=enc,fit=False)
        cache.append((a,z,xa,xz));trace.append({'inner_fold':j,'n_train':len(a),'n_validation':len(z),
            'train_ids_sha256':hashlib.sha256('|'.join(sorted(d.iloc[a].pointid)).encode()).hexdigest(),
            'matrix_sha256':enc['matrix_sha256']})
    protocol={'scope':'development diagnostic; outer fold 0 test excluded; no claim of unbiased full-CV performance',
              'seed':42,'outer_fold':0,'n_train':len(d),'n_excluded_test':int((b.fold==0).sum()),
              'grid_operational':narrow,'grid_wide':wide,'inner_trace':trace,
              'input_sha256':hashlib.sha256((Path(L.DATA)/'modeling_base.csv').read_bytes()).hexdigest(),
              'selection':'pooled inner OOF ROC-AUC, same as official main','jobs':4,'threads_per_fit':1}
    (Path(L.CFG)/'grid_budget_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    def fit_one(params):
        tic=time.perf_counter();oof=np.full(len(d),np.nan);scores=[]
        for a,z,xa,xz in cache:
            m=clone(base).set_params(**params).fit(xa,y[a]);oof[z]=m.predict_proba(xz)[:,1]
            scores.append(roc_auc_score(y[z],oof[z]))
        return params,roc_auc_score(y,oof),scores,time.perf_counter()-tic,oof
    rows=[];long=[];sums=[];vectors=[];labels=[]
    for label,grid in [('wide',wide),('operational',narrow)]:
        tic=time.perf_counter();combos=list(ParameterGrid(grid))
        results=Parallel(n_jobs=4,prefer='processes')(delayed(fit_one)(p) for p in combos)
        wall=time.perf_counter()-tic
        for i,(params,auc,sc,elapsed,oof) in enumerate(results):
            rows.append({'model':'XGBoost','outer_fold':0,'grid':label,'candidate':i,
                         'auc':auc,'mean_inner_fold_auc':float(np.mean(sc)),
                         'fit_time_s':elapsed,'params':json.dumps(params,sort_keys=True),**params})
            for j,score in enumerate(sc):long.append({'grid':label,'candidate':i,'inner_fold':j,'auc':score})
            vectors.append(oof);labels.append(f'{label}_{i}')
        best=max(results,key=lambda r:r[1])
        sums.append({'grid':label,'n_candidates':len(combos),'n_inner_fits':len(combos)*5,
                     'best_pooled_inner_auc':best[1],'best_params':json.dumps(best[0],sort_keys=True),
                     'summed_task_time_s':sum(r[3] for r in results),'wall_time_s':wall,
                     'n_outer_test_used':0})
        print(label,sums[-1],flush=True)
    R=Path(L.RES);D=Path(L.DATA)
    pd.DataFrame(rows).to_csv(R/'11_超参数预试算.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(long).to_csv(R/'11_网格逐内折得分.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(sums).to_csv(R/'11_超参数预试算汇总.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(D/'11_grid_inner_oof.npz',probabilities=np.array(vectors),pointid=d.pointid.to_numpy(dtype=str),
                         y=y,candidate=np.array(labels,dtype=str))
    # Derive all operational grid counts from source literals, without loading optional estimators.
    fn=next(n for n in ast.parse((Path(L.EXP)/'scripts/exp_lib.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='get_model_and_grid')
    effective=[]
    node=next(n for n in fn.body if isinstance(n,ast.If))
    while isinstance(node,ast.If):
        name=ast.literal_eval(node.test.comparators[0])
        grid=next(ast.literal_eval(n.value) for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='g' for t in n.targets))
        combos=list(ParameterGrid(grid));unique={json.dumps({k:v for k,v in p.items() if not (name=='SVM' and p['kernel']=='linear' and k=='gamma')},sort_keys=True) for p in combos}
        effective.append({'model':name,'nominal_candidates':len(combos),'effective_candidates':len(unique)})
        node=node.orelse[0] if node.orelse and isinstance(node.orelse[0],ast.If) else None
    pd.DataFrame(effective).to_csv(R/'11_有效组合统计.csv',index=False,encoding='utf-8-sig')
    S=pd.DataFrame(sums);gap=sums[0]['best_pooled_inner_auc']-sums[1]['best_pooled_inner_auc']
    md=['# XGBoost 搜索范围与预算诊断\n',S.to_markdown(index=False),
        f'\n宽网格相对操作网格的最佳内层合并OOF AUC差：{gap:.6f}。',
        '固定操作网格是预算与复现示例，不宣称无损或找到全局最优；不因本诊断分数较高而事后替换正式主方案。',
        '全部搜索只使用外层fold0训练记录；其测试区完全未用于本诊断。单折开发诊断不能替代五外折独立验证。',
        '原始输出包含564组组合、2820个内折得分、每个组合的550条内层OOF概率、参数、任务时间、墙钟时间和输入/矩阵哈希。']
    (R/'11_超参数预试算报告.md').write_text('\n'.join(md),encoding='utf-8')

if __name__=='__main__':main()
