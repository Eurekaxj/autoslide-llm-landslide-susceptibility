"""One additional paired run, scheduled after all other experiments, with CPU and wall time.

The three development repeats are retained, but overlapping development workloads
make their timing exploratory. This prospective sequential pair is the reported
timing check (n=1 paired full execution); no significance or universal speed claim.
"""
import importlib.util,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import exp_lib as L


def main():
    p=Path(L.EXP)/'scripts/26_pslsa_common.py'
    spec=importlib.util.spec_from_file_location('matched_comparison',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    b=L.load_base();F=L.GROUPS['G2'];_,grid=L.get_model_and_grid('XGBoost',42,n_jobs=1)
    splits={}
    for k in range(5):
        d=b[b.fold!=k].reset_index(drop=True)
        splits[k]=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(d,d.landslide,d.cell_id))
    protocol={'paired_repeats':1,'purpose':'prospective timing check after all other v4 experiment stages',
              'selection':'same shared mean inner-fold AUC protocol as experiment 26','threads_per_fit':1,
              'execution_order':['Framework_cached','PSLSA_source_adapter'],
              'scheduling':'runner executes serially; no concurrent v4 experiments',
              'limitation':'no OS-level exclusive CPU reservation; n=1 pair, descriptive only',
              'development_timing':'experiment 26 timings retained as exploratory; some development workloads overlapped'}
    (Path(L.CFG)/'timing_check_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[];times=[];par=[]
    for branch in protocol['execution_order']:
        cpu=time.process_time();tic=time.perf_counter()
        result=mod.run_branch(branch,b,F,grid,3,splits)
        elapsed=time.perf_counter()-tic;cpu_used=time.process_time()-cpu
        records.append(result[0]);par.extend(result[2])
        times.append({'branch':branch,'wall_time_s':elapsed,'process_cpu_s':cpu_used,
                      'search_model_fits':600,'refit_and_threshold_model_fits':30,
                      'preprocessing_fit_calls_derived_from_completed_protocol':30 if branch=='Framework_cached' else 630,
                      'paired_repeats':1,'scope':'this sequential run; no formal timing inference'})
    o=pd.concat(records,ignore_index=True);o.to_csv(Path(L.RES)/'29_共同条件OOF.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(par).to_csv(Path(L.RES)/'29_共同条件参数.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(times).to_csv(Path(L.RES)/'29_顺序配对计时.csv',index=False,encoding='utf-8-sig')
    a=records[0].set_index('pointid');z=records[1].set_index('pointid').reindex(a.index)
    diff=float(np.abs(a.oof-z.oof).max())
    checks={'n_records':len(a),'max_probability_difference':diff,'yhat_mismatches':int((a.yhat!=z.yhat).sum()),
            'baseline_over_framework_wall_time':times[1]['wall_time_s']/times[0]['wall_time_s'],
            'preprocessing_fits_baseline':630,'preprocessing_fits_framework':30,
            'reduction_in_redundant_preprocessing_fits_fraction':1-30/630,
            'model_fit_budget_equal':True,'timing_scope':protocol['limitation']}
    (Path(L.RES)/'29_等价性.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    if diff>1e-9 or checks['yhat_mismatches']:raise RuntimeError('Sequential timing pair changed predictions')
    print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
