"""Classify every experiment and preserve complete numerical source data for later figures."""
import json,hashlib,shutil,csv,datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
import exp_lib as L

E=Path(L.EXP);R=Path(L.RES);D=Path(L.DATA);SD=E/'source_data'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

GROUPS=[
 ('SD01_data','数据与空间分区',['data/modeling_base.csv','data/prediction_full_valid.csv','config/fold_table.csv','config/fold_table_meta.json','results/00_v4冻结验收.json','inputs/factor_metadata/因子清单.csv'],['R1.6','R2.13']),
 ('SD02_models','模型横向比较与因子组',['results/03_汇总.csv','results/03_逐折指标.csv','results/03_最优超参数.csv','results/03_训练集指标.csv','results/03_OOF预测.csv','results/03_模型配置与默认参数.json','results/03_全部折模型完整配置.jsonl'],['R1.2','R2.14']),
 ('SD03_sensitivity','土壤温度降雨与消融',['results/05_消融汇总.csv','results/05_消融逐折.csv','results/05_消融OOF.csv','results/05_最优超参数.csv','results/05_温度回归系数.csv','results/05_相对基准差值.csv','results/05_消融重采样AUC矩阵.csv','data/rain_candidates.csv','results/17_候选评分.csv','results/17_选择OOF.csv','results/17_折内选择明细.csv','results/17_最优超参数.csv','results/17_固定与选择条件差值.csv','config/rain_candidates_manifest.json','results/06_降雨窗口相关矩阵.csv','data/_rain_window/rain_rx5day_W01.tif','data/_rain_window/rain_rx5day_W10.tif','data/_rain_window/rain_raincv_W01.tif','data/_rain_window/rain_raincv_W10.tif'],['R2.3','R2.6','R2.7']),
 ('SD04_background','背景采样及缓冲敏感性',['results/04_缓冲点集统计.csv','results/04_缓冲敏感性汇总.csv','results/04_缓冲逐折指标.csv','results/04_缓冲fold一致性.csv','results/04_缓冲汇总.csv','results/04_复现与精度敏感性.csv','data/全区概率_缓冲.csv']+[f'data/nonlandslide_{r}km_seed{s}.csv' for r in [1,2,3] for s in range(42,47)],['R1.7']),
 ('SD05_diagnostics','学习曲线过拟合与校准',['results/08_空间学习曲线.csv','results/08_学习曲线_按比例汇总.csv','results/08_学习曲线_波动拆分.csv','results/08_过拟合诊断.csv','results/08_校准.csv','results/08_MLP_Loss.csv','results/08_超参数选择频率.csv'],['R1.8','R2.12']),
 ('SD06_reproducibility','重复运行与随机种子',['results/07_seed设计A.csv','results/07_seed设计A_OOF.csv','results/07_seed设计B.csv','results/07_seed设计B_OOF.csv','results/07_seed设计B联合.csv','results/07_seed设计B联合_OOF.csv','results/07_设计A一致性.json','results/07_域概率稳定性.json','data/seed概率矩阵.npy','data/prediction_base.csv','results/27_缓存与正式主结果等价性.csv'],['R1.9']),
 ('SD07_mapping','完整域地图适用域与预测率',['data/全区预测概率_完整域.csv','data/完整域空间留出概率.csv','data/domain_mask_full_valid.tif','data/适用域遮罩_完整域.csv','data/全区预测概率.csv','results/22_完整域构建.csv','results/22_原域一致性.csv','results/22_滑坡记录匹配.csv','results/22_易发性分级面积.csv','results/02_参数覆盖统计.csv','results/02_多变量距离.csv','results/02_土地利用类别覆盖.csv','results/02_适用域分量.csv','results/08c_预测率曲线.csv','results/08c_全样本地图曲线.csv','results/08c_计数核对.csv','results/09_G2G3差异.csv'],['R1.6','R2.11']),
 ('SD08_importance','原始因子分组置换重要性',['results/09_重要性置换明细.csv','results/09_重要性.csv','results/09_重要性汇总.csv'],['R2.15']),
 ('SD09_statistics','空间块条件不确定性',['results/15_bootstrap差值.csv','results/15_合并OOF条件区间.csv','results/15_重采样AUC矩阵.csv','results/15_重采样AP矩阵.csv','results/15_指标重采样明细.csv','results/15_块重采样诊断.csv','data/15_block_draw_indices.npy','data/15_block_labels.json'],['R1.2','R1.8']),
 ('SD10_grid','搜索范围与预算诊断',['results/11_超参数预试算.csv','results/11_网格逐内折得分.csv','results/11_超参数预试算汇总.csv','results/11_有效组合统计.csv','data/11_grid_inner_oof.npz','config/grid_budget_protocol.json'],['R1.4']),
 ('SD11_tools','PSLSA共同条件及计时',['results/26_共同条件OOF.csv','results/26_共同条件汇总.csv','results/26_共同条件逐折.csv','results/26_共同条件参数.csv','results/26_共同条件搜索明细.csv','results/26_共同条件内层OOF.csv','results/26_共同内层划分.csv','results/26_逐点等价性.csv','results/26_配对运行耗时.csv','results/26_原函数诊断留出.csv','results/26_原函数诊断.json','results/29_顺序配对计时.csv','results/29_共同条件OOF.csv','results/29_共同条件参数.csv','results/29_等价性.json','config/pslsa_matched_protocol.json','config/timing_check_protocol.json','vendor/PSLSA_extracted/protocol_adapter.diff'],['R1.5']),
 ('SD12_gis','开源地形计算同定义核对',['results/10_逐像元对照.csv','results/10_同定义独立复算.csv'],['R2.10']),
 ('SD13_history_PSLSA','历史PSLSA协议模拟（保留全部结果）',['results/18_PSLSA协议阶梯.csv','results/18_PSLSA逐点OOF.csv','results/18_PSLSA同域概率对比.csv','results/18_PSLSA折级明细.csv','results/18_PSLSA泄漏诊断.csv','results/18_历史评估计数修正.csv','data/PSLSA适配_全域概率.csv'],['R1.5']),
]

FIGURES=[
 ('fig_model_comparison','G2模型比较','SD02_models/03_逐折指标.csv','SD02_models/03_OOF预测.csv','fold mean and SD; pooled ROC separate'),
 ('fig_factor_groups','XGBoost因子组','SD02_models/03_逐折指标.csv','SD02_models/03_汇总.csv','same model; four frozen factor groups'),
 ('fig_ablation','XGBoost消融','SD03_sensitivity/05_消融逐折.csv','SD03_sensitivity/05_消融OOF.csv','fold mean/SD; aliases explicitly retained'),
 ('fig_buffer','缓冲半径与背景抽样','SD04_background/04_缓冲敏感性汇总.csv','SD04_background/04_缓冲逐折指标.csv','five background sets per radius; not independent spatial studies'),
 ('fig_learning','七模型学习曲线','SD05_diagnostics/08_空间学习曲线.csv','SD05_diagnostics/08_学习曲线_按比例汇总.csv','fold/within-fold subset variation; fixed selected parameters'),
 ('fig_calibration','OOF校准诊断','SD05_diagnostics/08_校准.csv','SD02_models/03_OOF预测.csv','balanced sampled dataset; not regional event probabilities'),
 ('fig_importance','外测分组置换重要性','SD08_importance/09_重要性置换明细.csv','SD08_importance/09_重要性汇总.csv','20 permutations x 5 outer folds x 15 raw factors; not causal effects'),
 ('fig_seed','随机种子诊断','SD06_reproducibility/07_seed设计B.csv','SD06_reproducibility/07_seed设计A.csv','30 seeds; fixed data/splits/parameters; no independent spatial replication'),
 ('fig_full_map','完整有效域易发性草图','SD07_mapping/全区预测概率_完整域.csv','SD01_data/modeling_base.csv','54057 cells; final full-sample fitted score; EPSG:32649'),
 ('fig_prediction_rate','空间留出预测率','SD07_mapping/08c_预测率曲线.csv','SD07_mapping/08c_全样本地图曲线.csv','336 unique landslide cells; held-out vs descriptive map separated'),
 ('fig_uncertainty','模型差值条件区间','SD09_statistics/15_bootstrap差值.csv','SD09_statistics/15_重采样AUC矩阵.csv','fixed-OOF spatial-block conditional intervals; no formal p/q'),
 ('fig_grid_budget','宽操作网格预算','SD10_grid/11_超参数预试算.csv','SD10_grid/11_网格逐内折得分.csv','outer-fold-0 training only; development diagnostic'),
 ('fig_pslsa_equivalence','PSLSA共同条件对照','SD11_tools/26_共同条件OOF.csv','SD11_tools/29_顺序配对计时.csv','same protocol; 3 development pairs + 1 sequential timing pair; no selective seed reporting'),
 ('fig_gis_agreement','地形同定义核对','SD12_gis/10_逐像元对照.csv','SD12_gis/10_同定义独立复算.csv','all valid compared pixels; borders and flat-aspect conventions recorded'),
]


def cleanup_historical_reports():
    keep={'11_超参数预试算报告.md','15_统计解释.md','10_开源实现范围.md','22_报告.md','08c_报告.md','02_覆盖报告.md','26_PSLSA共同条件报告.md','23_摘要.md'}
    hist=E/'provenance/v3_reports';hist.mkdir(parents=True,exist_ok=True)
    for p in R.glob('*.md'):
        if p.name not in keep and not p.name.startswith(('29_','30_','32_')):p.replace(hist/p.name)
    gui_history=E/'provenance/gui_branch_unverified';gui_history.mkdir(parents=True,exist_ok=True)
    for p in R.glob('21_*.csv'):p.replace(gui_history/p.name)
    legacy_mask=D/'适用域遮罩_53911.csv'
    if legacy_mask.exists():legacy_mask.replace(E/'provenance/v3_domain_mask_legacy_name.csv')
    # Recompute the legacy P1 counting diagnostic without changing any old prediction.
    from sklearn.model_selection import train_test_split
    idx=np.arange(702);train,_=train_test_split(idx,test_size=.3,random_state=13)
    _,a=train_test_split(idx,test_size=.3,random_state=128);_,z=train_test_split(idx[-210:-10],test_size=.7,random_state=135)
    ev=np.r_[a,z];unique=np.unique(ev)
    diag=pd.DataFrame([{'counting_unit':'evaluation row occurrences','n_eval':len(ev),'n_overlap':int(np.isin(ev,train).sum()),'overlap_fraction':np.isin(ev,train).mean()},
                       {'counting_unit':'unique evaluation records','n_eval':len(unique),'n_overlap':int(np.isin(unique,train).sum()),'overlap_fraction':np.isin(unique,train).mean()}])
    diag.to_csv(R/'18_历史评估计数修正.csv',index=False,encoding='utf-8-sig')
    p=R/'18_PSLSA泄漏诊断.csv'
    if p.exists():
        historical=E/'provenance/legacy_pslsa';historical.mkdir(parents=True,exist_ok=True)
        if not (historical/p.name).exists():shutil.copy2(p,historical/p.name)
        df=pd.read_csv(p);mask=df.protocol.eq('P1_native')
        df.loc[mask,'n_overlap']=int(np.isin(ev,train).sum());df.loc[mask,'overlap_ratio_of_eval']=np.isin(ev,train).mean()
        df.loc[mask,'说明']='Evaluation-row occurrences: 246/351; unique records: 212/306; 45 repeated rows. Slice-adapted simulation, not unmodified GUI.'
        df['protocol']=df.protocol.replace({'P1_native':'P1_slice_adapted_protocol_simulation'})
        df.to_csv(p,index=False,encoding='utf-8-sig')
    for p in R.glob('18_*.csv'):
        df=pd.read_csv(p)
        for col in ['protocol','A','B']:
            if col in df:df[col]=df[col].replace({'P1_native':'P1_slice_adapted_protocol_simulation'})
        df['evidence_scope']='Historical source-semantics protocol simulation; not the matched tool comparison'
        df.to_csv(p,index=False,encoding='utf-8-sig')
    (R/'18_历史协议诊断说明.md').write_text('# 历史PSLSA协议模拟\n\n'+diag.to_markdown(index=False)+
        '\n\n原始351评估行包含306条唯一记录和45条重复行。P1是切片适配后的协议模拟，不是原版GUI自然表现。'
        '\nP4→P5同时改变训练比例、测试对象与聚合，不能定量归因于空间泄漏。P6/P7使用不同网格，不用于宣称本框架更准确。'
        '\n本轮主工具对照以26（实际源函数的最小共同协议适配）和29（顺序计时）为准。旧有不利结果也全部保留在历史数据中。',encoding='utf-8')


def add_derived_sources():
    derived=R/'derived_plot_data';derived.mkdir(exist_ok=True)
    o=pd.read_csv(R/'03_OOF预测.csv');roc=[];pr=[];cm=[]
    for (g,m),d in o.groupby(['group','model']):
        fpr,tpr,t=roc_curve(d.y,d.oof);roc.append(pd.DataFrame({'group':g,'model':m,'fpr':fpr,'tpr':tpr,'threshold':t}))
        precision,recall,t=precision_recall_curve(d.y,d.oof)
        pr.append(pd.DataFrame({'group':g,'model':m,'precision':precision,'recall':recall,'threshold':np.r_[t,np.nan]}))
        for k,z in d.groupby('fold'):
            tn,fp,fn,tp=confusion_matrix(z.y,z.yhat,labels=[0,1]).ravel();cm.append({'group':g,'model':m,'fold':k,'tn':tn,'fp':fp,'fn':fn,'tp':tp,'n':len(z),'threshold':z.threshold.iloc[0]})
    pd.concat(roc,ignore_index=True).to_csv(derived/'ROC_all_groups_models.csv',index=False,encoding='utf-8-sig')
    pd.concat(pr,ignore_index=True).to_csv(derived/'PR_all_groups_models.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(cm).to_csv(derived/'confusion_by_fold.csv',index=False,encoding='utf-8-sig')
    PB=np.load(D/'seed概率矩阵.npy');pred=pd.read_csv(D/'prediction_base.csv')
    pred[['pointid','row','col','cell_id']].assign(score_sd=PB.std(axis=0,ddof=1),score_mean=PB.mean(axis=0)).to_csv(derived/'seed_stability_53911.csv',index=False,encoding='utf-8-sig')
    return {'SD02_models':[p.relative_to(E).as_posix() for p in derived.glob('*all_groups*')]+[(derived/'confusion_by_fold.csv').relative_to(E).as_posix()],
            'SD06_reproducibility':[(derived/'seed_stability_53911.csv').relative_to(E).as_posix()]}


def approved_numbers():
    rows=[]
    def add(name,value,unit,scope,source,domain=None,status='verified',note=''):
        rows.append({'item':name,'value':value,'unit':unit,'statistical_scope':scope,'domain_n':domain,
                     'source_file':source,'evidence_status':status,'parent_run_id':'v3_1_full_001','package_run_id':'v4_final_001','note':note})
    s=pd.read_csv(R/'03_汇总.csv')
    for z in s.itertuples():
        if z.group=='G2' or z.model=='XGBoost':
            for key in ['roc_auc_mean','roc_auc_std','oof_roc_auc','oof_pr_auc','oof_f1_foldthr','oof_brier']:
                add(f'{z.group}/{z.model}/{key}',getattr(z,key),'dimensionless','fold mean/SD or pooled OOF as field named','results/03_汇总.csv',note='Frozen official model; operational grid; fixed selected feature set')
    for z in pd.read_csv(R/'05_消融汇总.csv').itertuples():
        add('ablation/'+z.variant+'/pooled_auc',z.oof_roc_auc,'dimensionless','pooled OOF; XGBoost','results/05_消融汇总.csv',note='aliases are shared baselines, not independent experiments')
    for z in pd.read_csv(R/'22_易发性分级面积.csv').itertuples(index=False,name=None):
        add('area/'+z[0],z[2],'km2','actual affine pixel area; full valid raster mask','results/22_易发性分级面积.csv',54057,note='raster-mask area, not exact vector administrative area')
    sup=pd.read_csv(R/'02_适用域分量.csv')
    for z in sup.itertuples(index=False,name=None):add('support/'+z[0],z[1],'cells','full-training marginal coverage','results/02_适用域分量.csv',54057,note=f'fraction={z[2]}; no predictive reliability guarantee')
    for name,path in [('heldout','08c_预测率曲线.csv'),('descriptive_fit','08c_全样本地图曲线.csv')]:
        for z in pd.read_csv(R/path).to_dict('records'):
            add(f'prediction_rate/{name}/top_{z["前k%"]}pct',z['命中率'],'fraction of unique landslide cells','336 unique landslide cells; sampled background FPR separate','results/'+path,54057,note=f'background_sample_FPR={z["背景误报率"]}; not regional event probability')
    sj=json.loads((R/'07_域概率稳定性.json').read_text(encoding='utf-8'))
    add('seed/score_sd_mean',sj['prob_sd_mean'],'score','30 model seeds; fixed parameters','results/07_域概率稳定性.json',53911,note='common OLD domain; not full 54057 domain')
    for z in pd.read_csv(R/'11_超参数预试算汇总.csv').itertuples():add('grid/'+z.grid+'/best_inner_auc',z.best_pooled_inner_auc,'dimensionless','development diagnostic on outer-fold-0 training only','results/11_超参数预试算汇总.csv',note=f'candidates={z.n_candidates}; no unbiased full-CV superiority claim')
    for branch,g in pd.read_csv(R/'26_共同条件汇总.csv').groupby('branch'):
        add('matched_tools/'+branch+'/pooled_auc',g.roc_auc.iloc[0],'dimensionless','shared mean-inner-fold selection; all three development repeats retained','results/26_共同条件汇总.csv',note='same protocol for both tools; different selection aggregation from official main')
    timing=json.loads((R/'29_等价性.json').read_text(encoding='utf-8'))
    for key in ['max_probability_difference','baseline_over_framework_wall_time','preprocessing_fits_baseline','preprocessing_fits_framework','reduction_in_redundant_preprocessing_fits_fraction']:
        add('matched_tools/'+key,timing[key],'as named','one additional sequential full pair; descriptive only','results/29_等价性.json',note='preprocessing counts derived from completed protocol; no general LLM or human-time effect')
    out=pd.DataFrame(rows);out.to_csv(R/'23_权威数字表.csv',index=False,encoding='utf-8-sig')
    (R/'23_摘要.md').write_text('# 核准数字与引用边界\n\n'
        '本表依据已经核对的输入与输出生成；每项保留统计对象、域版本、来源及限制。不能脱离这些字段复制数字。\n\n'+out.to_markdown(index=False)+
        '\n\n未校准尾比例、Holm q值、折级独立t检验及“区间含零即等效”均不作为正式推断。分数来自1:1背景抽样，不能解释为全区真实发生概率。',encoding='utf-8')


def main():
    cleanup_historical_reports();extra=add_derived_sources();approved_numbers()
    assets=[];dictionary=[];catalog=[];arrays=[]
    units={'DEM':'m','slope':'degree','aspect':'degree; -1=flat','road':'m','river':'m','roughness':'m; mean absolute neighbour difference','soil_clay':'percent','soil_silt':'percent','rx5day':'mm; multiyear mean annual maximum 5-day sum','raincv':'dimensionless; mean annual daily CV','Lon':'degree east','Lat':'degree north','X':'m EPSG:32649','Y':'m EPSG:32649'}
    for gid,title,paths,reviewers in GROUPS:
        dest=SD/gid;dest.mkdir(parents=True,exist_ok=True)
        selected=paths+extra.get(gid,[])
        for rel in selected:
            source=E/rel
            if not source.is_file():raise FileNotFoundError(f'Source-data file missing: {rel}')
            target=dest/source.name;shutil.copy2(source,target)
            row={'experiment_id':gid,'file':target.relative_to(E).as_posix(),'canonical_source':rel,'sha256':sha(target),'bytes':target.stat().st_size,'n_rows':None,'n_columns':None,'unit_scope':'see data dictionary and experiment README'}
            if target.suffix=='.csv':
                d=pd.read_csv(target);row.update(n_rows=len(d),n_columns=len(d.columns))
                for c in d:
                    unit=units.get(c,'dimensionless score; not population risk' if c.startswith(('p_','oof','score')) else 'as named / see experiment scope')
                    dictionary.append({'experiment_id':gid,'file':row['file'],'column':c,'dtype':str(d[c].dtype),'unit_or_role':unit,'missing_values':int(d[c].isna().sum())})
            elif target.suffix=='.npy':
                a=np.load(target,mmap_mode='r',allow_pickle=False);row.update(n_rows=a.shape[0],n_columns=a.shape[1] if a.ndim>1 else 1)
                axes=('axis0=model_seed 0..29; axis1=prediction_base.csv row order' if target.name=='seed概率矩阵.npy' else
                      'axis0=bootstrap replicate 0..1999; axis1=draw slot; values index 15_block_labels.json')
                arrays.append({'file':row['file'],'shape':list(a.shape),'dtype':str(a.dtype),'axes':axes})
            elif target.suffix=='.npz':
                with np.load(target,allow_pickle=False) as bundle:
                    arrays.append({'file':row['file'],'arrays':{key:{'shape':list(bundle[key].shape),'dtype':str(bundle[key].dtype)} for key in bundle.files},'axes':'probabilities[candidate,pointid]; candidate and pointid arrays included'})
            assets.append(row)
        scope=('Historical protocol simulations retained including unfavourable results; not the primary matched comparison; no causal attribution of protocol deltas' if gid=='SD13_history_PSLSA' else
               'New matched source-function comparison; shared controlled protocol; three development pairs plus one sequential timing pair' if gid=='SD11_tools' else
               'Old common domain 53911 for seed maps; 30 fixed-setting repeats/seeds' if gid=='SD06_reproducibility' else
               'Full valid domain 54057 for final map/support/heldout curve; old G2/G3 contrast separately labelled' if gid=='SD07_mapping' else
               'Fixed-OOF conditional spatial-block uncertainty, not formal hypothesis testing' if gid=='SD09_statistics' else
               'See canonical sources; inherited verified v3 results plus explicitly identified v4 recomputations')
        expdir=E/'experiments'/gid;expdir.mkdir(parents=True,exist_ok=True)
        entries=[a for a in assets if a['experiment_id']==gid]
        note=f'# {title}\n\n实验编号：{gid}\n\n审稿意见：{", ".join(reviewers)}\n\n范围：{scope}\n\n'
        note+='数据完整保留，不按结果优劣筛选种子、折、参数组合或记录。CSV为可重绘源数据；NPY/NPZ另保留其轴ID与配置。\n\n'
        note+=pd.DataFrame(entries)[['file','canonical_source','n_rows','sha256']].to_markdown(index=False)
        if gid=='SD03_sensitivity':note+='\n\n固定降雨因子曾参与全数据探索性筛选；不能补称预注册。17选择流程在外层训练内选因子，内层分数条件于该选择，不声称选择器本身已完全内层嵌套。'
        if gid=='SD11_tools':note+='\n\n原版PSLSA GUI未实测。实际训练函数的提取源与最小补丁见vendor；历史18协议模拟不作主工具排名。运行时间不等于人工开发耗时。'
        (expdir/'README.md').write_text(note,encoding='utf-8');(dest/'README.md').write_text(note,encoding='utf-8')
        catalog.append({'experiment_id':gid,'title':title,'reviewer_ids':','.join(reviewers),'source_data_directory':'source_data/'+gid,'documentation':'experiments/'+gid+'/README.md','scope':scope,'n_assets':len(entries),'status':'computed_or_inherited_verified'})
    pd.DataFrame(assets).to_csv(SD/'asset_manifest.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dictionary).to_csv(SD/'data_dictionary.csv',index=False,encoding='utf-8-sig')
    (SD/'array_axes.json').write_text(json.dumps(arrays,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(catalog).to_csv(E/'experiments/catalog.csv',index=False,encoding='utf-8-sig')
    figures=[]
    for fid,title,primary,additional,scope in FIGURES:
        figures.append({'figure_id':fid,'title':title,'primary_source_data':'source_data/'+primary,
                        'additional_source_data':'source_data/'+additional,'raw_data_status':'complete observations and repeats retained',
                        'statistical_scope':scope,'script':'scripts/31_draft_figures.py','draft_png':'figures/drafts/'+fid+'.png',
                        'draft_svg':'figures/drafts/'+fid+'.svg','draft_pdf':'figures/drafts/'+fid+'.pdf','publication_status':'draft; final academic styling postponed'})
    pd.DataFrame(figures).to_csv(SD/'figure_manifest.csv',index=False,encoding='utf-8-sig')
    (E/'docs/返修实验总索引.md').write_text('# 返修实验与制图数据总索引\n\n'
        'v4是本轮定稿计算入口，v3及更早版本原目录保留。优先读experiments/catalog.csv、source_data/figure_manifest.csv和results/23_权威数字表.csv。\n\n'
        +pd.DataFrame(catalog).to_markdown(index=False)+'\n\n## 制图数据\n\n'+pd.DataFrame(figures).to_markdown(index=False)+
        '\n\n所有图均为草图。最终制图不必从PNG反读数据，直接按图件索引读取完整CSV/NPY/NPZ、字段字典和实验README。\n'
        '\nGUI、原版PSLSA交互和GitHub在线发布未被冒充为已完成；代码包、源函数适配和冻结输入计算均可本地复核。',encoding='utf-8')
    fixes=[('F01','completed','540/24 current-protocol grid diagnostic, 564 candidates and all inner OOF saved'),
           ('F02','completed within declared scope','actual PSLSA source method executed; minimal controlled adapter; identical dataset/protocol; no general human-time claim'),
           ('F03','completed','246/351 row occurrences; 212/306 unique records; 45 repeated rows; legacy protocol labelled'),
           ('F04','completed','pixel area from affine determinant; complete raster-mask area corrected'),
           ('F05','completed','formal outputs use mean/SD and conditional block intervals; no uncalibrated p/q'),
           ('F06','completed by disclosure','selection history preserved; conditional evaluation, no invented preregistration'),
           ('F07','completed','54057 main domain versus 53911 common-domain stability explicitly separated'),
           ('F08','local release prepared','frozen-input workflow, pinned dependencies and portable tests; online GitHub publishing not performed'),
           ('F09','completed','declared dependencies, required outputs, hashes, verified resume and fault-injection tests'),
           ('F10','completed by scope','balanced-sample susceptibility score, not regional event probability'),
           ('F11','completed','independent same-definition terrain calculations and every compared pixel exported'),
           ('F12','completed','rebuild summaries from final data with scope; historical reports archived')]
    fx=pd.DataFrame(fixes,columns=['issue','status','evidence'])
    (E/'docs/修复验收与审稿意见映射.md').write_text('# 修复与审稿意见数据映射\n\n'+fx.to_markdown(index=False)+
        '\n\n每条审稿意见的数据支撑见实验总索引。正文、文献和学术版图件后续据此重构；本轮不声称已完成整套稿件修订。'
        '\nR1.11：适合明确规则的批处理、模型实验组织与代码复用；已有成熟稳定工具时优先复用，不为固定任务反复生成代码；无专家审查时不宜直接用于工程安全决策。',encoding='utf-8')
    # Every previous revision remains classified and locatable; do not copy duplicate historical runs into the release data.
    history=[]
    for ver in ['返修实验','返修实验_v2','返修实验_v3']:
        parent=E.parent/ver
        for p in parent.rglob('*'):
            if p.is_file() and p.suffix.lower() in ['.csv','.json','.md','.py','.png','.npy','.joblib'] and not any(x in p.parts for x in ['__pycache__','runtime_deps']):
                history.append({'version':ver,'relative_path':'../'+ver+'/'+p.relative_to(parent).as_posix(),'bytes':p.stat().st_size,'category':p.parent.name,'status':'historical; not automatically the current paper result'})
    pd.DataFrame(history).to_csv(E/'provenance/history_file_index.csv',index=False,encoding='utf-8-sig')
    print('organized',len(catalog),'experiments,',len(assets),'source assets,',len(dictionary),'fields,',len(figures),'draft figure contracts',flush=True)

if __name__=='__main__':main()
