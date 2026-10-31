"""Preserve repeat-level importance, per-cell GIS checks, and cache equivalence evidence."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import roc_auc_score
import exp_lib as L


def main():
    R=Path(L.RES);D=Path(L.DATA);b=L.load_base();F=L.GROUPS['G2'];y=b.landslide.to_numpy(dtype=int)
    o=pd.read_csv(R/'03_OOF预测.csv');ref=o[(o.group=='G2')&(o.model=='XGBoost')].set_index('pointid')
    bp=pd.read_csv(R/'03_最优超参数.csv');bp=bp[(bp.group=='G2')&(bp.model=='XGBoost')].set_index('fold')
    # Verify the only numerical-core optimization against the frozen official output.
    result=L.run_spatial_cv(b,F,'XGBoost',seed=42,inner_jobs=4)
    diff=float(np.abs(result['oof']-ref.reindex(b.pointid).oof.to_numpy()).max())
    new=result['best_params'].set_index('fold');checks=[]
    for k in range(5):
        checks.append({'fold':k,'selected_params_match':json.loads(new.loc[k,'params'])==json.loads(bp.loc[k,'params']),
                        'threshold_difference':float(new.loc[k,'threshold']-bp.loc[k,'threshold']),
                        'global_oof_max_difference':diff})
    pd.DataFrame(checks).to_csv(R/'27_缓存与正式主结果等价性.csv',index=False,encoding='utf-8-sig')
    if diff>1e-9 or not all(r['selected_params_match'] for r in checks):raise RuntimeError('Cached numerical core changed official results')
    raw=[]
    for k in range(5):
        tr=b[b.fold!=k];te=b[b.fold==k];yt=te.landslide.to_numpy()
        xa,enc=L.build_matrix(tr,F);xt,_=L.build_matrix(te,F,enc=enc,fit=False)
        model=L.instantiate('XGBoost',json.loads(bp.loc[k,'params']),42,n_jobs=1).fit(xa,tr.landslide)
        base=roc_auc_score(yt,model.predict_proba(xt)[:,1]);rng=np.random.default_rng(1000+k)
        for factor in F:
            names=(['aspect_sin','aspect_cos','aspect_flat'] if factor=='aspect' else
                   [c for c in xt if c.startswith('lu_')] if factor=='land_use' else [factor])
            ix=[xt.columns.get_loc(c) for c in names]
            for repeat in range(20):
                perm=rng.permutation(len(xt));xp=xt.copy();xp.iloc[:,ix]=xt.iloc[perm,ix].to_numpy()
                perturbed=roc_auc_score(yt,model.predict_proba(xp)[:,1])
                raw.append({'model':'XGBoost','fold':k,'factor':factor,'repeat':repeat,
                             'baseline_auc':base,'permuted_auc':perturbed,'auc_drop':base-perturbed,'permutation_seed':1000+k})
    raw=pd.DataFrame(raw);raw.to_csv(R/'09_重要性置换明细.csv',index=False,encoding='utf-8-sig')
    agg=raw.groupby(['model','fold','factor']).agg(importance=('auc_drop','mean'),std=('auc_drop','std'),n_repeat=('repeat','count'),base_auc=('baseline_auc','first')).reset_index()
    agg.to_csv(R/'09_重要性.csv',index=False,encoding='utf-8-sig')
    agg.groupby('factor').agg(importance_mean=('importance','mean'),importance_std_across_folds=('importance','std'),n_folds=('fold','count')).sort_values('importance_mean',ascending=False).to_csv(R/'09_重要性汇总.csv',encoding='utf-8-sig')
    print('cache equality',diff,'importance repeats',len(raw),flush=True)
    terrain()


def terrain():
    R=Path(L.RES);D=Path(L.DATA)
    # Independent, same-definition terrain reconstruction. Cached GDAL rasters are references only.
    with rasterio.open(Path(L.RASTERS)/'dem.tif') as s:
        dem=s.read(1).astype(float);dem[dem==s.nodata]=np.nan;res=s.transform.a
    from scipy.ndimage import convolve
    dx=convolve(dem,np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)/8,mode='nearest')
    dy=convolve(dem,np.array([[-1,-2,-1],[0,0,0],[1,2,1]],float)/8,mode='nearest')
    sl=np.degrees(np.arctan(np.sqrt(dx*dx+dy*dy)/res));a=np.degrees(np.arctan2(-dy,dx))
    asp=np.where(a<0,90-a,np.where(a>90,360-a+90,90-a));asp[sl==0]=-1
    padded=np.pad(dem,1,constant_values=np.nan);rough=np.zeros_like(dem)
    for dr in range(3):
        for dc in range(3):
            if (dr,dc)==(1,1):continue
            z=padded[dr:dr+dem.shape[0],dc:dc+dem.shape[1]]
            rough+=np.abs(np.where(np.isfinite(z),z,dem)-dem)/8
    valid=np.isfinite(dem);rr,cc=np.where(valid)
    table=pd.DataFrame({'row':rr,'col':cc,'DEM':dem[valid],'numpy_slope':sl[valid],
                         'numpy_aspect':asp[valid],'numpy_mean_abs_neighbour_difference':rough[valid]})
    strict_interior=np.isfinite(dx)&np.isfinite(dy)&valid
    table['strict_3x3_interior']=strict_interior[valid]
    table['flat_aspect']=sl[valid]==0
    summary=[]
    for f,arr in [('slope',sl),('aspect',asp),('roughness',rough)]:
        with rasterio.open(Path(L.RASTERS)/(f+'.tif')) as src:reference=src.read(1).astype(float)
        table['reference_'+f]=reference[valid]
        mask=valid&np.isfinite(arr)
        if f=='aspect':mask&=reference>=0
        error=arr[mask]-reference[mask]
        if f=='aspect':error=(error+180)%360-180
        summary.append({'factor':f,'domain':'available DEM mask','n_compared':int(mask.sum()),'n_unavailable':int(valid.sum()-mask.sum()),
                         'mae':float(np.abs(error).mean()),'rmse':float(np.sqrt(np.mean(error**2))),
                         'max_absolute_error':float(np.abs(error).max()),
                         'definition':'mean absolute elevation difference to 8 neighbours; missing neighbour=center' if f=='roughness' else 'Horn 3x3; aspect excludes flat cells'})
        if f=='roughness':
            for scope,region in [('strict 3x3 interior',strict_interior),('boundary requiring outside-mask neighbours',valid&~strict_interior)]:
                err=arr[region]-reference[region]
                summary.append({'factor':f,'domain':scope,'n_compared':int(region.sum()),'n_unavailable':0,
                    'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt(np.mean(err**2))),
                    'max_absolute_error':float(np.abs(err).max()),'definition':'mean absolute neighbour difference; outside-mask source elevations are not available in frozen DEM'})
        cached=D/'_gdaldem'/f'{f}.tif'
        if cached.exists():
            with rasterio.open(cached) as src:vals=src.read(1).astype(float);vals[vals==src.nodata]=np.nan
            table['archived_gdal_'+f]=vals[valid]
    table.to_csv(R/'10_逐像元对照.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(R/'10_同定义独立复算.csv',index=False,encoding='utf-8-sig')
    (R/'10_开源实现范围.md').write_text('# 开源实现的数值核对范围\n\n'+pd.DataFrame(summary).to_markdown(index=False)+
        '\n\n本轮从现有DEM独立重算。逐像元数据同时保存原因子、NumPy结果及历史GDAL对照；历史GDAL栅格不是本轮重新执行GDAL。'
        '\nroughness的实际公式是8邻域平均绝对高差，不与GDAL的最大高差同名指标混作同一定义，也不误称Riley平方和开方指标。'
        '\n完整掩膜边界缺少掩膜外高程，不能验证其与原生产过程的邻域补值一致性；内部与边界分别报告，边界差异不隐藏。'
        '\n仅比较明确相同算法与有效区域，边界及平地约定单列；不据此宣称全部GIS流水线已经等价复现。',encoding='utf-8')
    print('terrain',summary,flush=True)

if __name__=='__main__':
    if '--terrain-only' in __import__('sys').argv:terrain()
    else:main()
