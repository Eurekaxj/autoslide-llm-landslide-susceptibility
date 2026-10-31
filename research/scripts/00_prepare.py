# -*- coding: utf-8 -*-
"""
00_prepare.py — 阶段一：数据冻结与结构验收（对应审稿人1-8可重复性 / 审稿人2-13）
================================================================================
1. 输入文件验收（存在性、数量、几何一致性、缺失、标签、ID唯一性）
2. 生成 run_manifest.json（含输入文件哈希）
3. 由坐标生成 row / col / cell_id / spatial_block_id / fold
4. 把辅助栅格（2019年累计降雨、年均温、克里金版Rx5day/CV、砂粒、容重）提取到
   702 样本与全区预测点，供消融与敏感性实验使用
输出： data/modeling_base.csv, data/prediction_base.csv,
       config/run_manifest.json, results/00_验收报告.md
"""
import os
import sys
import json
import time
import platform

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

LOG = os.path.join(L.LOG, "00_prepare.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 00 阶段一：数据冻结与结构验收 ===", LOG)

# ------------------------------------------------------------
# 1. 文件验收
# ------------------------------------------------------------
L.log("--- 1. 输入文件验收 ---", LOG)
accept = []
rasters = sorted([f for f in os.listdir(L.RASTERS) if f.endswith(".tif")])
L.log(f"因子栅格数量: {len(rasters)}", LOG)
geom_ref = None
for f in rasters:
    with rasterio.open(os.path.join(L.RASTERS, f)) as s:
        g = (s.width, s.height, round(s.transform.a, 4), round(s.transform.e, 4),
             s.crs.to_string(), s.nodata, int((s.read(1) != s.nodata).sum()))
    if geom_ref is None:
        geom_ref = g[:6]
    accept.append({"file": f, "width": g[0], "height": g[1], "res_x": g[2],
                   "res_y": g[3], "crs": g[4], "nodata": g[5], "valid": g[6],
                   "geom_consistent": g[:6] == geom_ref})
acc_df = pd.DataFrame(accept)
acc_df.to_csv(os.path.join(L.RES, "00_栅格验收.csv"), index=False,
              encoding="utf-8-sig")
L.log(f"15个栅格几何/掩膜完全一致: {bool(acc_df['geom_consistent'].all())}; "
      f"有效像元均为 {sorted(acc_df['valid'].unique())}", LOG)

la = pd.read_csv(os.path.join(L.POINTS, "landslide_factors.csv"))
nl = pd.read_csv(os.path.join(L.POINTS, "nonlandslide_factors.csv"))
pred = pd.read_csv(os.path.join(L.POINTS, "prediction_factors.csv"))
L.log(f"滑坡点 {len(la)} / 非滑坡点 {len(nl)} / 全区预测点 {len(pred)}", LOG)

checks = {}
df702 = pd.concat([la.assign(landslide=1), nl.assign(landslide=0)],
                  ignore_index=True)
FACTORS15 = ["DEM", "slope", "aspect", "plan", "profile", "NDVI", "land_use",
             "rx5day", "raincv", "roughness", "road", "river", "building",
             "soil_clay", "soil_silt"]
checks["n_total"] = len(df702)
checks["n_landslide"] = int((df702["landslide"] == 1).sum())
checks["n_nonlandslide"] = int((df702["landslide"] == 0).sum())
checks["labels_unique"] = sorted(df702["landslide"].unique().tolist())
checks["nan_in_factors"] = int(df702[FACTORS15].isna().sum().sum())
checks["valid_factors_all15"] = bool((df702["valid_factors"] == 15).all())
checks["pointid_unique"] = bool(df702["pointid"].is_unique)
checks["pred_nan"] = int(pred[FACTORS15].isna().sum().sum())
checks["pred_valid_all15"] = bool((pred["valid_factors"] == 15).all())
checks["pred_n"] = len(pred)
L.log(f"验收: {checks}", LOG)
# ------------------------------------------------------------
# 2. run_manifest
# ------------------------------------------------------------
import sklearn, xgboost, catboost, lightgbm, scipy, matplotlib, shap, rasterio as rio
manifest = {
    "run_name": "返修实验-最终因子集-G2",
    "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_version": "最终因子集 2026-10-27",
    "n_landslide": int((df702["landslide"] == 1).sum()),
    "n_nonlandslide": int((df702["landslide"] == 0).sum()),
    "n_prediction_points": int(len(pred)),
    "sampling_seed": 42,
    "spatial_fold_seed": 42,
    "model_seed": 42,
    "factors": FACTORS15,
    "grid": {"width": geom_ref[0], "height": geom_ref[1],
             "res_m": geom_ref[2], "crs": geom_ref[4]},
    "base_year": 2019,
    "rainfall_window": "1990-2019",
    "software": {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "catboost": catboost.__version__, "lightgbm": lightgbm.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "shap": shap.__version__, "rasterio": rio.__version__,
        "platform": platform.platform(),
    },
    "input_hashes": {},
}
for f in ["landslide_factors.csv", "nonlandslide_factors.csv",
          "prediction_factors.csv"]:
    manifest["input_hashes"][f] = L.file_hash(os.path.join(L.POINTS, f))
for f in rasters:
    manifest["input_hashes"]["raster/" + f] = L.file_hash(
        os.path.join(L.RASTERS, f))
L.save_json(manifest, os.path.join(L.CFG, "run_manifest.json"))
L.log("run_manifest.json 已生成", LOG)

# ------------------------------------------------------------
# 3. 坐标、cell_id、空间块、折叠
# ------------------------------------------------------------
L.log("--- 3. 生成 row/col/cell_id/空间块/折叠 ---", LOG)
with rasterio.open(os.path.join(L.RASTERS, "dem.tif")) as s:
    T = s.transform
    prof = s.profile

tf = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
for df in (df702, pred):
    x, y = tf.transform(df["Lon"].values, df["Lat"].values)
    df["X"] = x
    df["Y"] = y
    col = ((x - T.c) / T.a).astype(int)
    row = ((y - T.f) / T.e).astype(int)
    df["col"] = np.clip(col, 0, prof["width"] - 1)
    df["row"] = np.clip(row, 0, prof["height"] - 1)
    df["cell_id"] = df["row"].astype(str) + "_" + df["col"].astype(str)

df702["spatial_block_id"] = (df702["row"] // 20).astype(str) + "_" + (df702["col"] // 20).astype(str)
# 【v2 · C04】fold 由固定的 (row,col)→block→fold 查找表决定，与样本集合无关
df702["fold"] = L.make_spatial_folds(df702, n_splits=5, seed=42, block_cells=20)
df702["fold_random"] = L.make_random_folds(df702, n_splits=5, seed=42)

# 【v2 · C03】同像元多记录统计（保留全部记录，仅在分组时按 cell 处理）
checks["n_unique_cells"] = int(df702["cell_id"].nunique())
checks["n_dup_records"] = int(len(df702) - df702["cell_id"].nunique())
checks["n_unique_cells_landslide"] = int(
    df702.loc[df702["landslide"] == 1, "cell_id"].nunique())
checks["n_unique_cells_background"] = int(
    df702.loc[df702["landslide"] == 0, "cell_id"].nunique())

# 【v2 · C02】预测域一致性：有效栅格像元 - 预测点数 = 已剔除像元
with rasterio.open(os.path.join(L.RASTERS, "dem.tif")) as s:
    _dem = s.read(1)
    _valid_cells = int((_dem != s.nodata).sum())
checks["valid_raster_cells"] = _valid_cells
checks["pred_domain_cells"] = int(len(pred))
checks["pred_domain_missing_cells"] = int(_valid_cells - len(pred))
_pred_cells = set(pred["cell_id"])
_ls_in_pred = int(df702.loc[df702["landslide"] == 1, "cell_id"].isin(_pred_cells).sum())
checks["landslide_records_inside_pred_domain"] = _ls_in_pred
checks["landslide_records_outside_pred_domain"] = int(
    (df702["landslide"] == 1).sum() - _ls_in_pred)
L.log(f"同像元统计: 唯一cell={checks['n_unique_cells']}, 重复记录="
      f"{checks['n_dup_records']}", LOG)
L.log(f"预测域: 有效栅格 {_valid_cells} - 预测点 {len(pred)} = 缺失 "
      f"{checks['pred_domain_missing_cells']}；滑坡点落在预测域内 {_ls_in_pred}/351", LOG)

# 折叠泄漏检查
leak = []
for k in sorted(df702["fold"].unique()):
    tr = set(df702.loc[df702["fold"] != k, "cell_id"])
    te = set(df702.loc[df702["fold"] == k, "cell_id"])
    inter = tr & te
    leak.append({"fold": k, "n_train": int((df702["fold"] != k).sum()),
                 "n_test": int((df702["fold"] == k).sum()),
                 "n_test_pos": int(((df702["fold"] == k) & (df702["landslide"] == 1)).sum()),
                 "overlap_cell": len(inter)})
leak_df = pd.DataFrame(leak)
leak_df.to_csv(os.path.join(L.RES, "00_空间折叠.csv"), index=False,
               encoding="utf-8-sig")
L.log(f"空间折叠: \n{leak_df.to_string(index=False)}", LOG)
L.log(f"同像元跨训练/测试泄漏像元数合计: {int(leak_df['overlap_cell'].sum())}", LOG)
manifest["fold_sizes"] = leak_df.to_dict("records")
manifest["cellid_leakage"] = int(leak_df["overlap_cell"].sum())
manifest["fold_protocol"] = ("固定 (row,col)->block->fold 查找表；block=20x20 像元，"
                             "按 DEM 有效像元块几何中心 KMeans(seed=42) 聚 5 类；"
                             "与样本集合无关（C04）")
manifest["n_unique_cells"] = checks["n_unique_cells"]
manifest["n_dup_records"] = checks["n_dup_records"]
manifest["prediction_domain_cells"] = checks["pred_domain_cells"]
manifest["prediction_domain_missing_cells"] = checks["pred_domain_missing_cells"]
manifest["stage"] = "base（00_prepare 生成的建模基础表）"

# ------------------------------------------------------------
# 4. 辅助栅格提取
# ------------------------------------------------------------
L.log("--- 4. 辅助因子提取（年累计降雨 / 年均温 / 克里金降雨 / 砂粒 / 容重）---", LOG)
AUX = {
    "rainfall": os.path.join(L.AUX_RASTERS, "rainfall_432.49m.tif"),
    "temyear": os.path.join(L.AUX_RASTERS, "temyear_432.49m.tif"),
    "rx5day_krig": os.path.join(L.AUX_RASTERS, "rx5day_432.49m_kriged.tif"),
    "raincv_krig": os.path.join(L.AUX_RASTERS, "raincv_432.49m_kriged.tif"),
    "soil_sand": os.path.join(L.ROOT, "西安市arcgis文件", "soil_sand.tif"),
    "soil_bd": os.path.join(L.ROOT, "西安市arcgis文件", "soil_bd.tif"),
    "prcptot": os.path.join(L.AUX_RASTERS, "rain_prcptot_432.49m.tif"),
}


def sample_raster(path, xs, ys):
    with rasterio.open(path) as s:
        if (s.width, s.height, s.crs) != (prof["width"], prof["height"], prof["crs"]):
            raise ValueError(f"{path} 与模板网格不一致 {s.width}x{s.height}")
        nod = s.nodata if s.nodata is not None else -9999.0
        arr = s.read(1)
        r = ((ys - s.transform.f) / s.transform.e).astype(int)
        c = ((xs - s.transform.c) / s.transform.a).astype(int)
        r = np.clip(r, 0, s.height - 1)
        c = np.clip(c, 0, s.width - 1)
        v = arr[r, c].astype(float)
        v[v == nod] = np.nan
        v[np.abs(v) > 1e30] = np.nan
        return v


for name, path in AUX.items():
    if not os.path.exists(path):
        L.log(f"[跳过] 缺失 {name}: {path}", LOG)
        continue
    for df, tag in ((df702, "702"), (pred, "pred")):
        df[name] = sample_raster(path, df["X"].values, df["Y"].values)
    nn = int(df702[name].isna().sum())
    L.log(f"{name}: 702缺失={nn}, 范围 "
          f"[{np.nanmin(df702[name]):.3f}, {np.nanmax(df702[name]):.3f}]", LOG)

# soil_bd / sand 单位还原（CSDL 与 clay/silt 同样 ÷100）
for c in ["soil_sand"]:
    if c in df702 and df702[c].max() > 100:
        L.log(f"[注意] {c} 量级异常，未做缩放还原", LOG)

df702.to_csv(os.path.join(L.DATA, "modeling_base.csv"), index=False,
             encoding="utf-8-sig")
pred.to_csv(os.path.join(L.DATA, "prediction_base.csv"), index=False,
            encoding="utf-8-sig")
manifest["output_files"] = {
    # C01 验收要求完整 64 位 SHA-256（父输入亦须完整，供 06 登记 parent）
    "modeling_base.csv": L.file_hash(os.path.join(L.DATA, "modeling_base.csv"), length=64),
    "prediction_base.csv": L.file_hash(os.path.join(L.DATA, "prediction_base.csv"), length=64),
}
manifest["output_files_stage"] = "base（00_prepare 生成；06 之后另登记 final 版本）"
L.save_json(manifest, os.path.join(L.CFG, "run_manifest.json"))
L.save_json(checks, os.path.join(L.RES, "00_验收检查.json"))

# ------------------------------------------------------------
# 5. 报告
# ------------------------------------------------------------
rep = ["# 阶段一 数据冻结与结构验收报告\n",
       f"- 运行时间：{manifest['run_date']}",
       f"- 因子栅格：{len(rasters)} 个，尺寸 {geom_ref[0]}×{geom_ref[1]}，"
       f"分辨率 {geom_ref[2]} m，{geom_ref[4]}，有效像元 {geom_ref[5] if False else int(acc_df['valid'].iloc[0])}",
       f"- 几何与掩膜完全一致：{bool(acc_df['geom_consistent'].all())}",
       f"- 滑坡点 {checks['n_landslide']} / 非滑坡点 {checks['n_nonlandslide']} / 建模全集 {checks['n_total']}",
       f"- 全区预测点 {checks['pred_n']}",
       f"- 15 因子缺失值：{checks['nan_in_factors']}；valid_factors 全为 15：{checks['valid_factors_all15']}",
       f"- 点 ID 唯一：{checks['pointid_unique']}；标签取值：{checks['labels_unique']}",
       "", "## 空间折叠（5 折，空间块 KMeans，seed=42）", "",
       leak_df.to_markdown(index=False), "",
       f"- 同一 cell_id 跨训练/测试像元数：{int(leak_df['overlap_cell'].sum())}（0 表示无同像元泄漏）",
       "", "## 辅助因子提取", ""]
for name in AUX:
    if name in df702:
        rep.append(f"- `{name}`：702 缺失 {int(df702[name].isna().sum())}，"
                   f"范围 [{np.nanmin(df702[name]):.4g}, {np.nanmax(df702[name]):.4g}]")
with open(os.path.join(L.RES, "00_验收报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(rep))
L.log("=== 00 完成 ===", LOG)
