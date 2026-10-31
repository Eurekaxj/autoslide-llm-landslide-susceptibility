# -*- coding: utf-8 -*-
"""
17a_rain_candidates.py — 12 个降雨候选指标的**同一生产路径**提取（v3 · R03）
=====================================================================
v2 审查 R03 指出三处不可比：
  ① 候选 rx5day 与主表 rx5day 最大差 1.09 mm、raincv 差 0.053 —
     说明候选与固定分支不是同一产品；
  ② 候选 raincv 与主表 raincv_krig 几乎逐点相等（4.4e-16），
     即"最近邻"声明不实，实际是克里金版；
  ③ 候选哈希用 Python 内置 hash，只哈希名字且跨进程不稳定。

v3 做法：全部 12 个候选统一从 `Rasters_native/rain_*_0.05deg.tif`（0.05° 源格点）
按**最近邻**重采样到 432 m DEM 网格后取点——这与主表 rx5day/raincv 的生成路径
（14_rainfall_nearest.py）完全一致，因此同名指标必须与主表逐点相等；
若不等则直接报错，而不是用别的版本顶替。

输出： data/rain_candidates.csv（702 点 × 12 候选 + 内容哈希）
       results/17a_候选一致性.csv
"""
import os
import sys
import hashlib

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

LOG = os.path.join(L.LOG, "17a_rain_candidates.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 17a 降雨候选统一提取（R03）===", LOG)

NATIVE = L.configured_path("native_rainfall", required=True)
CAND = {
    "rx5day":   ("rain_rx5day_0.05deg.tif",   "强度"),
    "rx1day":   ("rain_rx1day_0.05deg.tif",   "强度"),
    "r99p":     ("rain_r99p_0.05deg.tif",     "强度"),
    "r95p":     ("rain_r95p_0.05deg.tif",     "强度"),
    "r95frac":  ("rain_r95frac_0.05deg.tif",  "强度"),
    "sdii":     ("rain_sdii_0.05deg.tif",     "强度"),
    "raincv":   ("rain_cv_0.05deg.tif",       "变率"),
    "pci":      ("rain_pci_0.05deg.tif",      "变率"),
    "cdd":      ("rain_cdd_0.05deg.tif",      "变率"),
    "cwd":      ("rain_cwd_0.05deg.tif",      "变率"),
    "prcptot":  ("rain_prcptot_0.05deg.tif",  "量级"),
    "raindays": ("rain_raindays_0.05deg.tif", "量级"),
}

df = L.load_base()
with rasterio.open(os.path.join(L.RASTERS, "dem.tif")) as d:
    T, H, W, crs = d.transform, d.height, d.width, d.crs
    dem = d.read(1)
    dnd = d.nodata
valid = dem != dnd

rows, cols = df["row"].values, df["col"].values


def warp_nearest(path):
    """0.05° 源格点 → 432 m DEM 网格（最近邻），与 14_rainfall_nearest.py 同法。"""
    with rasterio.open(path) as s:
        a = s.read(1).astype("float64")
        a[a == s.nodata] = np.nan
        stf, scrs = s.transform, s.crs
    dst = np.full((H, W), np.nan, dtype="float64")
    reproject(a, dst, src_transform=stf, src_crs=scrs,
              dst_transform=T, dst_crs=crs, resampling=Resampling.nearest,
              src_nodata=np.nan, dst_nodata=np.nan)
    dst[~valid] = np.nan
    return dst


out = df[["pointid", "row", "col", "landslide"]].copy()
hashes, checks = {}, []
for name, (fn, cat) in CAND.items():
    p = os.path.join(NATIVE, fn)
    if not os.path.exists(p):
        # R03：找不到源指标就明确不可比，不得用克里金版替代
        L.log(f"[不可比] {name}: 缺源格点 {fn}，本候选不参与筛选", LOG)
        continue
    arr = warp_nearest(p)
    v = np.asarray(arr[rows, cols], dtype=np.float64)
    v = np.asarray(v, dtype=np.float32).astype(np.float64)   # 共同数值规范（R01）
    out[name] = v
    hashes[name] = hashlib.sha256(
        np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest()
    L.log(f"{name}: 缺失={int(np.isnan(v).sum())} "
          f"[{np.nanmin(v):.3f}, {np.nanmax(v):.3f}]", LOG)

AVAIL = [c for c in CAND if c in out.columns]
L.log(f"可用候选 {len(AVAIL)}/{len(CAND)}", LOG)

# —— 与主表同名指标的一致性核对（R03 核心验收）——
# 固定分支（G2）实际使用的是 rx5day / raincv 两列，必须逐点相同；
# prcptot 在建模表中的版本是插值（克里金）产物，与本路径不同，仅作记录，
# 建模统一改用本脚本的最近邻版本（另存 prcptot_near）。
STRICT = ("rx5day", "raincv")
for name in ("rx5day", "raincv", "prcptot"):
    if name not in out.columns or name not in df.columns:
        continue
    a = L.canon_numeric(df[name].values)
    b = L.canon_numeric(out[name].values)
    d = np.abs(a - b)
    checks.append({"factor": name, "n": len(a),
                   "max_abs_diff": float(np.nanmax(d)),
                   "n_unequal": int(np.sum(a != b)),
                   "strict": name in STRICT,
                   "main_source": "modeling_base.csv",
                   "cand_source": "Rasters_native + 最近邻（本脚本）"})
CK = pd.DataFrame(checks)
CK.to_csv(os.path.join(L.RES, "17a_候选一致性.csv"), index=False,
          encoding="utf-8-sig")
L.log("\n" + CK.to_string(index=False), LOG)

bad = CK[(CK["strict"]) & (CK["n_unequal"] > 0)]
if len(bad):
    raise SystemExit(f"[失败] 固定分支使用的候选与主表不一致：\n{bad.to_string(index=False)}")
L.log("固定分支所用 rx5day / raincv 与主表逐点一致（0 不等），可比性成立", LOG)
if "prcptot" in out.columns:
    L.log("说明：主表 prcptot 为插值版（与本路径最大差见上表），"
          "建模统一改用最近邻版 prcptot_near", LOG)
    out["prcptot_near"] = out["prcptot"].values

out.to_csv(os.path.join(L.DATA, "rain_candidates.csv"), index=False,
           encoding="utf-8-sig")
L.save_json({"candidates": {k: {"file": CAND[k][0], "category": CAND[k][1],
                                "values_sha256": hashes.get(k)}
                            for k in AVAIL},
             "numeric_spec": "float32-canonical",
             "alignment": "0.05deg native -> nearest -> 432m DEM grid",
             "n_points": int(len(out))},
            os.path.join(L.CFG, "rain_candidates_manifest.json"))
L.log("=== 17a 完成 ===", LOG)
