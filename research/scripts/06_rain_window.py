# -*- coding: utf-8 -*-
"""
06_rain_window.py — 阶段四：降雨时间窗口敏感性（对应审稿人2-第7条）
==================================================================
审稿人原话：年累计降雨的使用过于笼统，滑坡通常由短期强降雨引发，
建议纳入长期日降雨变率。

本脚本补充「统计窗口」维度的敏感性：
  - W30：1990-2019（30年，主方案，已在最终因子集中）
  - W10：2010-2019（10年）
  - W01：2019 单年（诊断用，不作为主因子）
统一在 0.05° 源网格上计算 Rx5day 与日降雨 CV，再最近邻对齐到 432 m 建模网格，
与主方案口径完全一致（先算后投影，避免极值被抹平）。

用法： python 06_rain_window.py
"""
import os
import sys
import glob
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

SRC_DIR = L.configured_path("daily_rainfall", required=True)
WET_THR = 1.0
PAD_DEG = 0.15

LOG = os.path.join(L.LOG, "06_rain_window.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 06 降雨时间窗口敏感性 ===", LOG)

OUT = os.path.join(L.DATA, "_rain_window")
os.makedirs(OUT, exist_ok=True)

# ---------- 网格与窗口 ----------
with rasterio.open(os.path.join(L.RASTERS, "dem.tif")) as d:
    dem_meta = dict(crs=d.crs, transform=d.transform, width=d.width,
                    height=d.height, nodata=d.nodata)
    dem_arr = d.read(1).astype("float32")
    dem_valid = dem_arr != d.nodata
    b = transform_bounds(d.crs, "EPSG:4326", *d.bounds)

probe = sorted(glob.glob(os.path.join(SRC_DIR, "*.tif")))[0]
with rasterio.open(probe) as s:
    src_crs, src_nodata = s.crs, s.nodata
    win = from_bounds(b[0] - PAD_DEG, b[1] - PAD_DEG, b[2] + PAD_DEG, b[3] + PAD_DEG,
                      transform=s.transform).round_offsets().round_lengths()
    win_tf = s.window_transform(win)
hh, ww = int(win.height), int(win.width)
L.log(f"源窗口 {hh}x{ww} @0.05°", LOG)


def load_years(y0, y1):
    out = {}
    t0 = time.time()
    n = 0
    for y in range(y0, y1 + 1):
        files = sorted(glob.glob(os.path.join(SRC_DIR, f"{y}-*.tif")))
        if not files:
            L.log(f"[警告] {y} 无文件", LOG)
            continue
        arr = np.empty((len(files), hh, ww), dtype="float32")
        for i, f in enumerate(files):
            with rasterio.open(f) as s:
                a = s.read(1, window=win).astype("float32")
            a[a == src_nodata] = np.nan
            np.clip(a, 0, None, out=a)
            arr[i] = a
        out[y] = arr
        n += len(files)
        L.log(f"  {y}: {len(files)} 天已载入（累计 {n}）", LOG)
    L.log(f"载入完成 {n} 天，{time.time()-t0:.1f}s", LOG)
    return out


def rolling_sum(a, k):
    cs = np.concatenate([np.zeros((1,) + a.shape[1:], dtype="float64"),
                         np.nancumsum(a, axis=0, dtype="float64")], axis=0)
    return cs[k:] - cs[:-k]


def indices_for(data):
    """返回多年平均的 rx5day 与 cv（源网格）。"""
    years = sorted(data)
    rx, cv = [], []
    for y in years:
        a = data[y]
        rx.append(np.nanmax(rolling_sum(a, 5), axis=0))
        mean = np.nanmean(a, axis=0)
        cv.append(np.nanstd(a, axis=0) / np.maximum(mean, 1e-6))
    return np.nanmean(np.stack(rx, 0), 0), np.nanmean(np.stack(cv, 0), 0)


def warp_to_dem(arr):
    dst = np.full((dem_meta["height"], dem_meta["width"]), np.nan, dtype="float32")
    src = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    reproject(src, dst, src_transform=win_tf, src_crs=src_crs,
              src_nodata=-9999.0, dst_transform=dem_meta["transform"],
              dst_crs=dem_meta["crs"], dst_nodata=np.nan,
              resampling=Resampling.nearest)
    dst[~dem_valid] = np.nan
    return dst


def write_tif(arr, name):
    p = os.path.join(OUT, f"rain_{name}.tif")
    a = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    with rasterio.open(p, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="float32",
                       crs=dem_meta["crs"], transform=dem_meta["transform"],
                       nodata=-9999.0) as dst:
        dst.write(a, 1)
    return p


# W30（1990-2019）已由 09_daily_rainfall_indices.py 生成在最终因子集中，
# 此处直接复用其栅格作为 W30，避免重复载入 30 年逐日数据。
WINDOWS = {"W10": (2010, 2019), "W01": (2019, 2019)}
results = {}
with rasterio.open(os.path.join(L.RASTERS, "rx5day.tif")) as s:
    RX30 = s.read(1).astype(float); RX30[RX30 == s.nodata] = np.nan
with rasterio.open(os.path.join(L.RASTERS, "raincv.tif")) as s:
    CV30 = s.read(1).astype(float); CV30[CV30 == s.nodata] = np.nan
results["W30"] = {"rx5day": RX30, "raincv": CV30}
L.log("W30 复用最终因子集既有栅格（1990-2019）", LOG)
for name, (y0, y1) in WINDOWS.items():
    t0 = time.time()
    data = load_years(y0, y1)
    rx, cv = indices_for(data)
    RX = warp_to_dem(rx)
    CV = warp_to_dem(cv)
    write_tif(RX, f"rx5day_{name}")
    write_tif(CV, f"raincv_{name}")
    results[name] = {"rx5day": RX, "raincv": CV}
    L.log(f"{name} ({y0}-{y1}): rx5day [{np.nanmin(RX):.3f},{np.nanmax(RX):.3f}] "
          f"cv [{np.nanmin(CV):.3f},{np.nanmax(CV):.3f}] ({time.time()-t0:.0f}s)", LOG)
    del data

# ---------- 提取到点 ----------
df = L.load_base()
pred = L.load_prediction()


def sample(arr, xs, ys):
    r = np.clip(((ys - dem_meta["transform"].f) / dem_meta["transform"].e).astype(int),
                0, dem_meta["height"] - 1)
    c = np.clip(((xs - dem_meta["transform"].c) / dem_meta["transform"].a).astype(int),
                0, dem_meta["width"] - 1)
    return arr[r, c]


for name in list(WINDOWS) + ["W30"]:
    for tag, d in (("702", df), ("pred", pred)):
        d[f"rx5day_{name}"] = sample(results[name]["rx5day"], d["X"].values, d["Y"].values)
        d[f"raincv_{name}"] = sample(results[name]["raincv"], d["X"].values, d["Y"].values)
    L.log(f"{name}: 702 缺失 rx5day={int(df[f'rx5day_{name}'].isna().sum())}, "
          f"raincv={int(df[f'raincv_{name}'].isna().sum())}", LOG)

df.to_csv(os.path.join(L.DATA, "modeling_base.csv"), index=False, encoding="utf-8-sig")
pred.to_csv(os.path.join(L.DATA, "prediction_base.csv"), index=False, encoding="utf-8-sig")

# 【v2 · C01】登记最终派生表哈希与父输入，区分"基础表"与"追加窗口列后的最终表"
import json as _json
_mp = os.path.join(L.CFG, "run_manifest.json")
if os.path.exists(_mp):
    with open(_mp, encoding="utf-8") as fh:
        _m = _json.load(fh)
    _prev = _m.get("output_files", {})
    _m["stage"] = "final（06_rain_window 追加降雨窗口列后的建模表）"
    _m["derived_tables"] = {
        "modeling_base.csv": {
            "sha256": L.file_hash(os.path.join(L.DATA, "modeling_base.csv"), length=64),
            "parent_stage": "base（00_prepare）",
            "parent_sha256": _prev.get("modeling_base.csv"),
            "produced_by": "00_prepare.py -> 06_rain_window.py（追加 rx5day/raincv 的 W30/W10/W01 列）",
            "produced_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "prediction_base.csv": {
            "sha256": L.file_hash(os.path.join(L.DATA, "prediction_base.csv"), length=64),
            "parent_stage": "base（00_prepare）",
            "parent_sha256": _prev.get("prediction_base.csv"),
            "produced_by": "00_prepare.py -> 06_rain_window.py",
            "produced_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    _m["rainfall_windows"] = {"W30": "1990-2019（复用最终因子集栅格）",
                              "W10": "2010-2019（本脚本从逐日数据重算）",
                              "W01": "2019 单年（诊断）"}
    with open(_mp, "w", encoding="utf-8") as fh:
        _json.dump(_m, fh, ensure_ascii=False, indent=2)
    L.log("run_manifest.json 已更新：登记最终派生表 64 位 SHA-256 与父输入", LOG)

# 与主方案（最终因子集 rx5day/raincv）的一致性校验：W30 应几乎相同
for k in ["rx5day", "raincv"]:
    d = np.nanmax(np.abs(df[k].values - df[f"{k}_W30"].values))
    r = np.corrcoef(df[k].values, df[f"{k}_W30"].values)[0, 1]
    L.log(f"[校验] {k}: 主方案 vs W30重算 最大绝对差={d:.4f}, r={r:.6f}", LOG)

# 相关矩阵
cols = [f"{k}_{w}" for w in WINDOWS for k in ["rx5day", "raincv"]]
cm = df[cols + ["rx5day", "raincv"]].corr().round(4)
cm.to_csv(os.path.join(L.RES, "06_降雨窗口相关矩阵.csv"), encoding="utf-8-sig")
L.log("\n相关系数矩阵:\n" + cm.to_string(), LOG)

rep = ["# 阶段四-B 降雨时间窗口敏感性（审稿人2-第7条）\n",
       "- 三个窗口：W30 = 1990-2019（主方案）、W10 = 2010-2019、W01 = 2019 单年（诊断）",
       "- 统一在 0.05° 源网格逐年计算 Rx5day 与日降雨 CV 后取多年平均，再最近邻对齐 432 m 网格",
       "", "## 指标间相关矩阵（702 样本）", "", cm.to_markdown(), ""]
with open(os.path.join(L.RES, "06_降雨窗口报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in rep))
L.log("=== 06 完成 ===", LOG)
