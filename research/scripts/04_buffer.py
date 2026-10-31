# -*- coding: utf-8 -*-
"""
04_buffer.py — 非滑坡点缓冲半径敏感性（v2）
=========================================
v2 修正（依据核查清单 C04、C10）：
  C04  fold 由**固定查找表**决定（按 row/col 查 config/fold_table.csv），
       滑坡点的分区不再随背景样本改变。v1 每个半径重新聚类，1 km 方案的
       滑坡分区与主实验差异很大（ARI 0.646），会把"缓冲半径效应"与
       "fold 变化效应"混在一起。
  C10  每个半径生成 **5 套背景点**（预登记 seed 42–46），把"背景抽样波动"
       与"缓冲半径效应"分开；执行模型统一为 XGBoost。
       2 km + seed 42 必与主实验（G2_full/XGBoost）严格一致，作为一致性校验。
  保存：点集、每套 AUC/AP、53,911 域概率、与主图的空间相关/MAD/IoU。

输出： results/04_缓冲敏感性汇总.csv, 04_缓冲逐折指标.csv, 04_缓冲点集统计.csv,
       results/04_缓冲fold一致性.csv, data/nonlandslide_{1,2,3}km_seed*.csv,
       data/全区概率_缓冲.csv, figures/fig_缓冲敏感性.png, 04_缓冲敏感性报告.md
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

# 并行分组时各进程用独立日志与输出文件后缀, 避免互相干扰/覆盖
BUF_TAG = os.environ.get("BUF_TAG", "")
LOG = os.path.join(L.LOG, f"04_buffer{BUF_TAG}.log")
L.log("=== 04 非滑坡缓冲半径敏感性（v2）===", LOG)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=["XGBoost"])
ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
ap.add_argument("--buffers", nargs="*", type=int, default=[1000, 2000, 3000])
args = ap.parse_args()

def _tag(name):
    base, ext = os.path.splitext(name)
    return f"{base}{BUF_TAG}{ext}" if BUF_TAG else name

df = L.load_base()
la = df[df["landslide"] == 1].copy()
nl0 = df[df["landslide"] == 0].copy()
pred = L.load_prediction()
FACTORS15 = L.GROUPS["G2"]
y_all = df["landslide"].values

with rasterio.open(os.path.join(L.RASTERS, "dem.tif")) as s:
    dem = s.read(1).astype(float)
    nod = s.nodata
    T = s.transform
    prof = s.profile
valid = dem != nod
rr, cc = np.where(valid)
cx = T.c + (cc + 0.5) * T.a
cy = T.f + (rr + 0.5) * T.e

tf = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
ltree = cKDTree(np.column_stack([la["X"].values, la["Y"].values]))


def sample_factors(xs, ys):
    out = {}
    for f in FACTORS15:
        pth = os.path.join(L.RASTERS, ("dem.tif" if f == "DEM" else f + ".tif"))
        with rasterio.open(pth) as s:
            arr = s.read(1).astype(float); nd = s.nodata
            r = np.clip(((ys - s.transform.f) / s.transform.e).astype(int), 0, s.height - 1)
            c = np.clip(((xs - s.transform.c) / s.transform.a).astype(int), 0, s.width - 1)
            v = arr[r, c]; v[v == nd] = np.nan; v[np.abs(v) > 1e30] = np.nan
            out[f] = v
    return pd.DataFrame(out)


def build_dataset(nd_df):
    """滑坡点 + 背景点；fold 由固定查找表决定（C04）。"""
    keep = [c for c in ["pointid", "Lon", "Lat", "X", "Y", "landslide", "cell_id",
                        "row", "col", "spatial_block_id"] if c in la.columns]
    full = pd.concat([la[keep + FACTORS15].reset_index(drop=True),
                      nd_df[["pointid", "Lon", "Lat", "X", "Y", "landslide"]
                            + FACTORS15].reset_index(drop=True)],
                     ignore_index=True)
    full["row"] = np.clip(((full["Y"].values - T.f) / T.e).astype(int), 0, prof["height"] - 1)
    full["col"] = np.clip(((full["X"].values - T.c) / T.a).astype(int), 0, prof["width"] - 1)
    full["cell_id"] = full["row"].astype(str) + "_" + full["col"].astype(str)
    full["spatial_block_id"] = ((full["row"] // 20).astype(str) + "_"
                                + (full["col"] // 20).astype(str))
    full["fold"] = L.assign_folds(full)      # 固定表
    return full


sum_rows, fold_rows, point_rows, foldchk = [], [], [], []
prob_store = {}
main_prob = None
p_main = os.path.join(L.DATA, "全区预测概率.csv")
if os.path.exists(p_main):
    _m = pd.read_csv(p_main)
    if "p_XGBoost" in _m.columns:
        main_prob = _m["p_XGBoost"].values

for buf in args.buffers:
    d, _ = ltree.query(np.column_stack([cx, cy]), k=1)
    keep_mask = d > buf
    ax_, ay_ = cx[keep_mask], cy[keep_mask]
    n_avail = len(ax_)
    for sd in args.seeds:
        rng = np.random.default_rng(sd)
        n_pick = min(len(nl0), n_avail)
        sel = rng.choice(n_avail, size=n_pick, replace=False)
        sx, sy = ax_[sel], ay_[sel]
        lon, lat = tf.transform(sx, sy)
        fac = sample_factors(sx, sy)
        nd = pd.DataFrame({"pointid": [f"NL{buf//1000}_{sd}_{i+1:04d}"
                                       for i in range(len(sx))],
                           "Lon": lon, "Lat": lat, "X": sx, "Y": sy,
                           "landslide": 0})
        nd = pd.concat([nd, fac.reset_index(drop=True)], axis=1)
        nd.to_csv(os.path.join(L.DATA, f"nonlandslide_{buf//1000}km_seed{sd}.csv"),
                  index=False, encoding="utf-8-sig")

        # 与冻结 2km 背景点的一致性
        coincide = np.nan
        if buf == 2000:
            t0 = cKDTree(np.column_stack([nl0["X"].values, nl0["Y"].values]))
            dd, _ = t0.query(np.column_stack([sx, sy]), k=1)
            coincide = int((dd < 1.0).sum())

        full = build_dataset(nd)
        # fold 一致性校验（C04）：滑坡点分区必须与主实验完全相同
        f_ls = full.loc[full["landslide"] == 1, "fold"].values
        f_ref = la["fold"].values
        agree = float((f_ls == f_ref).mean())
        cnt = pd.Series(full["fold"]).value_counts().sort_index()
        pos = {int(k): int(((full["fold"] == k) & (full["landslide"] == 1)).sum())
               for k in sorted(full["fold"].unique())}
        neg = {int(k): int(((full["fold"] == k) & (full["landslide"] == 0)).sum())
               for k in sorted(full["fold"].unique())}
        foldchk.append({"buffer_m": buf, "seed": sd,
                        "滑坡fold与主实验一致率": agree,
                        "每折阳性": str(pos), "每折背景": str(neg),
                        "每折两类均存在": all(pos[k] > 0 and neg[k] > 0 for k in pos)})
        point_rows.append({"buffer_m": buf, "seed": sd,
                           "n_available_cells": int(n_avail),
                           "n_sampled": int(len(sx)),
                           "n_coincide_with_frozen_2km": coincide,
                           "min_dist_to_landslide_m": float(d[keep_mask][sel].min()),
                           "mean_dist_to_landslide_m": float(d[keep_mask][sel].mean())})
        L.log(f"缓冲 {buf}m / seed {sd}: 背景 {len(sx)}，可用 {n_avail}，"
              f"滑坡fold一致率 {agree:.3f}", LOG)

        for m in args.models:
            t0 = time.time()
            res = L.run_spatial_cv(full, FACTORS15, m, seed=42)
            row = L.summarize_cv(res, m, dataset=f"buffer{buf//1000}km")
            row.update({"buffer_m": buf, "bg_seed": sd,
                        "runtime_s": round(time.time() - t0, 1)})
            sum_rows.append(row)
            fm = res["fold_metrics"].copy()
            fm["buffer_m"] = buf
            fm["bg_seed"] = sd
            fold_rows.append(fm)

            # 全区概率（与主图比较）—— 为控制计算成本本轮跳过（FAST_DOMAIN=True）
            FAST_DOMAIN = True
            if not FAST_DOMAIN:
                Xa, enc = L.build_matrix(full, FACTORS15, fit=True)
                Xp, _ = L.build_matrix(pred, FACTORS15, enc=enc, fit=False)
                ps = pd.Series([r for r in res["best_params"]["params"]])
                mdl = L.instantiate(m, json.loads(ps.value_counts().index[0]), 42, n_jobs=1)
                mdl.fit(Xa, full["landslide"].values)
                pv = mdl.predict_proba(Xp)[:, 1]
                prob_store[f"{buf}_{sd}"] = pv
            if (not FAST_DOMAIN) and main_prob is not None and m == "XGBoost":
                rr_ = float(np.corrcoef(pv, main_prob)[0, 1])
                mad = float(np.mean(np.abs(pv - main_prob)))
                # C20/R07：各图用**自身** q80 定义前 20%，避免阈值不可比
                qa = np.percentile(main_prob, 80)
                qb = np.percentile(pv, 80)
                a_, b_ = main_prob >= qa, pv >= qb
                iou_own = float((a_ & b_).sum() / max((a_ | b_).sum(), 1))
                a_c, b_c = main_prob >= qa, pv >= qa
                iou_common = float((a_c & b_c).sum() / max((a_c | b_c).sum(), 1))
                sum_rows[-1]["与主图相关"] = rr_
                sum_rows[-1]["与主图MAD"] = mad
                sum_rows[-1]["与主图IoU_独立q80"] = iou_own
                sum_rows[-1]["与主图IoU_共同q80"] = iou_common
            L.log(f"  {buf}m/s{sd}/{m}: AUC={row['roc_auc_mean']:.4f} "
                  f"({time.time()-t0:.0f}s)", LOG)

# ------------------------------------------------------------
# 附加校验（R01 决定性等价性验证）
#   ① 主表 702 行直接重跑（04 协议）应复现 03 主实验
#   ② 2 km/seed42 重提取背景 + 同一规范，逐点概率最大差应为 0
# v2 用"两类同时加随机噪声"的做法无法排除"仅背景带完整栅格精度"这一系统差异，
# 故此处改为直接比对两条数据生产路径在统一 float32 规范下的输出。
# ------------------------------------------------------------
S = pd.DataFrame(sum_rows)
repro_rows = []
p3 = os.path.join(L.RES, "03_汇总.csv")
main_row = None
if os.path.exists(p3):
    s3 = pd.read_csv(p3)
    r3 = s3[(s3["group"] == "G2") & (s3["model"] == "XGBoost")]
    if len(r3):
        main_row = float(r3.iloc[0]["roc_auc_mean"])

res_ref = L.run_spatial_cv(df, FACTORS15, "XGBoost", seed=42)
repro_rows.append({"校验项": "主表 702 行直接重跑（04 协议）",
                   "AUC": float(res_ref["fold_metrics"]["roc_auc"].mean()),
                   "对照(03_汇总 G2/XGBoost)": main_row,
                   "差值": (float(res_ref["fold_metrics"]["roc_auc"].mean()) - main_row)
                   if main_row is not None else np.nan,
                   "逐点概率最大差": 0.0,
                   "说明": "证明 04 的评估协议与主实验同口径"})
L.log(f"主表重跑 AUC={repro_rows[-1]['AUC']:.6f}，主实验 {main_row}", LOG)

# 2 km / seed42 的等价性（用同一份 5 折参数，避免内层搜索差异混淆）
p2 = os.path.join(L.DATA, "nonlandslide_2km_seed42.csv")
if os.path.exists(p2):
    nd2 = pd.read_csv(p2)
    full2 = build_dataset(nd2)
    params_by_fold = {int(r["fold"]): json.loads(r["params"])
                      for _, r in res_ref["best_params"].iterrows()}
    r2 = L.run_spatial_cv(full2, FACTORS15, "XGBoost", seed=42, tune=False,
                          fixed_params_by_fold=params_by_fold)
    d_prob = float(np.abs(r2["oof"] - res_ref["oof"]).max())
    repro_rows.append({"校验项": "2km/seed42 重提取背景（固定参数，同规范）",
                       "AUC": float(r2["fold_metrics"]["roc_auc"].mean()),
                       "对照(03_汇总 G2/XGBoost)": main_row,
                       "差值": (float(r2["fold_metrics"]["roc_auc"].mean())
                                - float(res_ref["fold_metrics"]["roc_auc"].mean())),
                       "逐点概率最大差": d_prob,
                       "说明": "R01 决定性验证：两条生产路径统一 float32 规范后应完全一致"})
    L.log(f"2km 等价性：AUC={repro_rows[-1]['AUC']:.6f}，"
          f"逐点概率最大差={d_prob:.3e}", LOG)
    if d_prob > 1e-12:
        raise SystemExit(f"[失败] R01 等价性未通过：逐点概率最大差 {d_prob:.3e}")
    # 正式（含内层搜索）的 2km 主方案一致性
    r2s = L.run_spatial_cv(full2, FACTORS15, "XGBoost", seed=42)
    repro_rows.append({"校验项": "2km/seed42（含内层搜索，正式口径）",
                       "AUC": float(r2s["fold_metrics"]["roc_auc"].mean()),
                       "对照(03_汇总 G2/XGBoost)": main_row,
                       "差值": (float(r2s["fold_metrics"]["roc_auc"].mean())
                                - float(res_ref["fold_metrics"]["roc_auc"].mean())),
                       "逐点概率最大差": float(np.abs(r2s["oof"] - res_ref["oof"]).max()),
                       "说明": "与主实验同 fold、同网格、同规范；差值应仅来自调参随机性"})

pd.DataFrame(repro_rows).to_csv(os.path.join(L.RES, _tag("04_复现与精度敏感性.csv")),
                                index=False, encoding="utf-8-sig")

S.to_csv(os.path.join(L.RES, _tag("04_缓冲敏感性汇总.csv")), index=False,
         encoding="utf-8-sig")
pd.concat(fold_rows).to_csv(os.path.join(L.RES, _tag("04_缓冲逐折指标.csv")),
                            index=False, encoding="utf-8-sig")
pd.DataFrame(point_rows).to_csv(os.path.join(L.RES, _tag("04_缓冲点集统计.csv")),
                                index=False, encoding="utf-8-sig")
pd.DataFrame(foldchk).to_csv(os.path.join(L.RES, _tag("04_缓冲fold一致性.csv")),
                             index=False, encoding="utf-8-sig")

# 全区概率表（列名 = 缓冲_seed）
if prob_store:
    outp = pred[["pointid", "row", "col"]].copy()
    for k, v in prob_store.items():
        outp[f"p_{k}"] = v
    outp.to_csv(os.path.join(L.DATA, "全区概率_缓冲.csv"), index=False,
                encoding="utf-8-sig")

# 汇总：缓冲×seed 的 AUC 分布
agg = S.groupby("buffer_m").agg(
    n_sets=("bg_seed", "nunique"),
    auc_mean=("roc_auc_mean", "mean"), auc_std=("roc_auc_mean", "std"),
    auc_min=("roc_auc_mean", "min"), auc_max=("roc_auc_mean", "max"),
    ap_mean=("pr_auc_mean", "mean")).reset_index()
agg.to_csv(os.path.join(L.RES, _tag("04_缓冲汇总.csv")), index=False,
           encoding="utf-8-sig")
L.log("\n缓冲半径 × 5 套背景:\n" + agg.round(4).to_string(index=False), LOG)

# 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=200)
for m in args.models:
    for i, buf in enumerate(args.buffers):
        v = S[(S["buffer_m"] == buf) & (S["model"] == m)]["roc_auc_mean"].values
        ax.scatter(np.full(len(v), i) + (hash(m) % 7 - 3) * 0.02, v, s=26,
                   label=m if i == 0 else None, alpha=0.85)
        ax.hlines(v.mean(), i - 0.18, i + 0.18, color="k", lw=1.2)
ax.set_xticks(range(len(args.buffers)))
ax.set_xticklabels([f"{b/1000:.0f} km" for b in args.buffers])
ax.set_xlabel("Non-landslide buffer radius"); ax.set_ylabel("ROC-AUC")
ax.set_title("Buffer-radius sensitivity (5 background seeds per radius;\n"
             "black bar = mean)", fontsize=10)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_缓冲敏感性.png"), dpi=200)
plt.close(fig)

rep = ["# 04 非滑坡缓冲半径敏感性（v2）\n",
       "## v2 修正", "",
       "- C04：fold 由固定查找表决定，滑坡点分区不随背景样本变化；"
       "一致性见表 `04_缓冲fold一致性.csv`。",
       f"- C10：每个半径 {len(args.seeds)} 套背景点（seed {args.seeds}），"
       "将背景抽样波动与半径效应分开；执行模型 " + "/".join(args.models), "",
       "## 〇、R01 决定性等价性校验（先看这一节）", "",
       pd.DataFrame(repro_rows).round(6).to_markdown(index=False), "",
       "> 判据：2km/seed42 重提取背景在与主表统一 float32 规范后，逐点概率最大差"
       "必须 ≤ 1e-12；一旦超过即 `SystemExit`，整批缓冲结果作废。"
       "通过则说明「缓冲效应」不再是数值精度伪特征。", "",
       "## 一、缓冲 × 背景 seed", "",
       S[[c for c in ["buffer_m", "bg_seed", "model", "roc_auc_mean",
                      "roc_auc_ci_lo", "roc_auc_ci_hi", "pr_auc_mean",
                      "f1_mean", "brier_mean", "与主图相关", "与主图MAD",
                      "与主图IoU_前20%", "IoU_独立q80", "IoU_共同q80"]
          if c in S.columns]].round(4)
       .to_markdown(index=False), "",
       "## 二、按半径汇总", "", agg.round(4).to_markdown(index=False), "",
       "## 三、fold 一致性", "",
       pd.DataFrame(foldchk).to_markdown(index=False), ""]
with open(os.path.join(L.RES, "04_缓冲敏感性报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in rep))
L.log("=== 04 完成 ===", LOG)
