# -*- coding: utf-8 -*-
"""
07_seed.py — 随机种子重复性实验（v3 · R04）
==========================================
v2 审查 R04 的三处修正：

设计A（固定配置重复）
  v2 调用 run_spatial_cv(tune=False) **没有传入逐折参数**，公共函数回退到默认
  参数，A 的 OOF AUC 0.9177 与主实验 0.9165 不同（逐点最大差 0.574）。
  v3 逐折传入主实验的 PER_FOLD[k] 与 THR[k]，并**先断言 rep0 逐点复现主实验
  OOF**，再做 30 次重复。

设计B（纯模型 seed）
  外层/内层切分、每折参数、阈值全部固定，**只改模型 random_state**。

设计B-联合（可选，单独命名）
  连内层切分也随 seed 变并重新调参，属"联合随机性"，不与纯 B 混表。

区域图
  v2 的 predict_domain 内部固定 seed=42，30 张图只有 12 个唯一向量。
  v3 真正把 model_seed 传进去。

输出：07_seed设计A/B/Bjoint.csv + 各自 _OOF.csv、07_设计A一致性.json、
      07_域概率稳定性.json、07_seed汇总.csv、data/seed概率矩阵.npy、报告
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

LOG = os.path.join(L.LOG, "07_seed.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 07 随机种子实验（v3 · R04）===", LOG)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="XGBoost")
ap.add_argument("--n_repeat", type=int, default=30)
ap.add_argument("--skip_joint", action="store_true")
args = ap.parse_args()
M = args.model
N = args.n_repeat

df = L.load_base()
folds = df["fold"].values
factors = L.GROUPS["G2"]
y = df["landslide"].astype(int).values
pred = L.load_prediction()

bp = pd.read_csv(os.path.join(L.RES, "03_最优超参数.csv"))
bp = bp[(bp["group"] == "G2") & (bp["model"] == M)]
PER_FOLD = {int(r["fold"]): json.loads(r["params"]) for _, r in bp.iterrows()}
THR = {int(r["fold"]): float(r["threshold"]) for _, r in bp.iterrows()}
L.log(f"每折参数（来自主实验 03）：{PER_FOLD}", LOG)
L.log(f"每折阈值：{THR}", LOG)


def run_rep(tag, model_seed=None, tune=False, flow_seed=42):
    return L.run_spatial_cv(
        df, factors, M, seed=flow_seed, folds=folds, tune=tune,
        fixed_params_by_fold=(None if tune else PER_FOLD),
        fixed_params_thr=(None if tune else THR),
        model_seed=model_seed)


def pack_oof(res, col, val):
    out = []
    for k in range(5):
        m = folds == k
        out.append(pd.DataFrame({
            col: val, "fold": k,
            "pointid": df.loc[m, "pointid"].values,
            "cell_id": df.loc[m, "cell_id"].values,
            "y": y[m], "p": res["oof"][m],
            "threshold": res["thr_by_fold"][k],
            "yhat": (res["oof"][m] >= res["thr_by_fold"][k]).astype(int)}))
    return pd.concat(out, ignore_index=True)


def auc_row(res, extra):
    d = {"oof_auc": res["pooled"]["roc_auc"],
         "oof_ap": res["pooled"]["pr_auc"],
         "oof_brier": res["pooled"]["brier"],
         "fold_mean_auc": res["fold_metrics"]["roc_auc"].mean(),
         "oof_f1_foldthr": res["pooled_fold_thr"]["f1"]}
    d.update(extra)
    return d


# ================= 设计A：固定配置重复 =================
L.log(f"--- 设计A：固定配置重复 {N} 次（逐折参数+阈值）---", LOG)
rowsA, oofA = [], []
t0 = time.time()
for rep in range(N):
    res = run_rep("A")
    oofA.append(pack_oof(res, "rep", rep))
    rowsA.append(auc_row(res, {"repeat": rep}))
A = pd.DataFrame(rowsA)
A.to_csv(os.path.join(L.RES, "07_seed设计A.csv"), index=False, encoding="utf-8-sig")
OOFA = pd.concat(oofA, ignore_index=True)
OOFA.to_csv(os.path.join(L.RES, "07_seed设计A_OOF.csv"), index=False,
            encoding="utf-8-sig")

# —— R04 关键验收：rep0 是否逐点复现主实验 OOF ——
main_oof = pd.read_csv(os.path.join(L.RES, "03_OOF预测.csv"))
main_oof = main_oof[(main_oof["group"] == "G2") & (main_oof["model"] == M)]
rep0 = OOFA[OOFA["rep"] == 0].sort_values("pointid").reset_index(drop=True)
main_oof = main_oof.sort_values("pointid").reset_index(drop=True)
if len(main_oof) and (main_oof["pointid"].values == rep0["pointid"].values).all():
    d_main = float(np.abs(main_oof["oof"].values - rep0["p"].values).max())
else:
    d_main = float("nan")
L.log(f"设计A rep0 与主实验 OOF 逐点最大差 = {d_main:.3e}", LOG)

pivA = OOFA.pivot_table(index="pointid", columns="rep", values="p")
maxdiff = float(np.nanmax(np.abs(pivA.values - pivA.values[:, [0]])))
consistency = {
    "n_repeat": N,
    "n_unique_oof_auc": int(A["oof_auc"].nunique()),
    "oof_auc_std": float(A["oof_auc"].std(ddof=0)),
    "max_abs_diff_vs_rep0": maxdiff,
    "tolerance": 1e-12,
    "pass_repeat_identity": bool(maxdiff <= 1e-12),
    "rep0_vs_main_oof_max_abs_diff": d_main,
    "pass_rep0_matches_main": bool(np.isfinite(d_main) and d_main <= 1e-9),
    "note": "A 逐折传入主实验 PER_FOLD/THR；rep0 必须复现主实验 OOF，"
            "只满足重复一致而参数不同是不够的（R04）",
}
L.save_json(consistency, os.path.join(L.RES, "07_设计A一致性.json"))
L.log(f"设计A 一致性：{consistency}", LOG)
if not consistency["pass_rep0_matches_main"]:
    raise SystemExit("设计A rep0 未复现主实验 OOF，参数传递仍有问题（R04 未通过）")

# ================= 设计B：纯模型 seed =================
L.log(f"--- 设计B（纯模型 seed）：切分/参数/阈值全固定，仅改 random_state ---", LOG)
rowsB, oofB, probsB, paramsB = [], [], [], []
for sd in range(N):
    t1 = time.time()
    res = run_rep("B", model_seed=sd)
    oofB.append(pack_oof(res, "seed", sd))
    rowsB.append(auc_row(res, {"seed": sd,
                               "runtime_s": round(time.time() - t1, 1)}))
    paramsB.append({"seed": sd, "branch": "B_pure_model_seed",
                    "params_main": json.dumps(PER_FOLD[1], ensure_ascii=False),
                    "note": "参数固定为主实验逐折参数"})
    L.log(f"  seed {sd}: OOF AUC={res['pooled']['roc_auc']:.6f}", LOG)
B = pd.DataFrame(rowsB)
B.to_csv(os.path.join(L.RES, "07_seed设计B.csv"), index=False, encoding="utf-8-sig")
pd.concat(oofB, ignore_index=True).to_csv(
    os.path.join(L.RES, "07_seed设计B_OOF.csv"), index=False, encoding="utf-8-sig")

# ================= 设计B-联合（可选）=================
if not args.skip_joint:
    L.log(f"--- 设计B-联合：内层切分也随 seed 变并重新调参 ---", LOG)
    rowsJ, oofJ = [], []
    for sd in range(N):
        res = run_rep("BJ", model_seed=sd, tune=True, flow_seed=sd)
        oofJ.append(pack_oof(res, "seed", sd))
        ps = pd.Series(res["best_params"]["params"])
        rowsJ.append(auc_row(res, {"seed": sd,
                                   "params_main": ps.value_counts().index[0],
                                   "n_distinct_params": int(ps.nunique())}))
        L.log(f"  seed {sd}: OOF AUC={res['pooled']['roc_auc']:.6f}", LOG)
    BJ = pd.DataFrame(rowsJ)
    BJ.to_csv(os.path.join(L.RES, "07_seed设计B联合.csv"), index=False,
              encoding="utf-8-sig")
    pd.concat(oofJ, ignore_index=True).to_csv(
        os.path.join(L.RES, "07_seed设计B联合_OOF.csv"), index=False,
        encoding="utf-8-sig")

# ================= 区域图（真正传入 model_seed）=================
L.log("--- 区域 53,911 域概率：纯模型 seed，最终参数固定 ---", LOG)
Xa, enc = L.build_matrix(df, factors, fit=True)
Xp, _ = L.build_matrix(pred, factors, enc=enc, fit=False)
ps_main = pd.Series([json.dumps(PER_FOLD[k], sort_keys=True) for k in PER_FOLD])
final_params = json.loads(ps_main.value_counts().index[0])
L.log(f"区域图最终参数（5 折众数）：{final_params}", LOG)
for sd in range(N):
    mdl = L.instantiate(M, final_params, sd, n_jobs=1)
    if hasattr(mdl, "random_state"):
        mdl.set_params(random_state=sd)
    mdl.fit(Xa, y)
    probsB.append(mdl.predict_proba(Xp)[:, 1])
PB = np.vstack(probsB)
np.save(os.path.join(L.DATA, "seed概率矩阵.npy"), PB)
L.log(f"域概率矩阵 {PB.shape}，唯一向量数 {len(np.unique(PB.round(12), axis=0))}", LOG)

pd.DataFrame(paramsB).to_csv(os.path.join(L.RES, "07_seed参数配置.csv"),
                             index=False, encoding="utf-8-sig")

# ================= 汇总 =================
summ = []
for key, lab in [("oof_auc", "OOF ROC-AUC"), ("oof_ap", "OOF PR-AUC"),
                 ("fold_mean_auc", "折均值 AUC"), ("oof_brier", "OOF Brier")]:
    summ.append({"指标": lab,
                 "设计A_均值": A[key].mean(), "设计A_标准差": A[key].std(ddof=0),
                 "设计A_唯一值数": int(A[key].nunique()),
                 "设计B_均值": B[key].mean(), "设计B_标准差": B[key].std(ddof=1),
                 "设计B_最小": B[key].min(), "设计B_最大": B[key].max(),
                 "设计B_极差": B[key].max() - B[key].min(),
                 "设计B_变异系数%": 100 * B[key].std(ddof=1) / B[key].mean()})
S = pd.DataFrame(summ)
S.to_csv(os.path.join(L.RES, "07_seed汇总.csv"), index=False, encoding="utf-8-sig")
L.log("\n" + S.round(6).to_string(index=False), LOG)

# 域概率稳定性（各 seed 用**自身**的 q80 判定前 20%，避免阈值不可比）
sd_map = PB.std(axis=0, ddof=1)
main_p = None
pth = os.path.join(L.DATA, "全区预测概率.csv")
if os.path.exists(pth):
    _m = pd.read_csv(pth)
    if "p_XGBoost" in _m.columns:
        main_p = _m["p_XGBoost"].values
stab = {"n_seed": int(PB.shape[0]), "domain_cells": int(PB.shape[1]),
        "n_unique_probability_vectors": int(len(np.unique(PB.round(12), axis=0))),
        "prob_sd_mean": float(sd_map.mean()),
        "prob_sd_p95": float(np.percentile(sd_map, 95)),
        "prob_sd_max": float(sd_map.max()),
        "model_seed_actually_passed": True,
        "final_params": final_params}
if main_p is not None:
    qm = np.percentile(main_p, 80)
    a_ = main_p >= qm
    ious, corrs, mads = [], [], []
    for p in PB:
        qp = np.percentile(p, 80)
        b_ = p >= qp
        ious.append(float((a_ & b_).sum() / max((a_ | b_).sum(), 1)))
        corrs.append(float(np.corrcoef(p, main_p)[0, 1]))
        mads.append(float(np.mean(np.abs(p - main_p))))
    stab.update({"common_q80": float(qm),
                 "IoU_own_q80_mean": float(np.mean(ious)),
                 "IoU_own_q80_min": float(np.min(ious)),
                 "corr_vs_main_mean": float(np.mean(corrs)),
                 "MAD_vs_main_mean": float(np.mean(mads))})
L.save_json(stab, os.path.join(L.RES, "07_域概率稳定性.json"))
L.log(f"域概率稳定性：{stab}", LOG)

# ================= 图 =================
import rasterio
with rasterio.open(os.path.join(L.RASTERS, "dem.tif")) as s:
    dem = s.read(1).astype(float); dem[dem == s.nodata] = np.nan
g = np.full(dem.shape, np.nan)
g[pred["row"].values, pred["col"].values] = sd_map
fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=200)
im = ax.imshow(g, cmap="magma_r", interpolation="nearest", vmin=0,
               vmax=np.nanpercentile(sd_map, 99))
plt.colorbar(im, ax=ax, shrink=0.85, label="SD of probability across model seeds")
ax.set_title(f"{M} — probability stability (53,911 cells, "
             f"{stab['n_unique_probability_vectors']} unique vectors)", fontsize=10)
ax.set_xlabel("Column"); ax.set_ylabel("Row")
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_seed概率稳定性.png"), dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=150)
axes[0].plot(A["repeat"], A["oof_auc"], "o-", ms=3, color="#c0392b")
axes[0].set_title("Design A: fixed config, repeats", fontsize=10)
axes[0].set_xlabel("repeat"); axes[0].set_ylabel("OOF ROC-AUC")
axes[0].tick_params(labelsize=8)
axes[1].plot(B["seed"], B["oof_auc"], "o-", ms=3, color="#2f7fbf")
axes[1].set_title("Design B: model seed only (splits/params fixed)", fontsize=10)
axes[1].set_xlabel("seed"); axes[1].set_ylabel("OOF ROC-AUC")
axes[1].tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_seed轨迹.png"), dpi=150)
plt.close(fig)

rep = ["# 07 随机种子重复性实验（v3 · R04）\n", "## v3 修正", "",
       "- **设计A**：逐折传入主实验 `PER_FOLD[k]` 与 `THR[k]`（v2 传了 `tune=False` 却"
       "没传参数，回退到默认 XGB，A 的 AUC 与主实验差 0.574 点概率）；"
       f"rep0 与主实验 OOF 逐点最大差 **{d_main:.3e}**。",
       "- **设计B**：外层/内层切分、每折参数、阈值全固定，**只改模型 `random_state`**。",
       "- **设计B-联合**（若运行）：连内层切分一起变并重新调参，单独命名、不与纯 B 混表。",
       "- **区域图**：`predict_domain` 真正接收 seed（v2 内部写死 42，30 张图只有 12 个"
       "唯一向量）。", "",
       "## 一、设计A 一致性判定", "",
       f"```json\n{json.dumps(consistency, ensure_ascii=False, indent=2)}\n```", "",
       "## 二、设计A/B 汇总", "", S.round(6).to_markdown(index=False), "",
       "## 三、域概率稳定性（纯模型 seed）", "",
       f"```json\n{json.dumps(stab, ensure_ascii=False, indent=2)}\n```", "",
       "## 使用边界", "",
       "- B 只反映**模型随机源**引起的波动，不含调参与切分随机性；",
       "- 30 个 seed 不是 30 次独立空间重复，不能当作空间重复抽样使用。", ""]
with open(os.path.join(L.RES, "07_seed报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in rep))
L.log("=== 07 完成 ===", LOG)
