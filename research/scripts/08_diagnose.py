# -*- coding: utf-8 -*-
"""
08_diagnose.py — 过拟合诊断、学习曲线与校准（v2）
================================================
v2 相对 v1 的修正（依据核查清单）：
  C14  学习曲线改为**每个外层折独立**出曲线，超参数只取该折自己训练内选出的值，
       不再借用「5 折众数」；抽样按 cell 分组，每个子集独立拟合预处理；
       全量训练量只拟合 1 次（v1 重复 5 次实为同一结果）。
  C15  MLP 只保留**真实**的 training log loss；内部验证分数明确命名为
       validation_accuracy（sklearn 不暴露 val loss），不再用插值伪造验证曲线。
  C18  合并 OOF 指标改用逐折阈值。

输出： results/08_学习曲线.csv, 08_空间学习曲线.csv, 08_校准.csv,
       results/08_过拟合诊断.csv, 08_MLP_Loss.csv, 08_超参数选择频率.csv,
       figures/fig_学习曲线.png, fig_校准曲线.png
"""
import os
import sys
import json
import time
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

LOG = os.path.join(L.LOG, "08_diagnose.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 08 过拟合诊断 / 学习曲线 / 校准（v2）===", LOG)

df = L.load_base()
folds = df["fold"].values
factors = L.GROUPS["G2"]
y_all = df["landslide"].astype(int).values

# ---------- 读取每折各自的最优参数（C14：不借跨折众数） ----------
bp = pd.read_csv(os.path.join(L.RES, "03_最优超参数.csv"))
bp = bp[bp["group"] == "G2"]
PER_FOLD = {}
for m in bp["model"].unique():
    PER_FOLD[m] = {int(r["fold"]): json.loads(r["params"])
                   for _, r in bp[bp["model"] == m].iterrows()}
L.log(f"读取每折最优参数模型数: {len(PER_FOLD)}", LOG)

FREQ = []
for m, d in PER_FOLD.items():
    ps = [json.dumps(v, sort_keys=True) for v in d.values()]
    vc = pd.Series(ps).value_counts()
    for k, v in vc.items():
        FREQ.append({"model": m, "params": k, "被选中折数": int(v),
                     "占比": v / len(ps)})
pd.DataFrame(FREQ).to_csv(os.path.join(L.RES, "08_超参数选择频率.csv"),
                          index=False, encoding="utf-8-sig")

# ---------- 1. 过拟合诊断 ----------
main = pd.read_csv(os.path.join(L.RES, "03_汇总.csv"))
g2 = main[main["group"] == "G2"].copy()
cols = ["model", "roc_auc_mean", "roc_auc_std",
        "train_roc_auc", "gap_roc_auc", "f1_mean", "gap_f1", "brier_mean",
        "fold_mean_auc_minus_oof_auc"]
diag = g2[[c for c in cols if c in g2.columns]]
diag.to_csv(os.path.join(L.RES, "08_过拟合诊断.csv"), index=False,
            encoding="utf-8-sig")
L.log("\n过拟合诊断（G2）:\n" + diag.round(4).to_string(index=False), LOG)

# ---------- 2. 学习曲线 ----------
FRACS = [0.3, 0.5, 0.7, 1.0]
N_REP = 3
L.log(f"学习曲线：{len(PER_FOLD)} 模型 × 5 外层折 × {len(FRACS)} 训练量", LOG)


def stratified_group_subsample(dtr, ytr, frac, seed):
    """按 cell 分组的近似分层抽样：整 cell 抽取，不拆同像元记录。"""
    if frac >= 1.0:
        return np.arange(len(dtr))
    rng = np.random.default_rng(seed)
    cell = dtr["cell_id"].values
    uc = pd.unique(cell)
    cell_y = pd.Series(ytr, index=cell).groupby(level=0).max()
    pos_cells = cell_y[cell_y == 1].index.values
    neg_cells = cell_y[cell_y == 0].index.values
    n_target = max(40, int(round(frac * len(uc))))
    n_pos = max(1, int(round(n_target * len(pos_cells) / len(uc))))
    n_neg = max(1, n_target - n_pos)
    pick = np.concatenate([
        rng.choice(pos_cells, min(n_pos, len(pos_cells)), replace=False),
        rng.choice(neg_cells, min(n_neg, len(neg_cells)), replace=False)])
    return np.where(np.isin(cell, pick))[0]


def one_fit(m, k, frac, rep, dtr, dte, ytr, yte, params):
    seed = 1000 * k + 17 * rep + int(frac * 100)
    idx = stratified_group_subsample(dtr, ytr, frac, seed)
    d = dtr.iloc[idx]
    native = m in ("CatBoost", "LightGBM")
    if native:
        Xa, enc = L.matrix_for_categorical_model(d, factors, fit=True)
        Xb, _ = L.matrix_for_categorical_model(dte, factors, enc=enc, fit=False)
        ci = [list(Xa.columns).index(c) for c in enc.get("cat_cols", [])]
        fk = ({"cat_features": ci} if (m == "CatBoost" and ci)
              else ({"categorical_feature": enc.get("cat_cols", [])} if ci else {}))
    else:
        Xa, enc = L.build_matrix(d, factors, fit=True)
        Xb, _ = L.build_matrix(dte, factors, enc=enc, fit=False)
        fk = {}
    model = L.instantiate(m, params, 42, n_jobs=1)
    model.fit(Xa, ytr[idx], **fk)
    pa = model.predict_proba(Xa)[:, 1]
    pb = model.predict_proba(Xb)[:, 1]
    # R08/C14：补 n_cells / cell 集合哈希 / 训练集 AP，便于波动拆分与子集追溯
    import hashlib as _h
    sub_cells = sorted(map(str, d["cell_id"].values)) if "cell_id" in d.columns else []
    return {"model": m, "fold": k, "frac": frac, "rep": rep,
            "n_train": int(len(idx)),
            "n_cells": int(len(set(sub_cells))),
            "cell_ids_sha256": _h.sha256("|".join(sub_cells).encode()).hexdigest(),
            "train_auc": roc_auc_score(ytr[idx], pa),
            "train_ap": average_precision_score(ytr[idx], pa),
            "test_auc": roc_auc_score(yte, pb),
            "test_ap": average_precision_score(yte, pb),
            "gap": roc_auc_score(ytr[idx], pa) - roc_auc_score(yte, pb)}


rows = []
t0 = time.time()
for m in L.MODELS:
    if m not in PER_FOLD:
        continue
    for k in range(5):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]
        dtr = df.iloc[tr].reset_index(drop=True)
        dte = df.iloc[te].reset_index(drop=True)
        ytr, yte = y_all[tr], y_all[te]
        params = PER_FOLD[m][k]
        tasks = []
        for frac in FRACS:
            reps = 1 if frac >= 1.0 else N_REP     # C14：全量不重复
            for r in range(reps):
                tasks.append((m, k, frac, r, dtr, dte, ytr, yte, params))
        rows += Parallel(n_jobs=5)(delayed(one_fit)(*t) for t in tasks)
    L.log(f"  {m} 学习曲线完成（5 折，累计 {time.time()-t0:.0f}s）", LOG)

LC = pd.DataFrame(rows)
LC.to_csv(os.path.join(L.RES, "08_空间学习曲线.csv"), index=False,
          encoding="utf-8-sig")

# R08：08 本身不再按 n_train 碎片分组（v2 会与 08b 冲突），
# 直接按 (model, frac) 聚合，使 08 单独运行即得到正确版本；08b 只做补充拆分。
AGG = LC.groupby(["model", "frac"]).agg(
    n_train_mean=("n_train", "mean"), n_train_min=("n_train", "min"),
    n_train_max=("n_train", "max"), n_cells_mean=("n_cells", "mean"),
    n_fold_rep=("test_auc", "count"),
    train_auc_mean=("train_auc", "mean"), train_auc_std=("train_auc", "std"),
    train_ap_mean=("train_ap", "mean"),
    test_auc_mean=("test_auc", "mean"), test_auc_std=("test_auc", "std"),
    test_ap_mean=("test_ap", "mean"),
    gap_mean=("gap", "mean")).reset_index()
AGG.to_csv(os.path.join(L.RES, "08_学习曲线.csv"), index=False,
           encoding="utf-8-sig")
L.log("\n学习曲线（空间外测 AUC 均值）:\n" +
      AGG.pivot_table(index="frac", columns="model",
                      values="test_auc_mean").round(4).to_string() +
      "\n（行=训练量比例 frac；各 frac 的绝对样本数见 n_train_mean 列）", LOG)

# ---------- 3. MLP 真实 Loss ----------
from sklearn.neural_network import MLPClassifier
L.log("--- MLP Loss（真实 training log loss）---", LOG)
tr0 = df[folds != 0].reset_index(drop=True)
rng = np.random.default_rng(42)
tr0 = tr0.iloc[rng.permutation(len(tr0))].reset_index(drop=True)
y0 = tr0["landslide"].values
X0, _ = L.build_matrix(tr0, factors, fit=True)
p0 = PER_FOLD.get("MLP", {}).get(0, {})
mlp = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True,
                    n_iter_no_change=20, **p0)
mlp.fit(X0, y0)
loss_df = pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1),
                        "train_log_loss": mlp.loss_curve_})
va = getattr(mlp, "validation_scores_", None)
if va is not None and len(va):
    loss_df["validation_accuracy"] = np.nan
    n = min(len(va), len(loss_df))
    loss_df.loc[:n - 1, "validation_accuracy"] = np.asarray(va)[:n]
    loss_df["note"] = "validation_accuracy 为 sklearn 内部验证分数；未记录 val loss"
else:
    loss_df["validation_accuracy"] = np.nan
    loss_df["note"] = "无内部验证分数"
loss_df.to_csv(os.path.join(L.RES, "08_MLP_Loss.csv"), index=False,
               encoding="utf-8-sig")

# ---------- 4. 校准 ----------
L.log("--- 校准曲线（OOF）---", LOG)
oof = pd.read_csv(os.path.join(L.RES, "03_OOF预测.csv"))
oof = oof[oof["group"] == "G2"]
cal_rows = []
for m, g in oof.groupby("model"):
    yy = g["y"].values
    pp = g["oof"].values
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(pp, bins) - 1, 0, 9)
    recs = []
    for b in range(10):
        sel = idx == b
        if sel.sum() == 0:
            continue
        recs.append({"model": m, "bin": b,
                     "bin_center": (bins[b] + bins[b + 1]) / 2,
                     "n": int(sel.sum()),
                     "mean_predicted": float(pp[sel].mean()),
                     "observed_freq": float(yy[sel].mean())})
    # ECE 与 Brier 在全部箱算完后统一回填，避免 groupby.first() 只拿到第一个箱
    ece = float(sum(abs(r["mean_predicted"] - r["observed_freq"]) * r["n"] / len(yy)
                    for r in recs))
    brier = float(np.mean((pp - yy) ** 2))
    for r in recs:
        r["ECE"] = ece
        r["Brier"] = brier
        cal_rows.append(r)
CAL = pd.DataFrame(cal_rows)
CAL.to_csv(os.path.join(L.RES, "08_校准.csv"), index=False, encoding="utf-8-sig")
L.log("\n" + CAL.groupby("model")[["Brier", "ECE"]].first().round(4).to_string(), LOG)

# ---------- 图 ----------
models = [m for m in L.MODELS if m in AGG["model"].unique()]
ncol = 4
nrow = math.ceil(len(models) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.5 * nrow), dpi=160)
axes = np.atleast_1d(axes).ravel()
for i, m in enumerate(models):
    ax = axes[i]
    d = AGG[AGG["model"] == m]
    ts = d["train_auc_std"].fillna(0)
    vs = d["test_auc_std"].fillna(0)
    ax.plot(d["n_train_mean"], d["train_auc_mean"], "o-", color="#c0392b", ms=3,
            label="Training AUC")
    ax.fill_between(d["n_train_mean"], d["train_auc_mean"] - ts,
                    d["train_auc_mean"] + ts, color="#c0392b", alpha=0.15)
    ax.plot(d["n_train_mean"], d["test_auc_mean"], "s-", color="#1e7b34", ms=3.5,
            label="Spatial test AUC")
    ax.fill_between(d["n_train_mean"], d["test_auc_mean"] - vs,
                    d["test_auc_mean"] + vs, color="#1e7b34", alpha=0.18)
    ax.set_title(m, fontsize=10)
    ax.set_xlabel("Training samples", fontsize=8)
    ax.set_ylabel("ROC-AUC", fontsize=8)
    ax.set_ylim(0.5, 1.03)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right")
for j in range(len(models), len(axes)):
    axes[j].axis("off")
fig.suptitle("Per-fold learning curves (parameters from that fold's inner search; "
             "evaluated on the held-out spatial fold)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_学习曲线.png"), dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
ax = axes[0]
for m in models:
    d = CAL[CAL["model"] == m]
    ax.plot(d["mean_predicted"], d["observed_freq"], "o-", ms=3.5, label=m)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Calibration (OOF, G2)", fontsize=10)
ax.legend(fontsize=7); ax.tick_params(labelsize=8)
ax = axes[1]
ax.plot(loss_df["epoch"], loss_df["train_log_loss"], color="#8e44ad",
        label="Training log loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Log loss")
ax.set_title("MLP training loss (real, no interpolation)", fontsize=10)
ax.legend(fontsize=7); ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_校准曲线.png"), dpi=150)
plt.close(fig)

rep = ["# 08 过拟合诊断 / 学习曲线 / 校准（v2）\n",
       "## v2 修正", "",
       "- C14：学习曲线按**每个外层折**独立计算，参数只取该折训练内搜索结果；",
       "  抽样按 cell 分组；每个子集独立拟合预处理；全量训练量只拟合 1 次。",
       "- C15：MLP 仅报告真实 training log loss；内部验证分数命名为 "
       "`validation_accuracy`，未伪造 val loss。",
       "- C18：合并 OOF 指标改用逐折阈值。", "",
       "## 一、训练—空间测试差距", "", diag.round(4).to_markdown(index=False), "",
       "## 二、学习曲线（各折均值）", "",
       AGG.pivot_table(index="frac", columns="model",
                       values="test_auc_mean").round(4).to_markdown(), "",
       "## 三、校准", "",
       CAL.groupby("model")[["Brier", "ECE"]].first().round(4).to_markdown(), ""]
with open(os.path.join(L.RES, "08_诊断报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in rep))
L.log("=== 08 完成 ===", LOG)
