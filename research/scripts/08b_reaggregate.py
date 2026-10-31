# -*- coding: utf-8 -*-
"""
08b_reaggregate.py — 学习曲线重聚合与出图（v2）
=============================================
08_diagnose.py 输出的 `08_空间学习曲线.csv` 是逐(模型,折,比例,重复)的明细；
各折训练折大小不同（528/538/582/604/544），若直接按 n_train 分组会把同一比例
拆成多行。本脚本改为按 (model, frac) 聚合，n_train 用均值表示，并重绘曲线。
不重跑模型，只做聚合与绘图。
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

LOG = os.path.join(L.LOG, "08b_reaggregate.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 08b 学习曲线重聚合 ===", LOG)

LC = pd.read_csv(os.path.join(L.RES, "08_空间学习曲线.csv"))
L.log(f"明细 {LC.shape}；模型 {LC['model'].nunique()}；比例 {sorted(LC['frac'].unique())}", LOG)

AGG = LC.groupby(["model", "frac"]).agg(
    n_train_mean=("n_train", "mean"),
    n_train_min=("n_train", "min"),
    n_train_max=("n_train", "max"),
    n_obs=("test_auc", "count"),
    train_auc_mean=("train_auc", "mean"),
    train_auc_sd=("train_auc", "std"),
    test_auc_mean=("test_auc", "mean"),
    test_auc_sd=("test_auc", "std"),
    test_ap_mean=("test_ap", "mean"),
    gap_mean=("gap", "mean")).reset_index()
AGG.to_csv(os.path.join(L.RES, "08_学习曲线_按比例汇总.csv"), index=False,
           encoding="utf-8-sig")

# 各比例的折间+重复间波动（用于报告，不混入 t 分布 CI）
VAR = LC.groupby(["model", "frac"]).agg(
    test_auc_within_sd=("test_auc", "std"),
    test_auc_min=("test_auc", "min"),
    test_auc_max=("test_auc", "max")).reset_index()
VAR.to_csv(os.path.join(L.RES, "08_学习曲线_波动.csv"), index=False,
           encoding="utf-8-sig")


# ------------------------------------------------------------
# 【v3 · R08/C14】波动拆分：折间（空间差异）vs 折内（子集抽样）
# 每一 (model, frac) 把 test_auc 分解为
#   SS_total = SS_between_fold + SS_within_fold
# 折内部分再除以该比例的重复数即为"抽样噪声"的量级。
# ------------------------------------------------------------
DEC = []
for (m, f), g in LC.groupby(["model", "frac"]):
    gm = g.groupby("fold")["test_auc"]
    counts = gm.count()
    means = gm.mean()
    grand = g["test_auc"].mean()
    ss_between = float((counts * (means - grand) ** 2).sum())
    ss_within = float(((g["test_auc"] - g["fold"].map(means)) ** 2).sum())
    ss_total = ss_between + ss_within
    DEC.append({
        "model": m, "frac": f, "n_obs": int(len(g)),
        "n_fold": int(len(means)), "n_rep_per_fold": int(counts.max()),
        "grand_mean": grand,
        "ss_between_fold": ss_between, "ss_within_fold": ss_within,
        "share_between_fold": (ss_between / ss_total) if ss_total > 0 else np.nan,
        "sd_between_fold": float(means.std(ddof=1)) if len(means) > 1 else np.nan,
        "sd_within_fold_mean": float(
            g.groupby("fold")["test_auc"].std(ddof=1).mean()),
        "n_cells_mean": float(g["n_cells"].mean()) if "n_cells" in g else np.nan,
        "n_train_mean": float(g["n_train"].mean()),
        "train_ap_mean": float(g["train_ap"].mean()) if "train_ap" in g else np.nan,
        "test_ap_mean": float(g["test_ap"].mean()),
    })
DEC = pd.DataFrame(DEC)
DEC.to_csv(os.path.join(L.RES, "08_学习曲线_波动拆分.csv"), index=False,
           encoding="utf-8-sig")
L.log("\n波动拆分（折间占比）:\n" + DEC.pivot_table(
    index="frac", columns="model", values="share_between_fold"
).round(3).to_string())
L.log("\n按比例汇总（test AUC）:\n" +
      AGG.pivot_table(index="frac", columns="model",
                      values="test_auc_mean").round(4).to_string(), LOG)
L.log("\n按比例汇总（训练-测试 gap）:\n" +
      AGG.pivot_table(index="frac", columns="model",
                      values="gap_mean").round(4).to_string(), LOG)

models = [m for m in L.MODELS if m in AGG["model"].unique()]
ncol, nrow = 4, math.ceil(len(models) / 4)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), dpi=160)
axes = np.atleast_1d(axes).ravel()
for i, m in enumerate(models):
    ax = axes[i]
    d = AGG[AGG["model"] == m].sort_values("frac")
    ts = d["train_auc_sd"].fillna(0)
    vs = d["test_auc_sd"].fillna(0)
    ax.plot(d["n_train_mean"], d["train_auc_mean"], "o-", color="#c0392b", ms=3.5,
            label="Training AUC")
    ax.fill_between(d["n_train_mean"], d["train_auc_mean"] - ts,
                    d["train_auc_mean"] + ts, color="#c0392b", alpha=0.15)
    ax.plot(d["n_train_mean"], d["test_auc_mean"], "s-", color="#1e7b34", ms=4,
            label="Spatial test AUC")
    ax.fill_between(d["n_train_mean"], d["test_auc_mean"] - vs,
                    d["test_auc_mean"] + vs, color="#1e7b34", alpha=0.18)
    ax.set_title(m, fontsize=10)
    ax.set_xlabel("Training samples (mean over 5 folds)", fontsize=7.5)
    ax.set_ylabel("ROC-AUC", fontsize=8)
    ax.set_ylim(0.55, 1.03)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right")
for j in range(len(models), len(axes)):
    axes[j].axis("off")
fig.suptitle("Per-fold learning curves (each fold uses its own inner-search parameters;\n"
             "evaluated on the held-out spatial fold; band = SD across folds/repeats)",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_学习曲线.png"), dpi=160)
plt.close(fig)

rep = ["# 08b 学习曲线（按抽样比例聚合）\n",
       f"- 明细 {len(LC)} 行；{len(models)} 个模型；比例 {sorted(LC['frac'].unique())}",
       "- 各外层训练折大小不同（528/538/582/604/544），故按 `frac` 聚合，"
       "`n_train_mean` 为跨折均值；误差带为跨折+跨重复标准差。", "",
       "## 测试 AUC（空间外测）", "",
       AGG.pivot_table(index="frac", columns="model",
                       values="test_auc_mean").round(4).to_markdown(), "",
       "## 训练—测试 gap", "",
       AGG.pivot_table(index="frac", columns="model",
                       values="gap_mean").round(4).to_markdown(), "",
       "## 各比例波动（跨折+跨重复）", "", VAR.round(4).to_markdown(index=False), ""]
with open(os.path.join(L.RES, "08b_学习曲线报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in rep))
L.log("=== 08b 完成 ===", LOG)
