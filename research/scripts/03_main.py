# -*- coding: utf-8 -*-
"""
03_main.py — 阶段三：主模型与因子组消融（审稿人1-1 XGBoost/RF矛盾、审稿人1-7过拟合、
                                       审稿人2-7降雨口径）
====================================================================================
6 类模型 × 4 个因子组（G0/G1/G2/G3）× 5 折空间分块CV，训练折内完成
标准化/编码/阈值/超参数搜索。

用法：
  python 03_main.py                      # 全部
  python 03_main.py --groups G2 --models RF   # 子集
输出： results/03_逐折指标.csv, results/03_汇总.csv, results/03_训练集指标.csv,
       results/03_最优超参数.csv, results/03_OOF预测.csv, results/03_主实验报告.md
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=["G0", "G1", "G2", "G3"])
    ap.add_argument("--models", nargs="*", default=L.MODELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag
    LOG = os.path.join(L.LOG, f"03_main{tag}.log")
    if os.path.exists(LOG):
        os.remove(LOG)
    L.log(f"=== 03 主实验 groups={args.groups} models={args.models} ===", LOG)

    df = L.load_base()
    folds = df["fold"].values

    fold_rows, sum_rows, tr_rows, par_rows, oof_rows = [], [], [], [], []
    t_all = time.time()

    for g in args.groups:
        factors = L.GROUPS[g]
        for m in args.models:
            t0 = time.time()
            res = L.run_spatial_cv(df, factors, m, seed=args.seed,
                                   folds=folds, tune=True)
            dt = time.time() - t0
            row = L.summarize_cv(res, m, dataset=g)
            row["group"] = g
            row["group_desc"] = L.GROUP_DESC[g]
            row["runtime_s"] = round(dt, 1)
            sum_rows.append(row)
            res["fold_metrics"]["group"] = g
            res["train_metrics"]["group"] = g
            res["best_params"]["group"] = g
            fold_rows.append(res["fold_metrics"])
            tr_rows.append(res["train_metrics"])
            par_rows.append(res["best_params"])
            # 【v2 · C25】OOF 表含 ID/cell/fold/y/p/threshold/yhat
            _thr = res["thr_by_fold"]
            _fold_series = pd.Series(folds, index=df.index)
            _thr_vec = _fold_series.map(_thr).values
            oof_rows.append(pd.DataFrame({
                "pointid": df["pointid"], "cell_id": df["cell_id"],
                "group": g, "model": m,
                "fold": folds, "y": df["landslide"].values, "oof": res["oof"],
                "threshold": _thr_vec,
                "yhat": (res["oof"] >= _thr_vec).astype(int)}))
            L.log(f"{g}/{m}: AUC={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f} "
                  f"PR-AUC={row['pr_auc_mean']:.4f} F1={row['f1_mean']:.4f} "
                  f"OOF_AUC={row['oof_roc_auc']:.4f} gap={row['gap_roc_auc']:+.4f} "
                  f"({dt:.0f}s)", LOG)

    suffix = tag if tag else ""
    # 支持分批补跑（如单独补 SVM）：与既有结果按 (group, model) 合并，
    # 同 (group, model) 以本次结果为准，避免整表被覆盖。
    def _merge(new_df, path, keys):
        if tag or not os.path.exists(path):
            return new_df
        old = pd.read_csv(path)
        mask = old.set_index(keys).index.isin(
            new_df.set_index(keys).index)
        return pd.concat([old[~mask], new_df], ignore_index=True)

    out_fold = pd.concat(fold_rows)
    out_tr = pd.concat(tr_rows)
    out_par = pd.concat(par_rows)
    out_oof = pd.concat(oof_rows)
    S_new = pd.DataFrame(sum_rows)
    P = lambda n: os.path.join(L.RES, f"{n}{suffix}.csv")
    _merge(out_fold, P("03_逐折指标"), ["group", "model", "fold"]).to_csv(
        P("03_逐折指标"), index=False, encoding="utf-8-sig")
    _merge(out_tr, P("03_训练集指标"), ["group", "model", "fold"]).to_csv(
        P("03_训练集指标"), index=False, encoding="utf-8-sig")
    _merge(out_par, P("03_最优超参数"), ["group", "model", "fold"]).to_csv(
        P("03_最优超参数"), index=False, encoding="utf-8-sig")
    _merge(out_oof, P("03_OOF预测"), ["group", "model", "pointid"]).to_csv(
        P("03_OOF预测"), index=False, encoding="utf-8-sig")
    S = _merge(S_new, P("03_汇总"), ["group", "model"])
    S.to_csv(P("03_汇总"), index=False, encoding="utf-8-sig")
    L.log(f"\n总耗时 {time.time()-t_all:.0f}s", LOG)
    L.log("\n汇总:\n" + S[["group", "model", "roc_auc_mean", "roc_auc_std",
                          "pr_auc_mean", "f1_mean",
                          "accuracy_mean", "oof_roc_auc", "gap_roc_auc"]]
          .to_string(index=False), LOG)

    # 报告
    piv = S.pivot_table(index="model", columns="group",
                        values="roc_auc_mean").round(4)
    best = S.loc[S.groupby("group")["roc_auc_mean"].idxmax()]
    rep = ["# 阶段三 主实验：六类模型 × 四组因子（空间分块5折CV）\n",
           f"- 样本：702（滑坡351/非滑坡351）；折叠：空间块KMeans 5折，seed=42",
           "", "## ROC-AUC（折均值）", "", piv.to_markdown(), "",
           "## 各因子组最优模型", "",
           best[["group", "model", "roc_auc_mean", "pr_auc_mean", "f1_mean",
                 "accuracy_mean", "balanced_accuracy_mean", "brier_mean",
                 "oof_roc_auc", "gap_roc_auc"]].to_markdown(index=False), "",
           "## 全量汇总", "",
           S[["group", "model", "roc_auc_mean", "roc_auc_std",
              "pr_auc_mean", "f1_mean", "accuracy_mean", "precision_mean",
              "recall_mean", "specificity_mean", "balanced_accuracy_mean",
              "brier_mean", "oof_roc_auc", "train_roc_auc", "gap_roc_auc"]]
           .to_markdown(index=False), ""]
    with open(os.path.join(L.RES, f"03_主实验报告{suffix}.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(str(x) for x in rep))
    L.log("=== 03 完成 ===", LOG)


if __name__ == "__main__":
    main()
