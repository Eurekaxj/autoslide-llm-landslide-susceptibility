# -*- coding: utf-8 -*-
"""
XGBoost_v3.py — AutoSlide 建模后端（返修 v3 协议）
====================================================================================
与同目录 `XGBoost.py` **命令行接口完全一致**，供 AutoSlide GUI 的
「Model Design」页直接调用（GUI 用 `{model_name}.py` 拼脚本名，
因此把方法名选为 `XGBoost_v3` 即可命中本文件）。

与原 `XGBoost.py` 的协议差异（逐条对应返修要求）
--------------------------------------------------
| 项 | 原 XGBoost.py | 本文件 |
|---|---|---|
| 采样 | 从训练表按 1:1 重新抽样负类 | **不重采样**：直接使用冻结 702 记录 |
| 划分 | `train_test_split(test_size=0.2, random_state=42)` 随机划分 | **空间分块 5 折**，折号取自输入表的 `fold` 列（冻结折表） |
| 预处理 | `MinMaxScaler().fit_transform(X)` 全量拟合后再划分 | **折内拟合**：标准化只在训练折上 `fit` |
| 调参 | `GridSearchCV(cv=3)` 在外层训练集上普通 K 折 | **内层 cell 分组 5 折**（`StratifiedGroupKFold`，按 `cell_id` 分组） |
| 阈值 | 固定 0.5 | **折内内层 OOF 选 F1 最优阈值** |
| 指标 | 单次留出集 | 逐折指标 + 合并 OOF |
| 留痕 | 仅保存模型 | 另存 OOF、逐折参数/阈值、输入与代码哈希 |

命令行接口（与 GUI 一致，不可删改参数名）
    --result_dir --predictres_dir --prediction_file --train_files
    --param_grid --features --label_col

退出码：0 成功；非 0 表示训练/预测失败（GUI 日志会显示 "Fail"）。
"""
import os
import sys
import json
import argparse
import hashlib
import functools
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from joblib import dump

print = functools.partial(print, flush=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

SEED = 42
N_SPLITS = 5


def parse_args():
    p = argparse.ArgumentParser(description="XGBoost (v3 spatial-CV protocol)")
    p.add_argument("--result_dir", required=True)
    p.add_argument("--predictres_dir", required=True)
    p.add_argument("--prediction_file", required=True)
    p.add_argument("--train_files", required=True,
                   help="训练表（分号分隔，支持 .csv/.xls/.xlsx）")
    p.add_argument("--param_grid", type=str, default="",
                   help="JSON 或 param1=1,2;param2=0.1,0.2")
    p.add_argument("--features", type=str, default="")
    p.add_argument("--label_col", type=str, default="landslide")
    p.add_argument("--fold_col", type=str, default="fold",
                   help="空间折号列（冻结折表提供）")
    p.add_argument("--group_col", type=str, default="cell_id",
                   help="内层分组列（防止同像元跨内层折）")
    return p.parse_args()


def parse_param_grid(s):
    if not s:
        return {"n_estimators": [400], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]}
    try:
        return json.loads(s)
    except Exception:                                          # noqa: BLE001
        grid = {}
        for item in s.split(";"):
            if "=" in item:
                k, v = item.split("=")
                vals = []
                for x in v.split(","):
                    x = x.strip()
                    vals.append(float(x) if "." in x else int(x))
                grid[k.strip()] = vals
        return grid


def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"不支持的训练表格式：{path}")


def sha256(path, n=64):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:n]


def threshold_from_oof(oof, y):
    best, bt = -1.0, 0.5
    for t in np.unique(np.round(np.linspace(0.05, 0.95, 91), 3)):
        s = f1_score(y, (oof >= t).astype(int), zero_division=0)
        if s > best:
            best, bt = s, float(t)
    return bt


# ---------------------------------------------------------------------------
# 与主链一致的因子编码与数值规范（使 GUI 输出可与 03 主实验逐点对照）
#   · 连续因子先 float32 再 float64（float32-canonical，消除提取路径精度伪特征）
#   · aspect → aspect_sin / aspect_cos / aspect_flat（<0 记为平地）
#   · land_use → one-hot（类别集合由训练折决定，验证/预测按训练列重索引）
#   · 标准化用 ddof=0（与主链 build_matrix 一致，StandardScaler 默认 ddof=1 会引入差异）
# ---------------------------------------------------------------------------
def canon(a):
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def encode_factors(df, feats, cats=None):
    """返回 (矩阵 DataFrame, 连续列名, 类别集合)。"""
    cont = [f for f in feats if f not in ("aspect", "land_use")]
    X = pd.DataFrame({f: canon(df[f].values) for f in cont}, index=df.index)
    if "aspect" in feats:
        a = canon(df["aspect"].values)
        ar = np.where(a < 0, 0.0, np.deg2rad(a))
        X["aspect_sin"] = np.sin(ar)
        X["aspect_cos"] = np.cos(ar)
        X["aspect_flat"] = (a < 0).astype(float)
    if "land_use" in feats:
        lu = df["land_use"].astype(int)
        if cats is None:
            cats = sorted(lu.unique())
        d = pd.get_dummies(lu.astype("category").cat.set_categories(cats))
        d.columns = [f"lu_{int(c)}" for c in d.columns]
        d = d.astype(float)
        X = pd.concat([X, d], axis=1)
    return X, cont, cats


def std_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def std_apply(X, mu, sd):
    return (X - mu) / sd


def fold_matrices(df, itr, ite, feats):
    """
    按**训练子集的类别集合**编码 + 用训练子集统计量标准化（ddof=0）。
    与主链 build_matrix(fit=True/fit=False) 的口径一致：
    验证/测试子集不参与 land_use 类别确定，也不参与均值方差估计。
    """
    Xtr_df, _cont, cats = encode_factors(df.iloc[itr], feats)
    Xte_df, _, _ = encode_factors(df.iloc[ite], feats, cats=cats)
    Xte_df = Xte_df.reindex(columns=list(Xtr_df.columns), fill_value=0.0)
    A = np.asarray(Xtr_df, dtype=float)
    B = np.asarray(Xte_df, dtype=float)
    mu, sd = std_fit(A)
    return std_apply(A, mu, sd), std_apply(B, mu, sd)


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.predictres_dir, exist_ok=True)

    train_files = [x for x in args.train_files.split(";") if x.strip()]
    if not train_files:
        raise SystemExit("train_files 为空")
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    if not feats:
        raise SystemExit("features 为空")
    grid = parse_param_grid(args.param_grid)
    combos = list(ParameterGrid(grid))
    print(f"[v3] 训练表 {len(train_files)} 个；因子 {len(feats)} 个；"
          f"参数组合 {len(combos)} 组")

    frames = []
    for f in train_files:
        d = read_table(f)
        print(f"[v3] 读入 {os.path.basename(f)}：{d.shape[0]} 行 × {d.shape[1]} 列")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    need = feats + [args.label_col, args.fold_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"训练表缺列：{miss}")
    if df[args.label_col].isna().any():
        raise SystemExit("标签列存在缺失")
    y = df[args.label_col].astype(int).values
    if len(np.unique(y)) != 2:
        raise SystemExit("标签列必须为二分类（显式 1=滑坡，0=非滑坡）")
    folds = df[args.fold_col].astype(int).values
    groups = (df[args.group_col].values if args.group_col in df.columns
              else np.arange(len(df)))
    print(f"[v3] 样本 {len(df)}（1=滑坡 {int(y.sum())} / 0=非滑坡 "
          f"{int((1 - y).sum())}）；空间折 {sorted(set(folds.tolist()))}")

    Xdf, cont_cols, cats = encode_factors(df, feats)
    X_all = np.asarray(Xdf, dtype=float)
    print(f"[v3] 编码后矩阵 {X_all.shape}（连续 {len(cont_cols)} + "
          f"坡向派生 + 土地利用哑变量）")

    oof = np.zeros(len(df))
    rows, param_rows = [], []

    base = XGBClassifier(random_state=SEED, eval_metric="logloss",
                         n_jobs=1, tree_method="hist")
    # n_jobs=1 与主链 get_model_and_grid(..., n_jobs=1) 保持一致：
    # 多线程直方图累加的浮点求和顺序不同，会让内层近似并列的参数组合换序，
    # 进而选到不同超参数，导致与主实验不可逐点比较。

    for k in sorted(set(folds.tolist())):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]
        if len(te) == 0 or len(tr) == 0:
            raise SystemExit(f"折 {k} 为空，检查 fold 列")
        Xtr, Xte = fold_matrices(df, tr, te, feats)     # 折内编码 + 折内标准化

        inner = StratifiedGroupKFold(n_splits=min(N_SPLITS, len(np.unique(
            groups[tr]))), shuffle=True, random_state=SEED)
        splits = list(inner.split(Xtr, y[tr], groups=groups[tr]))

        best_auc, best_p, best_oof = -1.0, combos[0], None
        for params in combos:
            ioof = np.full(len(tr), np.nan)
            for a, b in splits:
                A, B = fold_matrices(df, tr[a], tr[b], feats)   # 内层再折内编码
                m = clone(base).set_params(**params)
                m.fit(A, y[tr][a])
                ioof[b] = m.predict_proba(B)[:, 1]
            ok = ~np.isnan(ioof)
            auc = roc_auc_score(y[tr][ok], ioof[ok])
            if auc > best_auc:
                best_auc, best_p, best_oof = auc, dict(params), ioof
        print(f"[v3] fold {k}: 内层网格 {len(combos)} 组完成，"
              f"最优 inner AUC={best_auc:.4f}")

        thr = threshold_from_oof(best_oof, y[tr])           # 折内阈值
        model = clone(base).set_params(**best_p)
        model.fit(Xtr, y[tr])
        oof[te] = model.predict_proba(Xte)[:, 1]

        yh = (oof[te] >= thr).astype(int)
        cm = confusion_matrix(y[te], yh, labels=[0, 1])
        rows.append({
            "fold": int(k), "n_train": int(len(tr)), "n_test": int(len(te)),
            "inner_auc": float(best_auc), "threshold": float(thr),
            "Test_Accuracy": float(accuracy_score(y[te], yh)),
            "Test_Precision": float(precision_score(y[te], yh, zero_division=0)),
            "Test_Recall": float(recall_score(y[te], yh, zero_division=0)),
            "Test_F1-Score": float(f1_score(y[te], yh, zero_division=0)),
            "Test_AUC": float(roc_auc_score(y[te], oof[te])),
        })
        param_rows.append({"fold": int(k), "threshold": float(thr),
                           "inner_auc": float(best_auc),
                           "params": json.dumps(best_p, ensure_ascii=False)})
        print(f"[v3] fold {k}: AUC={rows[-1]['Test_AUC']:.4f} "
              f"F1={rows[-1]['Test_F1-Score']:.4f} thr={thr:.3f}")

        fpr, tpr, _ = roc_curve(y[te], oof[te])
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, lw=2,
                 label=f"AUC = {rows[-1]['Test_AUC']:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC — spatial fold {k}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, f"ROC_{k}.png"), dpi=300)
        plt.close()

    # 合并 OOF 指标（用逐折阈值）
    thr_by_fold = {r["fold"]: r["threshold"] for r in rows}
    yh_all = np.array([(oof[i] >= thr_by_fold[folds[i]]).astype(int)
                       for i in range(len(oof))])
    pooled = {
        "Test_Accuracy": float(accuracy_score(y, yh_all)),
        "Test_Precision": float(precision_score(y, yh_all, zero_division=0)),
        "Test_Recall": float(recall_score(y, yh_all, zero_division=0)),
        "Test_F1-Score": float(f1_score(y, yh_all, zero_division=0)),
        "Test_AUC": float(roc_auc_score(y, oof)),
    }
    print(f"[v3] 合并 OOF：AUC={pooled['Test_AUC']:.4f} "
          f"F1={pooled['Test_F1-Score']:.4f}")

    M = pd.DataFrame(rows)
    M.loc["Average"] = M.mean(numeric_only=True)
    with pd.ExcelWriter(os.path.join(args.result_dir, "metrics.xlsx"),
                        engine="openpyxl") as w:
        M.to_excel(w, sheet_name="Metrics", index=False)

    pd.DataFrame({"pointid": df["pointid"].values if "pointid" in df.columns
                  else np.arange(len(df)),
                  "fold": folds, "y": y, "oof": oof,
                  "threshold": [thr_by_fold[f] for f in folds],
                  "yhat": yh_all}).to_csv(
        os.path.join(args.result_dir, "OOF_v3.csv"), index=False,
        encoding="utf-8-sig")
    pd.DataFrame(param_rows).to_csv(
        os.path.join(args.result_dir, "best_params_v3.csv"), index=False,
        encoding="utf-8-sig")

    # 平均 ROC
    mfpr = np.linspace(0, 1, 100)
    tprs = []
    for k in sorted(set(folds.tolist())):
        te = np.where(folds == k)[0]
        fpr, tpr, _ = roc_curve(y[te], oof[te])
        t = np.interp(mfpr, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    plt.figure()
    plt.plot(mfpr, mean_tpr, lw=2,
             label=f"Mean ROC (AUC = {pooled['Test_AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Mean ROC — spatial 5-fold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(args.result_dir, "Mean_ROC.png"), dpi=300)
    plt.close()

    # 最终模型（全量）→ 域预测
    pri = int(max(set(folds.tolist()),
                  key=lambda k: M.loc[M["fold"] == k, "Test_AUC"].values[0]))
    final_p = json.loads([r for r in param_rows
                          if r["fold"] == pri][0]["params"])
    mu_f, sd_f = std_fit(X_all)
    final = clone(base).set_params(**final_p).fit(
        std_apply(X_all, mu_f, sd_f), y)
    dump({"model": final, "mu": mu_f, "sd": sd_f, "features": feats,
          "threshold_by_fold": thr_by_fold, "params": final_p},
         os.path.join(args.result_dir, "XGBoost_v3_final.joblib"))

    pred = pd.read_csv(args.prediction_file)
    mp = [c for c in feats if c not in pred.columns]
    if mp:
        raise SystemExit(f"预测表缺因子列：{mp}")
    # 用**训练集学到的类别集合**编码域表，避免域内多出的 land_use 类别
    # 产生额外哑变量列（列数必须与训练矩阵一致）
    Xp, _, _ = encode_factors(pred, feats, cats=cats)
    Xp = Xp.reindex(columns=list(Xdf.columns), fill_value=0.0)
    if Xp.shape[1] != Xdf.shape[1]:
        raise SystemExit(f"域矩阵列数 {Xp.shape[1]} ≠ 训练 {Xdf.shape[1]}")
    pred["Susceptibility"] = final.predict_proba(
        std_apply(np.asarray(Xp, dtype=float), mu_f, sd_f))[:, 1]
    outp = os.path.join(args.predictres_dir, "prediction_v3.csv")
    pred.to_csv(outp, index=False, encoding="utf-8-sig")
    print(f"[v3] 预测完成，{len(pred)} 行 → {outp}")

    cfg = {
        "protocol": "v3-spatial-5fold-inner-nested",
        "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.executable,
        "code": {"XGBoost_v3.py": sha256(os.path.abspath(__file__))},
        "inputs": {os.path.basename(f): sha256(os.path.abspath(f))
                   for f in train_files + [args.prediction_file]},
        "n_samples": int(len(df)), "n_features": len(feats),
        "features": feats, "fold_col": args.fold_col,
        "group_col": args.group_col, "seed": SEED,
        "grid": grid, "n_combos": len(combos),
        "threshold_by_fold": thr_by_fold,
        "final_params": final_p,
        "pooled_oof": pooled,
    }
    with open(os.path.join(args.result_dir, "run_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("All Processes Completed")


if __name__ == "__main__":
    main()
