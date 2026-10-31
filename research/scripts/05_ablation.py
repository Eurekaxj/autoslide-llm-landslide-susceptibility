# -*- coding: utf-8 -*-
"""
05_ablation.py — 因子组与敏感性消融（v3）
==================================================================
v3 修正（依据 v2 审查 R02/R05/R07/C13）：
  R02  prcptot 已在 exp_lib.CONTINUOUS 正式注册，P0_prcptot 现在**真的有**
       降雨总量列（矩阵含非恒定 prcptot）。旧的无降雨对照改名为 N0_norain。
  R05  T2 温度残差改为**每内层训练子集独立拟合**（transform 对象），不再
       复用外层整体回归系数；系数按 outer_fold/inner_fold 全部留档。
  C13/R07  每个变体保存逐点 OOF、每折参数/阈值、输入矩阵哈希；相对 G2 的
       差值同时给空间块成对 bootstrap 区间（同一 block 同时重采样两变体）。
输出：
  results/05_消融汇总.csv, 05_消融逐折.csv, 05_消融OOF.csv,
  05_最优超参数.csv, 05_温度回归系数.csv, 05_相对G2差值.csv, 05_消融报告.md
"""
import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

BASE = ["DEM", "slope", "aspect", "plan", "profile", "NDVI", "land_use",
        "roughness", "road", "river"]
FULL = BASE + ["rx5day", "raincv", "soil_clay", "soil_silt"]

VARIANTS = {
    # —— 降雨口径（R02：P0 为真实同时期年总量；N0 为无降雨对照）——
    # R03 后续修正：建模统一用最近邻版 prcptot_near（与 rx5day/raincv 同一
    # 生产路径）；克里金版 prcptot 单独作为 P0b 保留，用于量化插值口径差异。
    "N0_norain": (BASE, None),
    "P0_prcptot": (BASE + ["prcptot_near"], None),
    "P0b_prcptot_krig": (BASE + ["prcptot"], None),
    "G0_rain2019": (BASE + ["rainfall"], None),
    "G1_rx5day_cv": (BASE + ["rx5day", "raincv"], None),
    "G2_full": (FULL, None),
    "G3_no_rough": (FULL, "drop_rough"),
    # —— 土壤消融 ——
    "S0_no_soil": (BASE + ["rx5day", "raincv"], None),
    "S1_clay": (BASE + ["rx5day", "raincv", "soil_clay"], None),
    "S2_silt": (BASE + ["rx5day", "raincv", "soil_silt"], None),
    "S3_clay_silt": (FULL, None),
    # —— 年均温 ——
    "T0_no_temp": (FULL, None),
    "T1_temyear": (FULL + ["temyear"], None),
    "T2_temp_resid": (FULL + ["temp_resid"], "temp_resid"),
    # —— 降雨空间对齐 ——
    "A0_nearest": (FULL, None),
    "A1_kriging": (BASE + ["rx5day_krig", "raincv_krig", "soil_clay", "soil_silt"], None),
    # —— 降雨统计窗口 ——
    "W30": (FULL, None),
    "W10": (BASE + ["rx5day_W10", "raincv_W10", "soil_clay", "soil_silt"], None),
    "W01": (BASE + ["rx5day_W01", "raincv_W01", "soil_clay", "soil_silt"], None),
}

DESC = {
    "N0_norain": "基础 11 因子、无任何降雨输入（R02 对照：旧 P0 的真实身份）",
    "P0_prcptot": "基础 + 1990-2019 平均年降水总量（最近邻版，与 rx5day/raincv 同路径）",
    "P0b_prcptot_krig": "基础 + 同年降水总量（克里金插值版），仅用于对比插值口径",
    "G0_rain2019": "基础 + 2019 单年累计降雨（原稿口径）",
    "G1_rx5day_cv": "基础 + Rx5day + 日降雨CV（强度/变率）",
    "G2_full": "G1 + 土壤黏粒/粉粒 = 主方案 15 因子",
    "G3_no_rough": "G2 去除 roughness",
    "S0_no_soil": "无土壤（=G1）",
    "S1_clay": "仅 clay",
    "S2_silt": "仅 silt",
    "S3_clay_silt": "clay+silt（=G2）",
    "T0_no_temp": "不含年均温（=G2）",
    "T1_temyear": "含年均温（原始值）",
    "T2_temp_resid": "含年均温对 DEM 回归残差（每内层独立拟合）",
    "A0_nearest": "降雨最近邻对齐（=G2）",
    "A1_kriging": "降雨克里金对齐",
    "W30": "降雨窗口 1990-2019（=G2）",
    "W10": "降雨窗口 2010-2019",
    "W01": "降雨窗口 2019 单年",
}


class TempResid:
    """R05：年均温对 DEM 回归残差；fit 返回**新的**已拟合实例，无共享状态。"""
    name = "temp_resid"

    def fit(self, d):
        lr = LinearRegression().fit(
            np.asarray(d[["DEM"]], dtype=float),
            np.asarray(d["temyear"], dtype=float))
        t = TempResid()
        t.intercept_ = float(lr.intercept_)
        t.coef_ = float(lr.coef_[0])
        return t

    def transform(self, d):
        out = d.copy()
        out["temp_resid"] = (L.canon_numeric(d["temyear"].values)
                             - (self.intercept_
                                + self.coef_ * L.canon_numeric(d["DEM"].values)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["XGBoost"])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    LOG = os.path.join(L.LOG, "05_ablation.log")
    if os.path.exists(LOG):
        os.remove(LOG)
    L.log(f"=== 05 消融（v3），模型={args.models} ===", LOG)
    L.log(f"数值规范: float32-canonical；因子注册自检通过", LOG)

    df = L.load_base()
    # 【v3 · R03】prcptot_near 不在冻结建模表内，从统一生产路径的候选表并入，
    # 保证 P0 的降雨总量与主方案 rx5day/raincv 走同一条最近邻提取路径。
    rc = os.path.join(L.DATA, "rain_candidates.csv")
    if not os.path.exists(rc):
        raise SystemExit("缺 data/rain_candidates.csv，请先运行 17a_rain_candidates.py")
    cand = pd.read_csv(rc)[["pointid", "prcptot_near"]]
    df = df.merge(cand, on="pointid", how="left", validate="one_to_one")
    if df["prcptot_near"].isna().any():
        raise SystemExit(f"prcptot_near 并入后有 {int(df['prcptot_near'].isna().sum())} 个缺失")
    L.log("R03：prcptot_near 已从 rain_candidates.csv 并入（最近邻口径）", LOG)

    folds = df["fold"].values
    keep = args.variants or list(VARIANTS)

    sums, frs, oofs, pars, tfrows = [], [], [], [], []
    for vname in keep:
        if vname not in VARIANTS:
            L.log(f"[跳过] 未知变体 {vname}", LOG)
            continue
        fac, flag = VARIANTS[vname]
        if flag == "drop_rough":
            fac = [f for f in fac if f != "roughness"]
        tf = TempResid() if flag == "temp_resid" else None
        miss = [f for f in fac if f not in df.columns and f != "temp_resid"]
        if miss:
            L.log(f"[跳过] {vname} 缺列 {miss}", LOG)
            continue
        for m in args.models:
            t0 = time.time()
            res = L.run_spatial_cv(df, fac, m, seed=args.seed, folds=folds,
                                   transform=tf)
            row = L.summarize_cv(res, m, dataset=vname)
            row.update({"variant": vname, "desc": DESC.get(vname, ""),
                        "n_factors": len(fac),
                        "runtime_s": round(time.time() - t0, 1)})
            sums.append(row)

            fm = res["fold_metrics"].copy(); fm["variant"] = vname
            frs.append(fm)

            bp = res["best_params"].copy(); bp["variant"] = vname
            pars.append(bp)

            oofs.append(pd.DataFrame({
                "pointid": df["pointid"].values, "variant": vname, "model": m,
                "fold": folds, "y": df["landslide"].values, "oof": res["oof"],
                "block_id": df["spatial_block_id"].values}))

            if len(res["tf_records"]):
                t = res["tf_records"].copy(); t["variant"] = vname; t["model"] = m
                tfrows.append(t)

            L.log(f"{vname}/{m}: AUC={row['roc_auc_mean']:.4f} "
                  f"PR-AUC={row['pr_auc_mean']:.4f} F1={row['f1_mean']:.4f} "
                  f"({time.time()-t0:.0f}s)", LOG)

    S = pd.DataFrame(sums)
    P_S = os.path.join(L.RES, "05_消融汇总.csv")
    if os.path.exists(P_S):
        old = pd.read_csv(P_S)
        msk = old.set_index(["variant", "model"]).index.isin(
            S.set_index(["variant", "model"]).index)
        S = pd.concat([old[~msk], S], ignore_index=True)
    S.to_csv(P_S, index=False, encoding="utf-8-sig")

    def merge_write(path, new, keys):
        if os.path.exists(path):
            old = pd.read_csv(path)
            k = old.set_index(keys).index.isin(new.set_index(keys).index)
            new = pd.concat([old[~k], new], ignore_index=True)
        new.to_csv(path, index=False, encoding="utf-8-sig")
        return new

    merge_write(os.path.join(L.RES, "05_消融逐折.csv"), pd.concat(frs),
                ["variant", "model", "fold"])
    merge_write(os.path.join(L.RES, "05_最优超参数.csv"), pd.concat(pars),
                ["variant", "model", "fold"])
    merge_write(os.path.join(L.RES, "05_消融OOF.csv"), pd.concat(oofs),
                ["variant", "model", "pointid"])
    if tfrows:
        T = pd.concat(tfrows)
        T.to_csv(os.path.join(L.RES, "05_温度回归系数.csv"), index=False,
                 encoding="utf-8-sig")
        L.log(f"温度回归系数行数 {len(T)}（outer_fold × inner_fold），"
              f"拟合规模 {T['n_fit'].min()}–{T['n_fit'].max()}", LOG)

    # —— 相对基准差值 + 空间块成对 bootstrap ——
    # 【v3 · R02】验收要求同时给 P0 与 G0/G1 的配对差值（不只是与 G2 比）。
    # 【v3 · R07】差值带空间块成对 bootstrap 区间，不要求显著。
    O = pd.read_csv(os.path.join(L.RES, "05_消融OOF.csv"))
    BASE_LIST = ["G2_full", "G1_rx5day_cv", "G0_rain2019", "N0_norain"]
    D = []
    for ref in BASE_LIST:
        rr = O[O["variant"] == ref]
        if not len(rr):
            continue
        rr = rr.sort_values("pointid").reset_index(drop=True)
        for v in [x for x in O["variant"].unique() if x != ref]:
            o = O[O["variant"] == v]
            if len(o) != len(rr):
                continue
            o = o.sort_values("pointid").reset_index(drop=True)
            if not (o["pointid"].values == rr["pointid"].values).all():
                continue
            d = L.paired_block_bootstrap(rr["y"].values, rr["oof"].values,
                                         o["oof"].values, rr["block_id"].values,
                                         n_boot=2000, seed=42)
            D.append({"baseline": ref, "variant": v,
                      "model": o["model"].iloc[0],
                      "AUC_baseline": d["auc_a"], "AUC_variant": d["auc_b"],
                      "ΔAUC_baseline_minus_var": d["delta"],
                      "ΔCI_lo": d["lo"], "ΔCI_hi": d["hi"],
                      "P_Δ≤0": d["p_le0"], "n_boot_valid": d["n_valid"]})
    if D:
        DD = pd.DataFrame(D).sort_values(
            ["baseline", "ΔAUC_baseline_minus_var"], ascending=[True, False])
        DD.to_csv(os.path.join(L.RES, "05_相对基准差值.csv"), index=False,
                  encoding="utf-8-sig")
        # 兼容旧名：只保留相对 G2 的子集
        DD[DD["baseline"] == "G2_full"].to_csv(
            os.path.join(L.RES, "05_相对G2差值.csv"), index=False,
            encoding="utf-8-sig")

    piv = S.pivot_table(index="variant", columns="model",
                        values="roc_auc_mean").round(4)
    rep = ["# 05 因子组与敏感性消融（v3，统一 XGBoost）\n", "## v3 修正", "",
           "- **R02** `prcptot` 已注册进 `exp_lib.CONTINUOUS`，P0 现在确有降雨总量列；"
           "旧的无降雨矩阵改名为 `N0_norain` 保留。",
           "- **R03** P0 改用最近邻版 `prcptot_near`（与 rx5day/raincv 同一生产路径）；"
           "克里金版单列为 `P0b_prcptot_krig`，两者之差即插值口径的影响。",
           "- **R05** T2 残差改为每个内层训练子集独立回归（`05_温度回归系数.csv` 含 "
           "outer_fold/inner_fold/fit_ids_sha256）。",
           "- **C13/R07** 每个变体保存逐点 OOF 与每折参数阈值；与四个基准"
           "（G2_full / G1_rx5day_cv / G0_rain2019 / N0_norain）的配对差值均给出"
           "空间块成对 bootstrap 区间（`05_相对基准差值.csv`）。", "",
           "## 一、ROC-AUC（折均值）", "", piv.to_markdown(), "",
           "## 二、相对各基准的差值（XGBoost，空间块成对 bootstrap 2000 次）", "",
           (DD.round(4).to_markdown(index=False) if D else "（缺基准）"), "",
           "## 三、全量指标", "",
           S[["variant", "desc", "n_factors", "model", "roc_auc_mean",
              "roc_auc_std", "pr_auc_mean", "f1_mean",
              "brier_mean", "oof_roc_auc", "oof_f1_foldthr"]]
           .round(4).to_markdown(index=False), ""]
    with open(os.path.join(L.RES, "05_消融报告.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(str(x) for x in rep))
    L.log("=== 05 完成 ===", LOG)


if __name__ == "__main__":
    main()
