# -*- coding: utf-8 -*-
"""
17_rain_selection.py — 降雨候选指标的折内筛选（v3 · R03）
=======================================================
v2 审查 R03 的四点修正：
  ① **土壤**：候选分支的 BASE 与 G2 相同（13 因子 = 11 基础 + clay + silt），
     两分支只差"是否执行指标选择"，不再同时改动土壤。
  ② **数据来源**：12 个候选全部来自 17a 的同一生产路径（0.05° 源格点 → 最近邻
     → 432 m），rx5day / raincv 与固定分支逐点一致（17a 已断言）；找不到源指标的
     候选一律标为"不可比"并剔除，不用克里金版顶替。
  ③ **选择规则预先登记**：先按类别各取判别力最高者（|AUC−0.5|），
     **然后**只在已选出的变量之间检查相关，|r|≥0.7 时保留判别力更高者、
     低者退让且**不补位**。这一行为写死在下面，与结果无关。
  ④ **留痕**：候选内容 SHA-256、每折训练 ID 哈希、候选评分、保留/排除理由、
     最优参数、阈值、702 条 OOF 全部落盘；用配对空间块 bootstrap 给差值区间。

输出：results/17_折内选择明细.csv, 17_折内选择频率.csv, 17_折内选择对比.csv,
      17_选择OOF.csv, 17_候选评分.csv, 17_降雨折内筛选报告.md
"""
import os
import sys
import json
import time
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

LOG = os.path.join(L.LOG, "17_rain_selection.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 17 降雨候选指标折内筛选（v3 · R03）===", LOG)

CAND = {
    "rx5day": "强度", "rx1day": "强度", "r99p": "强度", "r95p": "强度",
    "r95frac": "强度", "sdii": "强度",
    "raincv": "变率", "pci": "变率", "cdd": "变率", "cwd": "变率",
    "prcptot_near": "量级", "raindays": "量级",
}
# 与 G2 完全一致的基础因子（含土壤）——两分支唯一差别是"是否选择指标"
BASE = ["DEM", "slope", "aspect", "plan", "profile", "NDVI", "land_use",
        "roughness", "road", "river", "soil_clay", "soil_silt"]
R_THR = 0.7
MODEL = "XGBoost"
CATS = ["强度", "变率", "量级"]

# ---------- 候选值（17a 统一路径产出）----------
cp = os.path.join(L.DATA, "rain_candidates.csv")
if not os.path.exists(cp):
    raise SystemExit("缺 data/rain_candidates.csv，请先运行 17a_rain_candidates.py")
CAND_DF = pd.read_csv(cp)
df = L.load_base()
if not (CAND_DF["pointid"].values == df["pointid"].values).all():
    raise SystemExit("候选表与建模表 pointid 顺序不一致")
for c in CAND:
    if c in CAND_DF.columns:
        df[c] = L.canon_numeric(CAND_DF[c].values)

AVAIL = [c for c in CAND if c in df.columns and df[c].notna().all()]
UNCMP = [c for c in CAND if c not in AVAIL]
if UNCMP:
    L.log(f"[不可比] 缺源指标、不参与筛选：{UNCMP}", LOG)
L.log(f"可用候选 {len(AVAIL)}/{len(CAND)}: {AVAIL}", LOG)

with open(os.path.join(L.CFG, "rain_candidates_manifest.json"),
          encoding="utf-8") as f:
    CMAN = json.load(f)
L.log("候选内容哈希已登记（rain_candidates_manifest.json）", LOG)

folds = df["fold"].values
y = df["landslide"].astype(int).values


def disc_auc(yy, v):
    """方向无关的判别力：max(AUC, 1-AUC)。"""
    a = roc_auc_score(yy, v)
    return max(a, 1 - a)


def select_rule(dtr, ytr):
    """
    预先登记的选择规则（与结果无关）：
      第 1 步：每个类别（强度/变率/量级）各取判别力最高的 1 个；
      第 2 步：只在已选出的 ≤3 个变量之间两两查相关，|r|≥R_THR 时
              保留判别力更高者，另一者退让；**退让后不补位**。
    返回 (picked, scores, reason_rows)
    """
    sc = {}
    for k in AVAIL:
        v = dtr[k].values
        if np.nanstd(v) == 0:
            continue
        sc[k] = disc_auc(ytr, v)
    # 第 1 步
    picked, why = [], []
    for cat in CATS:
        cand = [k for k in sc if CAND[k] == cat]
        if not cand:
            continue
        best = max(cand, key=lambda k: sc[k])
        picked.append(best)
        for k in cand:
            why.append({"step": 1, "category": cat, "candidate": k,
                        "score": round(sc[k], 6),
                        "decision": "选中" if k == best else "类别内未选中"})
    # 第 2 步
    final = list(picked)
    for i in range(len(picked)):
        for j in range(i + 1, len(picked)):
            a, b = picked[i], picked[j]
            if a not in final or b not in final:
                continue
            r = float(np.corrcoef(dtr[a].values, dtr[b].values)[0, 1])
            if abs(r) >= R_THR:
                drop = a if sc[a] < sc[b] else b
                final.remove(drop)
                why.append({"step": 2, "category": "-", "candidate": drop,
                            "score": round(sc[drop], 6),
                            "decision": f"因与另一已选变量 |r|={abs(r):.3f}"
                                        f"≥{R_THR} 退让（不补位）"})
    return final, sc, why


rows, sel_rows, picks, oof_rows, par_rows = [], [], [], [], []
oof = np.full(len(df), np.nan)
for k in range(5):
    tr = np.where(folds != k)[0]
    te = np.where(folds == k)[0]
    dtr = df.iloc[tr].reset_index(drop=True)
    dte = df.iloc[te].reset_index(drop=True)
    ytr, yte = y[tr], y[te]
    t0 = time.time()

    picked, sc, why = select_rule(dtr, ytr)
    picks.append(picked)
    fac = BASE + picked
    L.assert_registry(); L.require_factors(fac)

    base_model, grid = L.get_model_and_grid(MODEL, 42, n_jobs=1)
    best, thr, iauc, _, _ = L.inner_select_raw(
        base_model, grid, dtr, ytr, fac, MODEL, seed=42, n_jobs=10)

    Xa, enc = L.build_matrix(dtr, fac, fit=True)
    Xb, _ = L.build_matrix(dte, fac, enc=enc, fit=False)
    mdl = clone(base_model); mdl.set_params(**best); mdl.fit(Xa, ytr)
    p = mdl.predict_proba(Xb)[:, 1]
    oof[te] = p
    auc_te = roc_auc_score(yte, p)

    train_ids_sha = hashlib.sha256(
        "|".join(sorted(map(str, dtr["pointid"].values))).encode()).hexdigest()
    sel_rows.append({
        "fold": k, "selected": "+".join(picked),
        "replaced_fixed": ("否" if set(picked) == {"rx5day", "raincv"} else "是"),
        "n_candidates": len(AVAIL), "n_incomparable": len(UNCMP),
        "train_ids_sha256": train_ids_sha,
        "matrix_sha256": enc["matrix_sha256"],
        "inner_auc": iauc, "test_auc": auc_te, "threshold": thr,
        "params": json.dumps(best, ensure_ascii=False),
        "top5_scores": json.dumps({k2: round(v, 4) for k2, v in
                                   sorted(sc.items(), key=lambda x: -x[1])[:5]}),
        "runtime_s": round(time.time() - t0, 1)})
    par_rows.append({"fold": k, "model": MODEL, "threshold": thr,
                     "params": json.dumps(best, ensure_ascii=False),
                     "n_factors": len(fac),
                     "selected": "+".join(picked)})
    for r in why:
        r["fold"] = k
        rows.append(r)
    L.log(f"折 {k}: 选中 {picked}（内层AUC={iauc:.4f}，外测AUC={auc_te:.4f}）", LOG)

SEL = pd.DataFrame(sel_rows)
SEL.to_csv(os.path.join(L.RES, "17_折内选择明细.csv"), index=False,
           encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(os.path.join(L.RES, "17_候选评分.csv"), index=False,
                          encoding="utf-8-sig")
pd.DataFrame(par_rows).to_csv(os.path.join(L.RES, "17_最优超参数.csv"),
                              index=False, encoding="utf-8-sig")
pd.DataFrame({"pointid": df["pointid"].values, "fold": folds, "y": y,
              "oof_select": oof,
              "block_id": df["spatial_block_id"].values}).to_csv(
    os.path.join(L.RES, "17_选择OOF.csv"), index=False, encoding="utf-8-sig")

freq = pd.Series([f for p_ in picks for f in p_]).value_counts()
FREQ = pd.DataFrame({"指标": freq.index, "被选中折数": freq.values,
                     "选择频率": freq.values / 5})
FREQ.to_csv(os.path.join(L.RES, "17_折内选择频率.csv"), index=False,
            encoding="utf-8-sig")
L.log("\n选择频率:\n" + FREQ.to_string(index=False), LOG)

# ---------- 与固定分支成对比较（同一批样本、同一批空间块）----------
CMP = []
g2 = None
p5 = os.path.join(L.RES, "05_消融OOF.csv")
if os.path.exists(p5):
    o5 = pd.read_csv(p5)
    g2 = o5[(o5["variant"] == "G2_full") & (o5["model"] == MODEL)]
if g2 is not None and len(g2) == len(df):
    g2 = g2.sort_values("pointid").reset_index(drop=True)
    order = np.argsort(df["pointid"].values)
    sel_sorted = oof[order]
    d = L.paired_block_bootstrap(g2["y"].values, g2["oof"].values, sel_sorted,
                                 g2["block_id"].values, n_boot=2000, seed=42)
    # paired_block_bootstrap(y, p_a, p_b) 返回 auc_a 对应 p_a(=G2)、auc_b 对应 p_b(=选择)，
    # delta = auc_a - auc_b = 固定 - 选择。对外要报「选择 - 固定」，故取负号并翻转区间。
    CMP.append({"分支": "折内选择（本脚本）", "AUC": d["auc_b"],
                "对照": "固定 rx5day+raincv（= G2_full，15 因子）",
                "AUC_对照": d["auc_a"], "ΔAUC_选择-固定": -d["delta"],
                "CI_lo": -d["hi"], "CI_hi": -d["lo"],
                "P(Δ≥0)": d["p_le0"],
                "逐折AUC": str([round(v, 4) for v in SEL["test_auc"]]),
                "说明": "两分支同 13 基础因子、同候选来源、同 fold、同网格；"
                        "AUC 为合并 OOF，区间为空间块成对 bootstrap"})
CMPDF = pd.DataFrame(CMP)
if len(CMPDF):
    CMPDF.to_csv(os.path.join(L.RES, "17_折内选择对比.csv"), index=False,
                 encoding="utf-8-sig")
    L.log("\n" + CMPDF.round(4).to_string(index=False), LOG)

rep = ["# 17 降雨候选指标的折内筛选（v3 · R03）\n", "## 一、口径修正", "",
       f"- 基础因子与 G2 相同（13 个，含 soil_clay / soil_silt）：`{'、'.join(BASE)}`；",
       f"- 12 个候选全部来自 `Rasters_native/*_0.05deg.tif` → 最近邻 → 432 m 的"
       f"**同一路径**；其中 rx5day、raincv 与固定分支逐点一致（0 不等，见 "
       f"`17a_候选一致性.csv`）；",
       f"- 不可比候选（缺源指标）：{UNCMP if UNCMP else '无'}；",
       "- 选择规则**预先登记**：各类别取判别力最高者 → 只在已选变量之间查相关 →"
       f" |r|≥{R_THR} 时低者退让且不补位。", "",
       "## 二、逐折选择", "", SEL.drop(columns=["params"]).to_markdown(index=False), "",
       "## 三、选择频率", "", FREQ.to_markdown(index=False), "",
       "## 四、与固定分支比较", "",
       (CMPDF.round(4).to_markdown(index=False) if len(CMPDF) else "（缺 G2 基线）"), "",
       "## 五、表述边界", "",
       "- 即使两分支存在差值，也只能称「两种流程的性能差异」，不能把一次 CV 的"
       "差值直接当作选择偏差的估计量；",
       "- 固定分支应表述为「在预先选定因子集上的条件评价」。", ""]
with open(os.path.join(L.RES, "17_降雨折内筛选报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in rep))
L.log("=== 17 完成 ===", LOG)
