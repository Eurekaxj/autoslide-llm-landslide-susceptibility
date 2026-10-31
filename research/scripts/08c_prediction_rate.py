# -*- coding: utf-8 -*-
"""
08c_prediction_rate.py — 预测率曲线（v3.1 重写，V3-02）
====================================================================================
v3 审查指出两张口径问题，本版全部修掉：

  ① 旧版把**每折 OOF 概率**逐行写进域表（`cell_p.loc[...] = p_oof`），
     再用这张"半 OOF、半全样本模型"的混合概率求全域面积阈值 ——
     面积阈值、区域概率与滑坡概率来自三套不同的训练范围，曲线不可解释。
     且同一像元对应多条调查记录时，后写值会覆盖先写值。

  ② 计数单位混用：旧报告写"域内滑坡像元 213"，实为 213 条**记录**、206 个**像元**。

v3.1 做法（审查建议的最小方案）：
  · 载入 03 的**每折参数与阈值**，在每折训练集上重新拟合（不调参、不换网格）；
  · 用第 k 折模型给**该折留出区域的全部有效像元**打分（不是只给 702 个调查点）；
  · 把 5 折的留出区域拼成一张"逐像元留出打分图"——每个像元都由
    **未见过其所在空间块的模型**打分；
  · 在这张图上统一计算面积占比、命中率与背景误报率。

同时单列一条"全样本最终模型地图"的描述性曲线，明确标注
**它是拟合结果、不是留出精度**，两条曲线不混用。

输出
  results/08c_预测率曲线.csv        留出口径主表
  results/08c_命中面积对照.csv      两种口径并列（留出 / 全样本描述性）
  results/08c_计数核对.csv          记录数 vs 唯一像元数、同像元概率一致性
  results/08c_报告.md
  figures/fig_预测率曲线.png
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

LOG = os.path.join(L.LOG, "08c_prediction_rate.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 08c 预测率曲线（v3.1 · 逐折留出区域，V3-02）===", LOG)

MODEL = "XGBoost"
FACTORS = L.GROUPS["G2"]
KS = [1, 2, 5, 10, 15, 20, 30, 40, 50]

# ---------------- 数据 ----------------
df = L.load_base()
y = df["landslide"].astype(int).values
folds = df["fold"].values

dom_p = os.path.join(L.DATA, "prediction_full_valid.csv")
if not os.path.exists(dom_p):
    raise SystemExit("缺 data/prediction_full_valid.csv，请先运行 22_full_domain.py")
dom = pd.read_csv(dom_p)
L.log(f"完整域 {len(dom)} 像元；折分布 "
      f"{dom['fold'].value_counts().sort_index().to_dict()}", LOG)

# 每折参数与阈值（03 产出，不重新搜索）
par_p = os.path.join(L.RES, "03_最优超参数.csv")
if not os.path.exists(par_p):
    raise SystemExit("缺 results/03_最优超参数.csv，请先运行 03_main.py")
par = pd.read_csv(par_p)
par = par[(par["group"] == "G2") & (par["model"] == MODEL)].sort_values("fold")
if len(par) != 5:
    raise SystemExit(f"G2/{MODEL} 应有 5 折参数，实际 {len(par)}")
PER_FOLD = {int(r.fold): json.loads(r.params) for r in par.itertuples()}
THR = {int(r.fold): float(r.threshold) for r in par.itertuples()}
L.log(f"每折阈值：{THR}", LOG)

# ---------------- 逐折留出打分 ----------------
p_hold = np.full(len(dom), np.nan)
dom_fold = dom["fold"].values
for k in sorted(PER_FOLD):
    tr = np.where(folds != k)[0]
    te_cells = np.where(dom_fold == k)[0]
    if len(te_cells) == 0:
        raise SystemExit(f"第 {k} 折在完整域上没有像元")
    dtr = df.iloc[tr]
    Xtr, enc = L.build_matrix(dtr, FACTORS, fit=True)
    Xte, _ = L.build_matrix(dom.iloc[te_cells], FACTORS, enc=enc, fit=False)
    mdl = L.instantiate(MODEL, PER_FOLD[k], 42, n_jobs=1)
    mdl.fit(Xtr, y[tr])
    p_hold[te_cells] = mdl.predict_proba(Xte)[:, 1]
    L.log(f"折 {k}：留出像元 {len(te_cells)}，参数 {PER_FOLD[k]}", LOG)

if not np.isfinite(p_hold).all():
    raise SystemExit(f"留出打分有 {int((~np.isfinite(p_hold)).sum())} 个像元缺失")

# 自检：用每折参数重拟合出的概率，在 702 个调查点所在像元上应与主链 OOF 一致。
# 若不一致，说明本脚本的折内预处理/参数复现与 03 不同，后续曲线不可用。
_oo = os.path.join(L.RES, "03_OOF预测.csv")
_consist = None
if os.path.exists(_oo):
    _o = pd.read_csv(_oo)
    _o = _o[(_o["group"] == "G2") & (_o["model"] == MODEL)]
    if len(_o):
        _b = L.load_base()[["pointid", "row", "col"]]
        _m = _o[["pointid", "oof"]].merge(_b, on="pointid", how="left")
        _m["cell_id"] = _m["row"].astype(str) + "_" + _m["col"].astype(str)
        _m["p_hold"] = _m["cell_id"].map(dict(zip(dom["cell_id"], p_hold)))
        _d = float(np.abs(_m["oof"].values - _m["p_hold"].values).max())
        _consist = _d
        L.log(f"自检：与主链 OOF 在 702 点上的最大差 = {_d:.3e}", LOG)
        if _d > 1e-9:
            raise SystemExit(f"[失败] 08c 的折内复现与 03 不一致，最大差 {_d:.3e}")

# ---------------- 全样本最终模型（描述性对照）----------------
final_p = os.path.join(L.DATA, "全区预测概率_完整域.csv")
if os.path.exists(final_p):
    fp = pd.read_csv(final_p)[["pointid", "row", "col", f"p_{MODEL}"]]
    dom = dom.merge(fp, on=["pointid", "row", "col"], how="left")
    if dom[f"p_{MODEL}"].isna().any():
        raise SystemExit("最终模型概率与完整域未对齐")
    p_final = dom[f"p_{MODEL}"].values
else:
    p_final = None
    L.log("未找到完整域最终模型概率，仅输出留出口径", LOG)

dom["p_holdout"] = p_hold
dom[["pointid", "row", "col", "cell_id", "fold", "p_holdout"]].to_csv(
    os.path.join(L.DATA, "完整域空间留出概率.csv"), index=False, encoding="utf-8-sig")
if p_final is not None:
    dom[f"p_final_{MODEL}"] = p_final

# ---------------- 调查点归属（按像元，不按记录覆盖）----------------
cells_all = set(dom["cell_id"])
la = df[df["landslide"] == 1].copy()
bg = df[df["landslide"] == 0].copy()
la["cell_id"] = la["row"].astype(str) + "_" + la["col"].astype(str)
bg["cell_id"] = bg["row"].astype(str) + "_" + bg["col"].astype(str)
la["in_dom"] = la["cell_id"].isin(cells_all)
bg["in_dom"] = bg["cell_id"].isin(cells_all)

n_la_rec, n_la_cell = len(la), la["cell_id"].nunique()
n_la_rec_in = int(la["in_dom"].sum())
n_la_cell_in = la.loc[la["in_dom"], "cell_id"].nunique()
n_bg_rec_in = int(bg["in_dom"].sum())
n_bg_cell_in = bg.loc[bg["in_dom"], "cell_id"].nunique()

# 留出分数并入调查点（用于命中/误报）
cell2p = dict(zip(dom["cell_id"], dom["p_holdout"]))
la["p_holdout"] = la["cell_id"].map(cell2p)
bg["p_holdout"] = bg["cell_id"].map(cell2p)
if p_final is not None:
    cell2pf = dict(zip(dom["cell_id"], dom[f"p_final_{MODEL}"]))
    la["p_final"] = la["cell_id"].map(cell2pf)
    bg["p_final"] = bg["cell_id"].map(cell2pf)

# 同像元概率一致性：同一 cell 内多条记录必须得到同一像元分数
dup = la.groupby("cell_id")["p_holdout"].nunique()
n_inconsistent = int((dup > 1).sum())
L.log(f"同像元多记录：滑坡 {n_la_rec} 条 / {n_la_cell} 像元；"
      f"背景 {len(bg)} 条 / {bg['cell_id'].nunique()} 像元；"
      f"同像元分数不一致的像元 {n_inconsistent}", LOG)

# ---------------- 曲线 ----------------
def curve(p_cells, p_la_cell, p_bg_cell, tag):
    """p_*_cell 为按唯一像元聚合后的概率（同一像元只算一次）。"""
    rows = []
    for k in KS:
        q = float(np.percentile(p_cells, 100 - k))
        sel = p_cells >= q
        area = float(sel.mean())
        hit = float((p_la_cell >= q).mean()) if len(p_la_cell) else np.nan
        fp = float((p_bg_cell >= q).mean()) if len(p_bg_cell) else np.nan
        rows.append({"前k%": k, "阈值q": q, "面积占比": area,
                     "滑坡像元数": int(len(p_la_cell)), "命中率": hit,
                     "背景像元数": int(len(p_bg_cell)), "背景误报率": fp,
                     "命中/面积比": hit / max(area, 1e-9), "口径": tag})
    return pd.DataFrame(rows)


p_cells = p_hold
la_cell = la.loc[la["in_dom"]].groupby("cell_id")["p_holdout"].first().values
bg_cell = bg.loc[bg["in_dom"]].groupby("cell_id")["p_holdout"].first().values
PR = curve(p_cells, la_cell, bg_cell, f"逐折留出（每像元由未见过其块的模型打分，n={len(p_cells)}）")
PR.to_csv(os.path.join(L.RES, "08c_预测率曲线.csv"), index=False,
          encoding="utf-8-sig")
L.log("留出口径：\n" + PR.round(4).to_string(index=False), LOG)

PR2 = None
if p_final is not None:
    la_cell_f = la.loc[la["in_dom"]].groupby("cell_id")["p_final"].first().values
    bg_cell_f = bg.loc[bg["in_dom"]].groupby("cell_id")["p_final"].first().values
    PR2 = curve(p_final, la_cell_f, bg_cell_f,
                "全样本最终模型地图（描述性拟合结果，不能当留出精度）")
    PR2.to_csv(os.path.join(L.RES, "08c_全样本地图曲线.csv"), index=False,
               encoding="utf-8-sig")

CMP = PR[["前k%", "面积占比", "命中率", "背景误报率"]].rename(
    columns={"命中率": "命中率_留出", "背景误报率": "背景误报率_留出"})
if PR2 is not None:
    CMP = CMP.merge(PR2[["前k%", "命中率", "背景误报率"]].rename(
        columns={"命中率": "命中率_全样本", "背景误报率": "背景误报率_全样本"}),
        on="前k%", how="left")
CMP.to_csv(os.path.join(L.RES, "08c_命中面积对照.csv"), index=False,
           encoding="utf-8-sig")

CNT = pd.DataFrame([
    {"项": "滑坡记录数", "值": n_la_rec},
    {"项": "滑坡唯一像元数", "值": n_la_cell},
    {"项": "滑坡记录落在完整域", "值": n_la_rec_in},
    {"项": "滑坡唯一像元落在完整域", "值": n_la_cell_in},
    {"项": "背景记录数", "值": len(bg)},
    {"项": "背景唯一像元数", "值": bg["cell_id"].nunique()},
    {"项": "背景记录落在完整域", "值": n_bg_rec_in},
    {"项": "背景唯一像元落在完整域", "值": n_bg_cell_in},
    {"项": "同像元分数不一致的像元数", "值": n_inconsistent},
    {"项": "完整域像元数", "值": len(dom)},
    {"项": "曲线计数单位", "值": "唯一像元（同像元的多条记录只计一次）"},
    {"项": "与主链 OOF 在 702 点上的最大差（自检）",
     "值": (f"{_consist:.3e}" if _consist is not None else "未取到 03_OOF预测.csv")},
])
CNT.to_csv(os.path.join(L.RES, "08c_计数核对.csv"), index=False,
           encoding="utf-8-sig")

# ---------------- 图 ----------------
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=220)
ax.plot(PR["面积占比"] * 100, PR["命中率"] * 100, "o-", color="#c0392b",
        label="Held-out (per-fold model on its own held-out blocks)")
if PR2 is not None:
    ax.plot(PR2["面积占比"] * 100, PR2["命中率"] * 100, "s--", color="#2c6fbb",
            label="Full-sample model map (descriptive fit, not held-out)")
ax.plot([0, 100], [0, 100], "k:", lw=1, label="Random")
ax.set_xlabel("Area fraction of full valid domain (%)")
ax.set_ylabel("Landslide cells captured (%)")
ax.set_title(f"{MODEL} — prediction-rate curve on full valid domain "
             f"({len(dom)} cells, {n_la_cell} landslide cells)", fontsize=10)
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_预测率曲线.png"), dpi=220)
plt.close(fig)

# ---------------- 报告 ----------------
rep = [
    "# 08c · 预测率曲线（v3.1 重写 · V3-02 修复）\n",
    "> 旧版问题：把每折 OOF 概率逐行写入域表后求全域面积阈值，"
    "导致面积阈值、区域概率与滑坡概率来自不同训练范围；且同像元多记录会后者覆盖前者。"
    "本版改为**逐折模型给该折留出区域全部有效像元打分**。\n",
    "## 一、口径定义\n",
    f"- 预测域：`prediction_full_valid.csv`，{len(dom)} 个共同有效像元；",
    f"- 打分：第 k 折模型（参数与阈值取自 `03_最优超参数.csv`，不重新搜索）"
    f"给 `fold == k` 的全部有效像元打分；每个像元都由**未见过其所在空间块**的模型打分；",
    "- 面积占比：以全部有效像元为分母，取分数最高的前 k%；",
    "- 命中率：完整域内**滑坡唯一像元**中落入前 k% 的比例；",
    "- 背景误报率：完整域内**背景唯一像元**中落入前 k% 的比例"
    "（1:1 采样背景，不代表全域误报面积比例）。\n",
    "## 二、计数核对（记录 vs 像元）\n",
    CNT.to_markdown(index=False), "",
    "## 三、留出口径曲线（主表）\n",
    PR.round(4).to_markdown(index=False), "",
    "## 四、两种口径并列\n",
    CMP.round(4).to_markdown(index=False), "",
    "> 全样本地图曲线是**拟合结果**，只能说明「模型在全样本上的表现」；"
    "不能当作独立预测精度，也不能与留出曲线混为一条。\n",
    "## 五、使用边界\n",
    "- 不同折模型的分数未做跨折校准，拼图后只作为**汇总排序诊断**；各折分数尺度差异仍是其局限，"
    "不要把某个绝对概率值当校准概率解读；",
    "- 留出图在训练点所在像元也输出分数（正常推断行为），"
    "因此命中率反映的是「空间留出下的排序能力」，不是「对未知区域的野外精度」；",
    "- 若需报告面积型指标，请以本表面积占比（基于完整域 54,057 像元）为准。", "",
]
with open(os.path.join(L.RES, "08c_报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in rep))
L.log("=== 08c 完成 ===", LOG)
print(CMP.round(4).to_string(index=False))
