# -*- coding: utf-8 -*-
"""
09_shap_map.py — 因子重要性、全区制图与候选图（v2）
=================================================
v2 修正（依据核查清单）：
  C19  permutation importance 改为在**外层测试折**上计算（v1 在训练样本上置换，
       模型见过这些样本，重要性被系统性低估且不独立）；
       且按**原始因子整体**置换（aspect → sin/cos/flat 同步置换；
       land_use → 全部 one-hot 列同步置换），而不是逐派生列独立置换。
  C20  主模型与主展示图**固定为 XGBoost G2**，不再按 AUC 或与集成平均相关性
       更换展示模型；其余模型图仅作候选附加输出。
  C02  所有概率、面积、重叠率统计统一在 53,911 点预测域内。

输出： results/09_重要性.csv（XGBoost 5 折 × 15 原始因子）,
       results/09_候选图统计.csv, 09_G2G3差异.csv,
       data/全区预测概率.csv, figures/fig_易发性图_XGBoost.png 等
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

LOG = os.path.join(L.LOG, "09_shap_map.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 09 重要性 / 全区制图（v2）===", LOG)

df = L.load_base()
pred = L.load_prediction()
y = df["landslide"].astype(int).values
folds = df["fold"].values
factors = L.GROUPS["G2"]

# 主模型固定 XGBoost（C20）
PRIMARY = "XGBoost"
L.log(f"主模型/主图固定为 {PRIMARY}（C20，不按排名更换）", LOG)

bp = pd.read_csv(os.path.join(L.RES, "03_最优超参数.csv"))
bp = bp[bp["group"] == "G2"]
PER_FOLD = {}
for m in bp["model"].unique():
    PER_FOLD[m] = {int(r["fold"]): json.loads(r["params"])
                   for _, r in bp[bp["model"] == m].iterrows()}

# ------------------------------------------------------------
# 1. 原始因子 → 矩阵列 的映射（用于整体置换）
# ------------------------------------------------------------
def fit_matrix(d, m):
    native = m in ("CatBoost", "LightGBM")
    if native:
        return L.matrix_for_categorical_model(d, factors, fit=True)
    return L.build_matrix(d, factors, fit=True)


def factor_col_map(columns):
    mp = {}
    for f in factors:
        if f == "aspect":
            mp[f] = [c for c in ["aspect_sin", "aspect_cos", "aspect_flat"]
                     if c in columns]
        elif f == "land_use":
            mp[f] = [c for c in columns if c.startswith("lu_")]
            if not mp[f] and "land_use_code" in columns:
                mp[f] = ["land_use_code"]
        else:
            mp[f] = [f] if f in columns else []
    return {k: v for k, v in mp.items() if v}


# ------------------------------------------------------------
# 2. 外层测试折上的分组置换重要性（C19）
# ------------------------------------------------------------
L.log("--- 分层置换重要性（外层测试折，原始因子整体置换）---", LOG)
N_REPEAT = 20
imp_rows = []
t0 = time.time()
for k in range(5):
    tr = np.where(folds != k)[0]
    te = np.where(folds == k)[0]
    dtr, dte = df.iloc[tr], df.iloc[te]
    ytr, yte = y[tr], y[te]
    params = PER_FOLD[PRIMARY][k]
    native = PRIMARY in ("CatBoost", "LightGBM")
    if native:
        Xtr, enc = L.matrix_for_categorical_model(dtr, factors, fit=True)
        Xte, _ = L.matrix_for_categorical_model(dte, factors, enc=enc, fit=False)
        ci = [list(Xtr.columns).index(c) for c in enc.get("cat_cols", [])]
        fk = ({"cat_features": ci} if (PRIMARY == "CatBoost" and ci)
              else ({"categorical_feature": enc.get("cat_cols", [])} if ci else {}))
    else:
        Xtr, enc = L.build_matrix(dtr, factors, fit=True)
        Xte, _ = L.build_matrix(dte, factors, enc=enc, fit=False)
        fk = {}
    model = L.instantiate(PRIMARY, params, 42, n_jobs=1)
    model.fit(Xtr, ytr, **fk)
    base_auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    cmap = factor_col_map(list(Xte.columns))
    rng = np.random.default_rng(1000 + k)
    for f, cols in cmap.items():
        idxs = [list(Xte.columns).index(c) for c in cols]
        drops = []
        for r in range(N_REPEAT):
            Xp = Xte.copy()
            perm = rng.permutation(len(Xp))
            vals = Xp.iloc[perm, idxs].values
            Xp.iloc[:, idxs] = vals
            drops.append(base_auc - roc_auc_score(yte, model.predict_proba(Xp)[:, 1]))
        imp_rows.append({"model": PRIMARY, "fold": k, "factor": f,
                         "n_repeat": N_REPEAT,
                         "importance": float(np.mean(drops)),
                         "std": float(np.std(drops, ddof=1)),
                         "base_auc": base_auc})
    L.log(f"  折 {k} 重要性完成（base AUC={base_auc:.4f}）", LOG)

IMP = pd.DataFrame(imp_rows)
IMP.to_csv(os.path.join(L.RES, "09_重要性.csv"), index=False, encoding="utf-8-sig")
IMP_SUM = IMP.groupby("factor").agg(
    importance_mean=("importance", "mean"),
    importance_std_across_folds=("importance", "std"),
    n_folds=("fold", "count")).sort_values("importance_mean", ascending=False)
IMP_SUM.to_csv(os.path.join(L.RES, "09_重要性汇总.csv"), encoding="utf-8-sig")
L.log("\n重要性（外层测试折，5 折均值）:\n" +
      IMP_SUM.round(4).to_string(), LOG)

# ------------------------------------------------------------
# 3. 全区分级预测（主模型 XGBoost，全 702 训练）
# ------------------------------------------------------------
L.log("--- 全区预测 ---", LOG)
with rasterio.open(os.path.join(L.RASTERS, "dem.tif")) as s:
    dem = s.read(1).astype(float)
    dem[dem == s.nodata] = np.nan
    H, W = dem.shape
    b = s.bounds

CANDIDATES = ["XGBoost", "RF", "CatBoost", "LogReg", "MLP", "SVM"]
prob_df = pred[["pointid", "row", "col"]].copy()
grids = {}
MODELS_OBJ, ENCODERS, PARAMS_ALL = {}, {}, {}
for m in CANDIDATES:
    if m not in PER_FOLD:
        continue
    native = m in ("CatBoost", "LightGBM")
    # 全样本训练：参数取该模型 5 折中出现最多的组合（记录为配置）
    ps = pd.Series([json.dumps(v, sort_keys=True) for v in PER_FOLD[m].values()])
    params_all = json.loads(ps.value_counts().index[0])
    if native:
        Xa, enc = L.matrix_for_categorical_model(df, factors, fit=True)
        Xp, _ = L.matrix_for_categorical_model(pred, factors, enc=enc, fit=False)
        ci = [list(Xa.columns).index(c) for c in enc.get("cat_cols", [])]
        fk = ({"cat_features": ci} if (m == "CatBoost" and ci)
              else ({"categorical_feature": enc.get("cat_cols", [])} if ci else {}))
    else:
        Xa, enc = L.build_matrix(df, factors, fit=True)
        Xp, _ = L.build_matrix(pred, factors, enc=enc, fit=False)
        fk = {}
    mdl = L.instantiate(m, params_all, 42, n_jobs=1)
    mdl.fit(Xa, y, **fk)
    MODELS_OBJ[m] = mdl
    ENCODERS[m] = enc
    PARAMS_ALL[m] = params_all
    p = mdl.predict_proba(Xp)[:, 1]
    prob_df[f"p_{m}"] = p
    g = np.full((H, W), np.nan)
    g[pred["row"].values, pred["col"].values] = p
    grids[m] = g
prob_df.to_csv(os.path.join(L.DATA, "全区预测概率.csv"), index=False,
               encoding="utf-8-sig")

# R08：保存最终模型与编码器工件，使主图可被独立重建
import joblib
ART = os.path.join(L.EXP, "artifacts")
os.makedirs(ART, exist_ok=True)
FINAL_CFG = {}
for m in grids:
    joblib.dump({"model": MODELS_OBJ[m], "encoder": ENCODERS[m]},
                os.path.join(ART, f"final_{m}.joblib"))
    FINAL_CFG[m] = {"params": PARAMS_ALL[m],
                    "encoder_cols": list(ENCODERS[m].get("matrix_cols", [])),
                    "matrix_sha256": ENCODERS[m].get("matrix_sha256"),
                    "numeric_spec": ENCODERS[m].get("numeric_spec")}
L.save_json({"primary": PRIMARY, "final_config": FINAL_CFG,
             "factors": factors,
             "train_n": int(len(df)), "domain_n": int(len(pred))},
            os.path.join(ART, "final_config.json"))
L.log(f"最终模型/编码器已保存：{ART}", LOG)

ens = np.nanmean(np.stack([grids[m] for m in grids]), axis=0)
rows = []
for m in grids:
    g = grids[m]
    both = np.isfinite(g) & np.isfinite(ens)
    v = g[np.isfinite(g)]
    rows.append({"候选图": m, "预测均值": float(v.mean()),
                 "预测标准差": float(v.std()),
                 "极高区(P>0.8)面积占比": float(np.mean(v > 0.8)),
                 "与集成平均相关": float(np.corrcoef(g[both], ens[both])[0, 1]),
                 "与集成平均MAD": float(np.mean(np.abs(g[both] - ens[both]))),
                 "是否正文主图": m == PRIMARY})
CAND = pd.DataFrame(rows).sort_values("与集成平均相关", ascending=False)
CAND.to_csv(os.path.join(L.RES, "09_候选图统计.csv"), index=False,
            encoding="utf-8-sig")
L.log("\n候选图统计:\n" + CAND.round(4).to_string(index=False), LOG)
L.log(f"正文主图 = {PRIMARY}（C20 固定，不按排名更换）", LOG)

# ------------------------------------------------------------
# 4. G2 vs G3 空间差异（同一模型 XGBoost）
# ------------------------------------------------------------
L.log("--- G2 vs G3 空间差异（XGBoost）---", LOG)
bp3 = pd.read_csv(os.path.join(L.RES, "03_最优超参数.csv"))
bp3 = bp3[(bp3["group"] == "G3")]
diff_rows = []
if len(bp3) and "XGBoost" in set(bp3["model"]):
    f3 = L.GROUPS["G3"]
    d3 = bp3[bp3["model"] == "XGBoost"]
    p3 = json.loads(pd.Series([json.dumps(json.loads(x), sort_keys=True)
                               for x in d3["params"]]).value_counts().index[0])
    Xa3, enc3 = L.build_matrix(df, f3, fit=True)
    Xp3, _ = L.build_matrix(pred, f3, enc=enc3, fit=False)
    m3 = L.instantiate("XGBoost", p3, 42, n_jobs=1)
    m3.fit(Xa3, y)
    prob_g3 = m3.predict_proba(Xp3)[:, 1]
    prob_df["p_XGBoost_G3"] = prob_g3
    g3 = np.full((H, W), np.nan)
    g3[pred["row"].values, pred["col"].values] = prob_g3
    prob_df.to_csv(os.path.join(L.DATA, "全区预测概率.csv"), index=False,
                   encoding="utf-8-sig")
    g2 = grids["XGBoost"]
    both = np.isfinite(g2) & np.isfinite(g3)
    r = float(np.corrcoef(g2[both], g3[both])[0, 1])
    mad = float(np.mean(np.abs(g2[both] - g3[both])))
    # IoU（C20/R07）：必须各图用**自身** q80 定义前 20%，否则阈值不可比；
    # 共同阈值版本仅作对照并列报出。
    q1 = np.percentile(g2[both], 80)
    q2 = np.percentile(g3[both], 80)
    a_, b_ = g2[both] >= q1, g3[both] >= q2
    iou_top20 = float((a_ & b_).sum() / max((a_ | b_).sum(), 1))
    a_c, b_c = g2[both] >= q1, g3[both] >= q1
    iou_common = float((a_c & b_c).sum() / max((a_c | b_c).sum(), 1))
    fixed_thr = 0.5
    a2, b2 = g2[both] >= fixed_thr, g3[both] >= fixed_thr
    iou_fix = float((a2 & b2).sum() / max((a2 | b2).sum(), 1))
    diff_rows.append({"model": "XGBoost", "空间相关_r": r, "平均绝对差_MAD": mad,
                      "IoU_独立q80": iou_top20,
                      "IoU_共同q80": iou_common,
                      "IoU_固定阈值0.5": iou_fix,
                      "q80_G2": float(q1), "q80_G3": float(q2),
                      "交并集计数_前20%": f"{int((a_ & b_).sum())}/{int((a_ | b_).sum())}",
                      "交并集计数_固定0.5": f"{int((a2 & b2).sum())}/{int((a2 | b2).sum())}"})
    DIFF = pd.DataFrame(diff_rows)
    DIFF.to_csv(os.path.join(L.RES, "09_G2G3差异.csv"), index=False,
                encoding="utf-8-sig")
    L.log("\n" + DIFF.round(4).to_string(index=False), LOG)
else:
    DIFF = pd.DataFrame()
    L.log("[跳过] 尚无 G3 的 XGBoost 结果", LOG)

# ------------------------------------------------------------
# 5. 图件
# ------------------------------------------------------------
def plot_map(g, ax, title, pts=None):
    im = ax.imshow(g, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest",
                   extent=[b.left, b.right, b.bottom, b.top])
    ax.set_title(title, fontsize=10)
    if pts is not None:
        ax.scatter(pts["X"], pts["Y"], c="white", s=5, marker="o",
                   edgecolors="black", linewidths=0.3, zorder=5)
    ax.set_xlabel("Easting (km)", fontsize=8)
    ax.set_ylabel("Northing (km)", fontsize=8)
    return im


la = df[df["landslide"] == 1]
# 主图：XGBoost
fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=300)
im = plot_map(grids[PRIMARY], ax, f"Landslide susceptibility map — {PRIMARY} (G2)",
              pts=la)
ticks_x = np.linspace(b.left, b.right, 4)
ticks_y = np.linspace(b.bottom, b.top, 4)
ax.set_xticks(ticks_x); ax.set_yticks(ticks_y)
ax.set_xticklabels([f"{x/1000:,.0f}" for x in ticks_x], fontsize=8)
ax.set_yticklabels([f"{y/1000:,.0f}" for y in ticks_y], fontsize=8)
ax.set_xlabel("Easting (km, UTM 49N)", fontsize=9)
ax.set_ylabel("Northing (km, UTM 49N)", fontsize=9)
cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Susceptibility probability", fontsize=9)
cb.ax.tick_params(labelsize=8)
ax.scatter([], [], c="white", edgecolors="black", s=12, marker="o",
           label=f"Landslide inventory (n={len(la)})")
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, f"fig_易发性图_{PRIMARY}.png"), dpi=300,
            bbox_inches="tight")
plt.close(fig)

# 候选图（附加）
for m in grids:
    if m == PRIMARY:
        continue
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    im = plot_map(grids[m], ax, f"Candidate map — {m} (G2, supplementary)", pts=la)
    plt.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(L.FIG, f"fig_易发性图_{m}.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)

# 五套候选图
fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=150)
for i, m in enumerate(list(grids)):
    plot_map(grids[m], axes.ravel()[i], m, pts=la)
for j in range(len(grids), 6):
    axes.ravel()[j].axis("off")
fig.suptitle("Candidate susceptibility maps (all trained on 702 records; "
             "main figure fixed to XGBoost)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_五套候选易发性图.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# 重要性图
fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
d = IMP_SUM.sort_values("importance_mean")
ax.barh(d.index, d["importance_mean"],
        xerr=d["importance_std_across_folds"], color="#2f7fbf", alpha=0.85,
        error_kw={"elinewidth": 0.8, "capsize": 2})
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Drop in ROC-AUC after permuting the factor (outer test folds)",
              fontsize=9)
ax.set_title(f"Permutation importance — {PRIMARY} (G2, 5 folds × {N_REPEAT} repeats)",
             fontsize=10)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_重要性对比.png"), dpi=200,
            bbox_inches="tight")
plt.close(fig)

rep = ["# 09 因子重要性与全区制图（v2）\n",
       "## v2 修正", "",
       "- C19：置换重要性在**外层测试折**上计算，按原始因子整体置换"
       "（aspect 的 sin/cos/flat 与 land_use 的全部 one-hot 列同步置换）。",
       "- C20：主模型与主图**固定为 XGBoost G2**，不按 AUC 或相关性更换。",
       "- C02：全部空间统计在 53,911 点预测域内。", "",
       f"## 一、置换重要性（{PRIMARY}，5 折均值）", "",
       IMP_SUM.round(4).to_markdown(), "",
       "## 二、候选图统计", "", CAND.round(4).to_markdown(index=False), ""]
if len(DIFF):
    rep += ["## 三、G2 vs G3 空间差异（XGBoost）", "",
            DIFF.round(4).to_markdown(index=False), ""]
with open(os.path.join(L.RES, "09_制图报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in rep))
L.log("=== 09 完成 ===", LOG)
