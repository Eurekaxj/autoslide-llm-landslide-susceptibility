# -*- coding: utf-8 -*-
"""
02_coverage.py — 参数覆盖与外推风险 / 适用域（v2）
================================================
v2 修正（依据核查清单 C21、C02）：
  - 适用域分量分开统计并保存组合遮罩：① 13 个连续因子的样本区间外推；
    ② land_use 中样本未出现的类别（3、6 类共 29 像元）；③ 坡向平地支持；
    v1 只统计了连续因子，把未知类别误当作"已支持"。
  - 分母固定为 53,911 点预测域（不含作者已剔除的 146 像元）。
  - 补「以训练环境确定阈值的标准化多变量距离」（马氏距离，阈值取训练集自身 P95/P99）。
  - 逐像元输出各分量与组合标记，可复算。

输出： results/02_参数覆盖统计.csv, 02_土地利用类别覆盖.csv, 02_适用域分量.csv,
       results/02_外推风险统计.csv, 02_逐因子外推比例.csv,
       figures/fig_外推风险图.png, fig_因子覆盖对比.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

LOG = os.path.join(L.LOG, "02_coverage.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 02 参数覆盖 / 外推风险 / 适用域（v2）===", LOG)

df = L.load_base()
# 【V3-01】适用域分母改为**完整共同有效域**（54,057 像元）。
# 旧域（53,911）保留作对照：它缺的 146 个像元集中在滑坡附近，
# 用旧域当分母会系统性低估/错位适用域比例。
_full = os.path.join(L.DATA, "prediction_full_valid.csv")
pred = pd.read_csv(_full) if os.path.exists(_full) else L.load_prediction()
DOMAIN_TAG = "完整共同有效域" if os.path.exists(_full) else "旧域"
N_DOMAIN = len(pred)
_old_p = os.path.join(L.DATA, "prediction_base.csv")
_n_old = len(pd.read_csv(_old_p)) if os.path.exists(_old_p) else None
L.log(f"适用域分母 = {DOMAIN_TAG}，{N_DOMAIN} 点"
      + (f"（旧域 {_n_old} 点；本版补全 {N_DOMAIN - _n_old} 个）" if _n_old else ""), LOG)

CONT = ["DEM", "slope", "plan", "profile", "NDVI", "rx5day", "raincv",
        "roughness", "road", "river", "soil_clay", "soil_silt"]
Q = [1, 5, 25, 50, 75, 95, 99]

# ------------------------------------------------------------
# 1. 连续因子覆盖
# ------------------------------------------------------------
rows, ext_flags = [], {}
for f in CONT:
    r = pred[f].values.astype(float)
    s = df[f].values.astype(float)
    la = df.loc[df["landslide"] == 1, f].values.astype(float)
    nl = df.loc[df["landslide"] == 0, f].values.astype(float)
    rq, sq = np.percentile(r, Q), np.percentile(s, Q)
    lo, hi = s.min(), s.max()
    # 【V3-01b】"等于极值"的判定对数值精度极敏感：旧 prediction_base.csv 存的是
    # 未规范化的 float64，与新域的 float32-canonical 值在 1e-6 量级有差异，
    # 直接用 < / > 会让恰好落在极值上的像元在两种表示间翻转（实测 147 个）。
    # 因此区间判定加一个 float32 相对容差，容差内视为"在样本范围内"。
    tol = max(abs(lo), abs(hi), 1.0) * float(np.finfo(np.float32).eps) * 4
    inside = (r >= lo - tol) & (r <= hi + tol)
    within = float(np.mean(inside))
    row = {"因子": f}
    for q, v in zip(Q, rq):
        row[f"全区P{q}"] = v
    for q, v in zip(Q, sq):
        row[f"样本P{q}"] = v
    row.update({"全区最小": r.min(), "全区最大": r.max(),
                "样本最小": lo, "样本最大": hi,
                "滑坡组均值": la.mean(), "非滑坡组均值": nl.mean(),
                "滑坡组范围": f"[{la.min():.4g}, {la.max():.4g}]",
                "非滑坡组范围": f"[{nl.min():.4g}, {nl.max():.4g}]",
                "样本覆盖全区像元比例": within,
                "低于样本下限比例": float(np.mean(r < lo - tol)),
                "高于样本上限比例": float(np.mean(r > hi + tol)),
                "外推像元比例": 1 - within})
    rows.append(row)
    ext_flags[f] = ~inside
cov = pd.DataFrame(rows)
cov.to_csv(os.path.join(L.RES, "02_参数覆盖统计.csv"), index=False,
           encoding="utf-8-sig")
L.log("\n" + cov[["因子", "样本最小", "样本最大", "全区最小", "全区最大",
                  "外推像元比例"]].to_string(index=False), LOG)

# ------------------------------------------------------------
# 2. 坡向（单独：平地支持 + 八方位）
# ------------------------------------------------------------
a, ap = df["aspect"].values.astype(float), pred["aspect"].values.astype(float)
train_has_flat = bool((a < 0).any())
pred_flat = ap < 0
asp = pd.DataFrame({
    "项目": ["训练样本坡向范围", "全区坡向范围", "训练集是否含平地(-1)",
             "样本平地占比", "全区平地像元数", "平地支持状态",
             "样本八方位覆盖数", "全区八方位覆盖数"],
    "值": [f"[{a.min():.2f}, {a.max():.2f}]", f"[{ap.min():.2f}, {ap.max():.2f}]",
           str(train_has_flat), f"{(a < 0).mean()*100:.3f}%", str(int(pred_flat.sum())),
           (f"已支持但代表性稀疏（训练集仅 {int((a < 0).sum())} 条平地记录）"
            if train_has_flat else "未支持（全区存在平地但训练集无平地样本）"),
           len(np.unique(np.digitize(a[a >= 0], [45, 90, 135, 180, 225, 270, 315]))),
           len(np.unique(np.digitize(ap[ap >= 0], [45, 90, 135, 180, 225, 270, 315])))],
})
asp.to_csv(os.path.join(L.RES, "02_坡向覆盖.csv"), index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 3. 土地利用类别
# ------------------------------------------------------------
lu_r = pred["land_use"].astype(int)
lu_s = df["land_use"].astype(int)
lu_tab = []
for c in sorted(set(lu_r.unique()) | set(lu_s.unique())):
    lu_tab.append({"类别": int(c), "全区像元数": int((lu_r == c).sum()),
                   "全区占比": float((lu_r == c).mean()),
                   "样本数": int((lu_s == c).sum()),
                   "滑坡点数": int(((lu_s == c) & (df["landslide"] == 1)).sum()),
                   "非滑坡点数": int(((lu_s == c) & (df["landslide"] == 0)).sum()),
                   "样本中是否出现": bool((lu_s == c).any())})
lu_df = pd.DataFrame(lu_tab)
lu_df.to_csv(os.path.join(L.RES, "02_土地利用类别覆盖.csv"), index=False,
             encoding="utf-8-sig")
L.log("\n土地利用类别覆盖:\n" + lu_df.to_string(index=False), LOG)
unknown_lu = ~lu_r.isin(set(lu_s.unique())).values
L.log(f"未知土地类别像元数: {int(unknown_lu.sum())}", LOG)

# ------------------------------------------------------------
# 4. 组合遮罩（C21）
# ------------------------------------------------------------
ext_cnt = np.zeros(N_DOMAIN, dtype=int)
for f in CONT:
    ext_cnt += ext_flags[f].astype(int)
any_ext = ext_cnt > 0

# 【v3 · R06】平地遮罩的语义必须与训练类别集合一致：
# 训练集确实有平地样本（aspect = -1，如 NL_0111），因此"平地"不是未见类别。
# 只有**训练集完全没有平地**时，pred_flat 才构成不支持。
flat_ids = df.loc[df["aspect"].astype(float) < 0, "pointid"].tolist()
unsupported_flat = pred_flat & (not train_has_flat)
L.log(f"[R06] 训练集平地记录 {len(flat_ids)} 条 {flat_ids}；"
      f"pred_flat={int(pred_flat.sum())}，计入不支持的平地={int(unsupported_flat.sum())}", LOG)

mask = pd.DataFrame({
    "pointid": pred["pointid"], "row": pred["row"], "col": pred["col"],
    "n_extrapolated_continuous": ext_cnt,
    "extrapolated_continuous": any_ext,
    "unknown_landuse": unknown_lu,
    "flat_aspect": pred_flat,
    "unsupported_flat": unsupported_flat,
})
mask["any_unsupported"] = (mask["extrapolated_continuous"]
                           | mask["unknown_landuse"] | mask["unsupported_flat"])
mask.to_csv(os.path.join(L.DATA, "适用域遮罩_完整域.csv"), index=False,
            encoding="utf-8-sig")

su = pd.DataFrame([
    {"分量": "连续因子外推（≥1 个）", "像元数": int(any_ext.sum()),
     "占预测域比例": float(any_ext.mean())},
    {"分量": "未知土地利用类别", "像元数": int(unknown_lu.sum()),
     "占预测域比例": float(unknown_lu.mean())},
    {"分量": "不支持平地（仅当训练集无平地时计入）",
     "像元数": int(unsupported_flat.sum()),
     "占预测域比例": float(unsupported_flat.mean())},
    {"分量": "参考：全区平地区像元（已被训练覆盖，不计入不支持）",
     "像元数": int(pred_flat.sum()),
     "占预测域比例": float(pred_flat.mean())},
    {"分量": "任一不支持（并集）", "像元数": int(mask["any_unsupported"].sum()),
     "占预测域比例": float(mask["any_unsupported"].mean())},
    {"分量": "完全支持", "像元数": int((~mask["any_unsupported"]).sum()),
     "占预测域比例": float((~mask["any_unsupported"]).mean())},
])
su.to_csv(os.path.join(L.RES, "02_适用域分量.csv"), index=False,
          encoding="utf-8-sig")
L.log("\n适用域分量:\n" + su.round(6).to_string(index=False), LOG)

# ------------------------------------------------------------
# 5. 标准化多变量距离（马氏距离，阈值由训练集自身确定）
# ------------------------------------------------------------
mu = df[CONT].mean().values
sd = df[CONT].std(ddof=0).replace(0, 1.0).values
Ztr = (df[CONT].values - mu) / sd
Zpr = (pred[CONT].values - mu) / sd
S = np.cov(Ztr, rowvar=False)
S = S + np.eye(len(CONT)) * 1e-8
Sinv = np.linalg.pinv(S)


def mahal(Z):
    return np.sqrt(np.einsum("ij,jk,ik->i", Z, Sinv, Z))


d_tr, d_pr = mahal(Ztr), mahal(Zpr)
thr95, thr99 = np.percentile(d_tr, [95, 99])
md = pd.DataFrame([
    {"项目": "训练集马氏距离中位数", "值": float(np.median(d_tr))},
    {"项目": "训练集 P95", "值": float(thr95)},
    {"项目": "训练集 P99", "值": float(thr99)},
    {"项目": "预测域超过训练 P95 的像元数", "值": int((d_pr > thr95).sum())},
    {"项目": "预测域超过训练 P95 的比例", "值": float((d_pr > thr95).mean())},
    {"项目": "预测域超过训练 P99 的像元数", "值": int((d_pr > thr99).sum())},
    {"项目": "预测域超过训练 P99 的比例", "值": float((d_pr > thr99).mean())},
    {"项目": "预测域马氏距离中位数", "值": float(np.median(d_pr))},
])
md.to_csv(os.path.join(L.RES, "02_多变量距离.csv"), index=False, encoding="utf-8-sig")
mask["mahal_dist"] = d_pr
mask["mahal_beyond_p95"] = d_pr > thr95
mask.to_csv(os.path.join(L.DATA, "适用域遮罩_完整域.csv"), index=False,
            encoding="utf-8-sig")
L.log("\n" + md.round(4).to_string(index=False), LOG)

# ------------------------------------------------------------
# 6. 外推风险分级与图
# ------------------------------------------------------------
ext_df = pd.DataFrame({"n_extrapolated": ext_cnt})
ext_df["risk"] = pd.cut(ext_df["n_extrapolated"], [-1, 0, 1, 2, 4, 99],
                        labels=["0 无外推", "1 个因子外推", "2 个因子外推",
                                "3-4 个因子外推", "≥5 个因子外推"])
stat = ext_df["risk"].value_counts().sort_index()
stat_df = stat.rename_axis("外推等级").reset_index(name="像元数")
stat_df["占预测域比例"] = stat_df["像元数"] / N_DOMAIN
stat_df.to_csv(os.path.join(L.RES, "02_外推风险统计.csv"), index=False,
               encoding="utf-8-sig")
ex = pd.DataFrame({"因子": CONT,
                   "外推像元比例": [float(ext_flags[f].mean()) for f in CONT]
                   }).sort_values("外推像元比例", ascending=False)
ex.to_csv(os.path.join(L.RES, "02_逐因子外推比例.csv"), index=False,
          encoding="utf-8-sig")

with rasterio.open(os.path.join(L.RASTERS, "dem.tif")) as s:
    dem = s.read(1).astype(float)
    dem[dem == s.nodata] = np.nan
grid = np.full(dem.shape, np.nan)
grid[pred["row"].values, pred["col"].values] = ext_cnt
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=max(ext_cnt.max(), 1),
               interpolation="nearest")
ax.set_title("Extrapolation risk (number of continuous factors outside "
             "the 702-sample range)", fontsize=10)
cb = plt.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("# extrapolated factors", fontsize=9)
ax.set_xlabel("Column"); ax.set_ylabel("Row")
ax.scatter(df.loc[df.landslide == 1, "col"], df.loc[df.landslide == 1, "row"],
           c="blue", s=2, label="Landslide")
ax.scatter(df.loc[df.landslide == 0, "col"], df.loc[df.landslide == 0, "row"],
           c="green", s=2, label="Non-landslide")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_外推风险图.png"), dpi=200)
plt.close(fig)

fig, axes = plt.subplots(4, 4, figsize=(14, 11), dpi=150)
for i, f in enumerate(CONT):
    ax = axes.ravel()[i]
    r, s = pred[f].values.astype(float), df[f].values.astype(float)
    lo_, hi_ = np.percentile(np.concatenate([r, s]), [0.5, 99.5])
    bins = np.linspace(lo_, hi_, 40)
    ax.hist(r, bins=bins, density=True, alpha=0.5, color="#7f8c9a", label="Whole area")
    ax.hist(s, bins=bins, density=True, alpha=0.6, color="#c0392b", label="702 samples")
    ax.set_title(f, fontsize=9); ax.tick_params(labelsize=6)
    if i == 0:
        ax.legend(fontsize=7)
for j in range(len(CONT), 16):
    axes.ravel()[j].axis("off")
fig.suptitle("Whole-area (53,911 cells) vs 702-sample factor distributions",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(L.FIG, "fig_因子覆盖对比.png"), dpi=150)
plt.close(fig)

rep = ["# 02 参数覆盖 / 外推风险 / 适用域（v2）\n",
       "## v2 修正（C21/C02）", "",
       "- 适用域分**三个分量**分别统计并保存组合遮罩（v1 只算了连续因子）；",
       "- 分母固定为 53,911 点预测域；",
       "- 增加以训练环境定阈值的**马氏距离**（训练集自身 P95/P99）。", "",
       "## 一、适用域分量", "", su.round(6).to_markdown(index=False), "",
       "## 二、多变量距离", "", md.round(4).to_markdown(index=False), "",
       "## 三、连续因子覆盖", "",
       cov[["因子", "样本最小", "样本最大", "全区最小", "全区最大",
            "外推像元比例"]].round(6).to_markdown(index=False), "",
       "## 四、土地利用类别", "", lu_df.to_markdown(index=False), "",
       "## 五、坡向", "", asp.to_markdown(index=False), "",
       "## 六、外推风险分级", "", stat_df.to_markdown(index=False), ""]
with open(os.path.join(L.RES, "02_覆盖报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in rep))
L.log("=== 02 完成 ===", LOG)
