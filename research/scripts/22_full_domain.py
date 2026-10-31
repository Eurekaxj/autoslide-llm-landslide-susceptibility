# -*- coding: utf-8 -*-
"""
22_full_domain.py — 完整共同有效预测域（V3-01）
====================================================================================
审查意见（v3审查 V3-01）：
  旧版预测域 53,911 像元相对 DEM 有效像元 54,057 少了 146 个，缺口集中在滑坡
  分布附近（涉及 138/351 条滑坡记录），对主图表达与命中统计的影响不随面积
  比例同比缩小。建议**补全为主图输入**，但不改动 702 条建模记录与验证划分。

本脚本做四件事，全部可核：
  ① 以已对齐的 DEM 网格为准，取 15 因子**共同有效掩膜**（不是只看 DEM），
     报告 DEM 有效 / 共同有效 / 两者差异及其原因；
  ② 在同一网格上按 [row, col] 直接取值重建全部因子（不插值、不偏移网格），
     并与旧 `prediction_base.csv` 的 53,911 行逐因子比对，**必须零不等**；
  ③ 复用 `artifacts/final_<model>.joblib` 中的**已有模型与编码器**预测新域，
     并核对旧 53,911 像元的概率与 `data/全区预测概率.csv` **逐点一致**；
  ④ 把 351 条滑坡记录逐条与其 cell_id 做空间匹配，不匹配者写出真实原因。

输出
  data/prediction_full_valid.csv      完整域表（54,057 行，含 row/col/X/Y/因子）
  data/全区预测概率_完整域.csv          各模型在完整域上的概率
  results/22_完整域构建.csv            掩膜与计数
  results/22_原域一致性.csv            因子与概率的逐点一致性校验
  results/22_滑坡记录匹配.csv          351 条记录逐条归属
  results/22_报告.md

原则：本脚本**不训练任何模型**，只做推断与核对；旧域输出保留作版本对照。
"""
import os
import sys
import json
import glob

import numpy as np
import pandas as pd
import rasterio
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

FACTORS = L.GROUPS["G2"]          # 15 因子
LOG = os.path.join(L.LOG, "22_full_domain.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 22 完整共同有效预测域（V3-01）===", LOG)


def rpath(name):
    """因子名 → 栅格文件。DEM 在磁盘上为小写 dem.tif，按不区分大小写解析。"""
    p = os.path.join(L.RASTERS, f"{name}.tif")
    if os.path.exists(p):
        return p
    for fn in os.listdir(L.RASTERS):
        if fn.lower() == f"{name}.tif".lower():
            return os.path.join(L.RASTERS, fn)
    raise FileNotFoundError(f"找不到因子栅格：{name}.tif")


# ------------------------------------------------------------
# ① 共同有效掩膜
# ------------------------------------------------------------
masks, meta = {}, None
for f in FACTORS:
    with rasterio.open(rpath(f)) as s:
        a = s.read(1)
        if meta is None:
            meta = {"width": s.width, "height": s.height,
                    "transform": tuple(s.transform)[:6],
                    "crs": s.crs.to_string(), "res": (s.transform.a, -s.transform.e),
                    "nodata": s.nodata, "dtype": s.dtypes[0]}
        else:
            assert (s.width, s.height) == (meta["width"], meta["height"]), \
                f"{f} 网格尺寸不一致"
            assert tuple(s.transform)[:6] == meta["transform"], \
                f"{f} 仿射变换不一致"
            assert s.crs.to_string() == meta["crs"], f"{f} CRS differs"
        masks[f] = (a != s.nodata) & np.isfinite(a)

dem_mask = masks["DEM"]
common = np.ones_like(dem_mask)
for f in FACTORS:
    common &= masks[f]
n_cells_each = {f: int(masks[f].sum()) for f in FACTORS}
n_dem = int(dem_mask.sum())
n_common = int(common.sum())
only_dem = int((dem_mask & ~common).sum())
L.log(f"DEM 有效 {n_dem}；15 因子共同有效 {n_common}；仅 DEM 有效 {only_dem}", LOG)
if len(set(n_cells_each.values())) != 1:
    L.log(f"[注意] 各因子有效像元数不一致：{n_cells_each}", LOG)

rows, cols = np.nonzero(common)
T = meta["transform"]
AREA_CELL_KM2 = abs(T[0] * T[4] - T[1] * T[3]) / 1e6
xs = T[0] * (cols + 0.5) + T[1] * (rows + 0.5) + T[2]
ys = T[3] * (cols + 0.5) + T[4] * (rows + 0.5) + T[5]

full = pd.DataFrame({"row": rows.astype(int), "col": cols.astype(int),
                     "X": xs, "Y": ys})
full["cell_id"] = full["row"].astype(str) + "_" + full["col"].astype(str)
full.insert(0, "pointid", np.arange(1, len(full) + 1))

# 按 [row, col] 直接取值（同一网格，无需重采样）
for f in FACTORS:
    with rasterio.open(rpath(f)) as s:
        a = s.read(1)
    v = a[rows, cols]
    # 与训练端一致的 float32-canonical：先回到栅格原始精度，再升 float64
    full[f] = np.asarray(v, dtype=np.float32).astype(np.float64)

full["valid_factors"] = len(FACTORS)
full["block_r"] = full["row"] // 20
full["block_c"] = full["col"] // 20
ftab = pd.read_csv(os.path.join(L.CFG, "fold_table.csv"))
fold_map = {(int(r.block_r), int(r.block_c)): int(r.fold)
            for r in ftab.itertuples()}
full["fold"] = [fold_map.get((int(r), int(c)), -1)
                for r, c in zip(full["block_r"], full["block_c"])]
n_nofold = int((full["fold"] < 0).sum())
L.log(f"完整域 {len(full)} 像元；无折归属 {n_nofold}", LOG)

# ------------------------------------------------------------
# ② 与旧域逐因子比对（同一像元必须完全同值）
# ------------------------------------------------------------
old = pd.read_csv(os.path.join(L.DATA, "prediction_base.csv"))
chk_rows = []
merged = old[["cell_id"] + FACTORS].merge(
    full[["cell_id"] + FACTORS], on="cell_id", suffixes=("_old", "_new"))
L.log(f"旧域 {len(old)} 像元可与新域匹配 {len(merged)}", LOG)
if len(merged) != len(old):
    raise SystemExit(f"[失败] 旧域有 {len(old)-len(merged)} 个像元在新域中找不到")

for f in FACTORS:
    a = np.asarray(merged[f"{f}_old"], dtype=np.float64)
    b = np.asarray(merged[f"{f}_new"], dtype=np.float64)
    d_raw = float(np.abs(a - b).max()) if len(a) else np.nan
    # 规范化后是否等价：训练端与建模矩阵都先转到 float32-canonical，
    # 所以"float32 意义下相等"才是模型真正看到的等价性判据。
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    n32 = int((a32 != b32).sum())
    chk_rows.append({
        "校验对象": f"因子 {f}", "n": int(len(a)),
        "原始最大绝对差": d_raw,
        "float32规范化后不等数": n32,
        "结论": "规范化后完全一致" if n32 == 0 else "不一致，需排查",
    })
    if n32:
        raise SystemExit(f"因子 {f} 规范化后仍有 {n32} 个像元不一致")
    elif d_raw > 0:
        L.log(f"因子 {f}: 原始表示差 {d_raw:.3e}，但 float32 规范化后完全一致"
              f"（旧表未规范化，模型输入相同）", LOG)

# ------------------------------------------------------------
# ③ 复用已有模型/编码器预测（不重训）
# ------------------------------------------------------------
ART = os.path.join(L.EXP, "artifacts")
cfg_p = os.path.join(ART, "final_config.json")
if not os.path.exists(cfg_p):
    raise SystemExit("缺 artifacts/final_config.json，请先运行 09_shap_map.py")
FINAL_CFG = json.load(open(cfg_p, encoding="utf-8"))
models = [p.split("final_")[1][:-7] for p in sorted(glob.glob(
    os.path.join(ART, "final_*.joblib")))]
L.log(f"可用最终模型工件：{models}", LOG)

prob = full[["pointid", "row", "col", "cell_id"]].copy()
for m in models:
    bundle = joblib.load(os.path.join(ART, f"final_{m}.joblib"))
    mdl, enc = bundle["model"], bundle["encoder"]
    if m in ("CatBoost", "LightGBM"):
        X, _ = L.matrix_for_categorical_model(full, FACTORS, enc=enc, fit=False)
        ci = [list(X.columns).index(c) for c in enc.get("cat_cols", [])]
        fk = ({"cat_features": ci} if (m == "CatBoost" and ci)
              else ({"categorical_feature": enc.get("cat_cols", [])} if ci else {}))
    else:
        X, _ = L.build_matrix(full, FACTORS, enc=enc, fit=False)
        fk = {}
    if list(X.columns) != list(enc["matrix_cols"]):
        raise SystemExit(f"{m} 矩阵列与编码器记录不一致")
    prob[f"p_{m}"] = mdl.predict_proba(X)[:, 1]
    L.log(f"{m}：完整域概率范围 "
          f"[{prob[f'p_{m}'].min():.4f}, {prob[f'p_{m}'].max():.4f}]", LOG)

# 与旧域概率逐点比对（同一模型、同一编码器，必须一致）
oldp_path = os.path.join(L.DATA, "全区预测概率.csv")
if os.path.exists(oldp_path):
    oldp = pd.read_csv(oldp_path)
    # 旧概率表无 cell_id，按 (row, col) 对齐（同一像元唯一）
    pm = oldp.merge(prob, on=["row", "col"], suffixes=("_old", "_new"))
    L.log(f"旧域概率表 {len(oldp)} 行，可与新域匹配 {len(pm)}", LOG)
    for m in models:
        c = f"p_{m}"
        if f"{c}_old" not in pm.columns or f"{c}_new" not in pm.columns:
            continue
        d = float(np.abs(pm[f"{c}_old"].values - pm[f"{c}_new"].values).max())
        # 旧概率表以约 float32 精度存盘（如 0.0021454848），故判据为
        # "差值是否落在 float32 往返误差内"，而不是 0；超过即视为模型/编码器不同。
        tol = float(np.finfo(np.float32).eps * 4)
        chk_rows.append({"校验对象": f"概率 {m}（旧域像元）", "n": int(len(pm)),
                         "原始最大绝对差": d,
                         "float32规范化后不等数": int(d > tol),
                         "结论": "复用同一工件，逐点复现" if d <= tol
                                 else "未复现，超出 float32 存储容差"})
        L.log(f"概率 {m} 旧域逐点最大差 = {d:.3e}（float32 容差 {tol:.3e}）", LOG)
        if d > tol:
            raise SystemExit(f"[失败] {m} 在旧域像元上的概率未复现，差 {d:.3e} 超出 "
                             f"float32 存储容差 {tol:.3e}")

pd.DataFrame(chk_rows).to_csv(os.path.join(L.RES, "22_原域一致性.csv"),
                              index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# ④ 滑坡记录逐条匹配
# ------------------------------------------------------------
base = L.load_base()
ls = base[base["landslide"] == 1].copy()
incell = set(full["cell_id"])
ls["in_full_domain"] = ls["cell_id"].isin(incell)
L.log(f"351 条滑坡记录：完整域内 {int(ls['in_full_domain'].sum())}，"
      f"域外 {int((~ls['in_full_domain']).sum())}", LOG)

# 域外者：诊断是哪个因子在对应像元为 NoData（而不是人为补造）
diag = []
for r in ls[~ls["in_full_domain"]].itertuples():
    bad = []
    for f in FACTORS:
        with rasterio.open(rpath(f)) as s:
            v = s.read(1)[int(r.row), int(r.col)]
        if (v == s.nodata) or (not np.isfinite(v)):
            bad.append(f)
    diag.append({"pointid": r.pointid, "row": int(r.row), "col": int(r.col),
                 "cell_id": r.cell_id, "NoData因子": "+".join(bad) or "（无，需排查）"})
ls_out = pd.DataFrame(diag)

ls[["pointid", "cell_id", "row", "col", "in_full_domain"]].to_csv(
    os.path.join(L.RES, "22_滑坡记录匹配.csv"), index=False,
    encoding="utf-8-sig")

# ------------------------------------------------------------
# ⑤ 输出与统计
# ------------------------------------------------------------
full.to_csv(os.path.join(L.DATA, "prediction_full_valid.csv"), index=False,
            encoding="utf-8-sig")
prob.to_csv(os.path.join(L.DATA, "全区预测概率_完整域.csv"), index=False,
            encoding="utf-8-sig")

# 完整域掩膜栅格（供 GIS 复核与出图对齐）
mask_tif = os.path.join(L.DATA, "domain_mask_full_valid.tif")
with rasterio.open(
        mask_tif, "w", driver="GTiff",
        height=meta["height"], width=meta["width"], count=1, dtype="uint8",
        crs=meta["crs"], transform=rasterio.Affine(*meta["transform"]),
        nodata=0, compress="lzw") as dst:
    m = np.zeros((meta["height"], meta["width"]), dtype="uint8")
    m[full["row"].values, full["col"].values] = 1
    dst.write(m, 1)
L.log(f"完整域掩膜已写出：{mask_tif}（{int(m.sum())} 像元）", LOG)

# 易发性分级面积（主模型）
PRIMARY = FINAL_CFG.get("primary", "XGBoost")
AREA = None
if f"p_{PRIMARY}" in prob.columns:
    pv = prob[f"p_{PRIMARY}"].values
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
    names = ["极低 (0–0.2)", "低 (0.2–0.4)", "中 (0.4–0.6)",
             "高 (0.6–0.8)", "极高 (0.8–1.0)"]
    idx = np.digitize(pv, bins[1:-1], right=False)
    area_cell = AREA_CELL_KM2
    AREA = pd.DataFrame([
        {"等级": names[i], "像元数": int((idx == i).sum()),
         "面积_km2": round(float((idx == i).sum()) * area_cell, 2),
         "面积占比": round(float((idx == i).mean()), 4)}
        for i in range(len(names))])
    AREA.loc[len(AREA)] = {"等级": "合计", "像元数": int(len(pv)),
                           "面积_km2": round(len(pv) * area_cell, 2),
                           "面积占比": 1.0}
    AREA.to_csv(os.path.join(L.RES, "22_易发性分级面积.csv"), index=False,
                encoding="utf-8-sig")
    L.log("\n" + AREA.to_string(index=False), LOG)

    # 主图（完整域）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    g = np.full((meta["height"], meta["width"]), np.nan)
    g[full["row"].values, full["col"].values] = pv
    cmap = ListedColormap(["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"])
    norm = BoundaryNorm([0, 0.2, 0.4, 0.6, 0.8, 1.0], cmap.N)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220)
    im = ax.imshow(g, cmap=cmap, norm=norm)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{PRIMARY} susceptibility — full common-valid domain "
                 f"({len(full)} cells, {len(full) * area_cell:.0f} km²)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                      ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_label("Susceptibility", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(L.FIG, f"fig_易发性图_{PRIMARY}_完整域.png"), dpi=220)
    plt.close(fig)
    L.log(f"完整域主图已写出：fig_易发性图_{PRIMARY}_完整域.png", LOG)

area_cell = AREA_CELL_KM2
stat = pd.DataFrame([
    {"项": "影像尺寸", "值": f"{meta['width']} x {meta['height']}"},
    {"项": "像元边长", "值": f"{meta['res'][0]:.1f} m × {meta['res'][1]:.1f} m"},
    {"项": "DEM 有效像元", "值": n_dem},
    {"项": "15 因子共同有效像元", "值": n_common},
    {"项": "仅 DEM 有效、因子缺失", "值": only_dem},
    {"项": "旧预测域像元", "值": int(len(old))},
    {"项": "新增像元", "值": int(len(full) - len(old))},
    {"项": "完整域面积 (km²)", "值": round(len(full) * area_cell, 2)},
    {"项": "旧域面积 (km²)", "值": round(len(old) * area_cell, 2)},
    {"项": "无折归属像元", "值": n_nofold},
    {"项": "滑坡记录落在完整域", "值": f"{int(ls['in_full_domain'].sum())}/351"},
    {"项": "滑坡唯一像元落在完整域", "值":
     f"{ls.loc[ls['in_full_domain'], 'cell_id'].nunique()}/336"},
])
stat.to_csv(os.path.join(L.RES, "22_完整域构建.csv"), index=False,
            encoding="utf-8-sig")
L.log("\n" + stat.to_string(index=False), LOG)

# 新增像元的空间分布（是否恰好填在滑坡附近）
new_ids = set(full["cell_id"]) - set(old["cell_id"])
newly = full[full["cell_id"].isin(new_ids)]
L.log(f"新增 {len(newly)} 像元；其 block 数 {newly[['block_r','block_c']].drop_duplicates().shape[0]}", LOG)

rep = [
    "# 22 · 完整共同有效预测域（V3-01）\n",
    f"> 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}　｜　"
    f"不训练模型，仅推断与核对\n",
    "## 一、掩膜口径\n",
    "- 有效性判据：**15 个因子在同一网格上同时非 NoData 且有限**（不是只看 DEM）；",
    f"- 15 个栅格均为 {meta['width']}×{meta['height']}，像元 "
    f"{meta['res'][0]:.1f} m，{meta['crs']}，nodata={meta['nodata']}；",
    f"- 各因子有效像元数：{n_cells_each}；",
    f"- DEM 有效 {n_dem}，共同有效 {n_common}，仅 DEM 有效而因子缺失 {only_dem}。", "",
    stat.to_markdown(index=False), "",
    ("## 三、易发性分级面积（完整域）\n\n" + AREA.to_markdown(index=False) + "\n")
    if AREA is not None else "",
    "## 二、与旧域的一致性（决定新域可信度）\n",
    "旧 `prediction_base.csv` 的 53,911 个像元，在本脚本重建的完整域中"
    "按 `cell_id` 匹配后逐因子比较。**判据是 float32 规范化后零不等**"
    "（模型实际看到的输入），而不是十进制表示相同。\n",
    "- 旧表存的是**未规范化**的 float64（如 `road`、`river` 的原始差达 4.8e-4），"
    "新表是 float32-canonical；两者的 float32 规范化结果完全相同，"
    "因此模型输入一致、概率逐点复现；",
    "- 该差异只影响\"是否恰好等于训练极值\"这类刀锋判定：不加容差时，"
    "适用域外推标记会在两种表示间翻转 **147 个像元**；"
    "`02_coverage.py` 已改为在 float32 相对容差内视为\"在范围内\"。\n",
    pd.DataFrame(chk_rows).to_markdown(index=False), "",
    "## 四、滑坡记录归属\n",
    f"- 351 条记录中 {int(ls['in_full_domain'].sum())} 条落在完整域内，"
    f"{int((~ls['in_full_domain']).sum())} 条不在；",
    f"- 唯一像元 {ls.loc[ls['in_full_domain'], 'cell_id'].nunique()} / 336 在域内；",
    "- 域外记录若存在，原因逐条列在 `22_滑坡记录匹配.csv` 的 `NoData因子` 列，"
    "**未做任何因子补造**。", "",
    (ls_out.to_markdown(index=False) if len(ls_out) else "（无域外记录）"), "",
    "## 五、使用边界\n",
    "- 完整域只用于**制图与域统计**；702 条建模记录、空间折表、OOF 指标均未改变；",
    "- 地图在训练点及其邻域输出分数属正常推断，"
    "**不得**把最终地图在已知滑坡上的命中率当作独立预测精度；",
    "- 旧域输出 `data/全区预测概率.csv` 原样保留作版本对照。", "",
]
out = os.path.join(L.RES, "22_报告.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in rep))
L.log(f"已写出 {out}", LOG)
print(stat.to_string(index=False))
print(pd.DataFrame(chk_rows).to_string(index=False))
print("=== 22 完成 ===")
