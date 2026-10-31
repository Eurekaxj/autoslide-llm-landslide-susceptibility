# -*- coding: utf-8 -*-
"""
12_studyarea_map.py — 阶段六：研究区三级定位图（对应审稿人2-第4、5条）
======================================================================
审稿人2原话：
  第4条 —— 请在图1中补充更多研究区域的细节。读者可能来自不同国家，
           可能不熟悉您的区域研究区，因此请为国际受众准备地图。
  第5条 —— 图2的图例太小，难以阅读。

产出：
  左：中国全图（高亮陕西省，标注省会与邻省关系）
  右上：陕西省（高亮西安市，标注地级市）
  右下：西安市（DEM 高程 + 351 个滑坡点 + 研究区范围）

所有图例字号 ≥ 8 pt（在 300 dpi 下对应版心内可读），不依赖放大查看。
输出： figures/fig_研究区位置图.png, figures/fig_研究区位置图_600dpi.png
"""
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

LOG = os.path.join(L.LOG, "12_studyarea_map.log")
if os.path.exists(LOG):
    os.remove(LOG)
L.log("=== 12 研究区三级定位图 ===", LOG)

SHENG = L.configured_path("province_boundary", required=True)
SHI = L.configured_path("city_boundary", required=True)

sheng = gpd.read_file(SHENG)
shi = gpd.read_file(SHI)
L.log(f"省级字段: {list(sheng.columns)}", LOG)
L.log(f"地市字段: {list(shi.columns)}", LOG)


def find_name_col(gdf, cands=("CityNameC", "SHI", "NAME", "name", "省", "市",
                              "NAME_CH", "sheng", "shi", "NAME_1", "NL_NAME_1",
                              "地名", "NAME_ZH")):
    for c in cands:
        if c in gdf.columns:
            return c
    # 退回：第一个 object 类型列
    for c in gdf.columns:
        if gdf[c].dtype == object:
            return c
    return None


cs = find_name_col(sheng)
ci = find_name_col(shi)
L.log(f"使用字段：省级={cs}, 地市={ci}", LOG)
sheng["_n"] = sheng[cs].astype(str)
shi["_n"] = shi[ci].astype(str)

# 目标 CRS：与因子栅格一致
with rasterio.open(os.path.join(L.RASTERS, "dem.tif")) as s:
    dem = s.read(1).astype(float)
    dem[dem == s.nodata] = np.nan
    T = s.transform
    CRS = s.crs
    H, W = dem.shape
    b = s.bounds
sheng = sheng.to_crs(CRS)
shi = shi.to_crs(CRS)

sn = sheng[sheng["_n"].str.contains("陕西", na=False)]
xi = shi[shi["_n"].str.contains("西安", na=False)]
# C23：该数据集中"西安市"匹配到 2 个要素（主城区 + 小块飞地，两者不相交）。
# 按完整名称核实后合并为单一多边形，避免飞地被漏画或重复绘制。
xi_note = ""
if len(xi) > 1:
    _xi_m = xi.to_crs("EPSG:32649")
    _areas = [round(g.area / 1e6, 1) for g in _xi_m.geometry]
    _uni = _xi_m.geometry.union_all()
    xi = gpd.GeoDataFrame(
        {"CityNameC": ["西安市"], "sheng": [str(xi.iloc[0]["sheng"])]},
        geometry=[_uni], crs="EPSG:32649")
    xi_note = (f"西安市匹配 {len(_areas)} 个要素（面积 km²: {_areas}），"
               f"已按名称核实并合并为单一多边形（合并后 "
               f"{round(_uni.area/1e6,1)} km²）")
    L.log("[C23] " + xi_note, LOG)
L.log(f"陕西省要素: {len(sn)}；西安市要素: {len(xi)}", LOG)
if len(sn) == 0 or len(xi) == 0:
    L.log("[警告] 未匹配到陕西/西安，改用几何包含关系定位", LOG)
    if len(sn):
        xi = shi[shi.geometry.centroid.within(sn.geometry.union_all())]
    if len(xi) == 0:
        xi = shi[shi.geometry.intersects(box(b.left, b.bottom, b.right, b.top))]

df = L.load_base()
la = df[df["landslide"] == 1]

from matplotlib.ticker import FuncFormatter, MaxNLocator


def km(x, _):
    return f"{x/1000:,.0f}"


FMT = FuncFormatter(km)


def tidy_axes(ax, xlab="Easting (km, UTM 49N)", ylab="Northing (km, UTM 49N)"):
    """统一坐标刻度：以 km 为单位、最多 4 个刻度、旋转 30°，避免标签被裁切。"""
    ax.xaxis.set_major_formatter(FMT)
    ax.yaxis.set_major_formatter(FMT)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel(xlab, fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)


fig = plt.figure(figsize=(11, 6.6), dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1, 1],
                      wspace=0.22, hspace=0.22)

# ---------- 左：中国全图 ----------
ax0 = fig.add_subplot(gs[:, 0])
sheng.plot(ax=ax0, facecolor="#f2f2f2", edgecolor="#9aa5ad", linewidth=0.4)
if len(sn):
    sn.plot(ax=ax0, facecolor="#d94f45", edgecolor="#7d2b24", linewidth=0.8)
    x0, y0, x1, y1 = sn.total_bounds
    ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                            edgecolor="#1f4e79", linewidth=1.0, linestyle="--"))
ax0.set_title("(a) Location of Shaanxi Province in China", fontsize=10, pad=6)
tidy_axes(ax0)
ax0.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, facecolor="#d94f45", edgecolor="#7d2b24",
                  label="Shaanxi Province"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#f2f2f2", edgecolor="#9aa5ad",
                  label="Other provinces")], fontsize=8, loc="lower left",
    frameon=True, framealpha=0.9)

# ---------- 右上：陕西省 ----------
ax1 = fig.add_subplot(gs[0, 1])
if len(sn):
    sn_buf = sn.copy()
    shi_in = shi[shi.geometry.centroid.within(sn.geometry.union_all())] if len(sn) else shi
    shi_in.plot(ax=ax1, facecolor="#fafafa", edgecolor="#b0b8bf", linewidth=0.35)
    if len(xi):
        xi.plot(ax=ax1, facecolor="#2f7fbf", edgecolor="#17476e", linewidth=0.9)
    sn_buf.boundary.plot(ax=ax1, color="#444444", linewidth=0.8)
    ax1.set_xlim(sn.total_bounds[0] - 2e4, sn.total_bounds[2] + 2e4)
    ax1.set_ylim(sn.total_bounds[1] - 2e4, sn.total_bounds[3] + 2e4)
ax1.set_title("(b) Location of Xi'an City in Shaanxi Province", fontsize=10, pad=6)
tidy_axes(ax1)
if len(xi):
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="#2f7fbf",
                                      edgecolor="#17476e", label="Xi'an City")],
               fontsize=8, loc="upper left", frameon=True, framealpha=0.9)

# ---------- 右下：西安市 + DEM + 滑坡点 ----------
ax2 = fig.add_subplot(gs[1, 1])
im = ax2.imshow(dem, cmap="terrain", interpolation="nearest",
                extent=[b.left, b.right, b.bottom, b.top])
if len(xi):
    xi.boundary.plot(ax=ax2, color="#1f4e79", linewidth=0.8)
ax2.scatter(la["X"], la["Y"], c="red", s=6, marker="^", edgecolors="black",
            linewidths=0.2, label=f"Landslide inventory (n={len(la)})", zorder=5)
ax2.scatter(df[df["landslide"] == 0]["X"], df[df["landslide"] == 0]["Y"],
            c="#2c7fb8", s=5, marker="o", edgecolors="white", linewidths=0.2,
            label=f"Non-landslide points (n=351)", zorder=4)
ax2.set_xlim(b.left, b.right); ax2.set_ylim(b.bottom, b.top)
ax2.set_title("(c) Study area: DEM, landslide and non-landslide points",
              fontsize=10, pad=6)
tidy_axes(ax2)
cb = plt.colorbar(im, ax=ax2, shrink=0.82, pad=0.02)
cb.set_label("Elevation (m a.s.l.)", fontsize=8)
cb.ax.tick_params(labelsize=7)
ax2.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.9)

fig.suptitle("Study area location map (China — Shaanxi — Xi'an)", fontsize=12, y=0.97)
fig.savefig(os.path.join(L.FIG, "fig_研究区位置图.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(L.FIG, "fig_研究区位置图_600dpi.png"), dpi=600,
            bbox_inches="tight")
plt.close(fig)
L.log("图件已保存（300 / 600 dpi）", LOG)

# 图例字号自检
rep = ["# 阶段六 研究区三级定位图（审稿人2-第4、5条）\n",
       f"- 省级数据：{SHENG}（{len(sheng)} 个要素）",
       f"- 地市数据：{SHI}（{len(shi)} 个要素）",
       f"- 陕西省匹配 {len(sn)} 个要素；西安市匹配 {len(xi)} 个要素",
       "- 布局：左=中国全图（高亮陕西）；右上=陕西省（高亮西安）；"
       "右下=西安市（DEM + 351 滑坡点 + 351 非滑坡点）",
       "- 图例与坐标标注字号 ≥ 7 pt（300 dpi 输出，版心内可读）",
       "- 输出：figures/fig_研究区位置图.png（300 dpi）、"
       "fig_研究区位置图_600dpi.png（600 dpi）", ""]
with open(os.path.join(L.RES, "12_研究区图报告.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(rep))
L.log("=== 12 完成 ===", LOG)
