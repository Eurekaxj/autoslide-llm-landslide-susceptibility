# -*- coding: utf-8 -*-
"""
exp_lib.py — 返修实验公共库
===========================
统一提供：数据加载、cell_id/空间块生成、训练折内预处理、六类模型与
超参数网格、空间分块交叉验证、指标与95%置信区间。

铁律（与 05_最终实验设置与完整执行方案.md 一致）：
1. 所有标准化/编码/阈值/超参数只在外层训练折内拟合，测试折与全区预测
   点仅使用训练折参数。
2. 所有实验共用同一套 702 样本、同一套空间折叠。
3. 指标、图号、性能提升幅度必须由实际运行产生，不得预填。
"""
import os
import sys
import json
import time
import hashlib
import warnings

import numpy as np
import pandas as pd

# Keep runtime warnings visible in the run logs.

# ============================================================
# 路径
# ============================================================
# 【V3-04】不再写死作者机器的 D 盘目录。解析顺序：
#   ① 环境变量 LSA_ROOT（指向研究根目录，即包含 最终因子集/ 的那一层）
#   ② config/paths.json（可分别指定 root / final_factor_set / point_dataset /
#      factor_rasters / aux_rasters，相对路径以 EXP 为基准）
#   ③ 回退：本文件所在 scripts/ 的上一级（即 返修实验_v3/）的上一级
# 整包拷到任意路径、或把因子集放到别处，都无需改代码。
EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_paths():
    _p = os.path.join(EXP, "config", "paths.json")
    if os.path.exists(_p):
        try:
            with open(_p, encoding="utf-8") as _f:
                return json.load(_f)
        except Exception:                                    # noqa: BLE001
            return {}
    return {}


def _abs(v, base):
    if not v:
        return None
    return v if os.path.isabs(v) else os.path.abspath(os.path.join(base, v))


_PATHS = _load_paths()
ROOT = (_abs(os.environ.get("LSA_ROOT"), EXP)
        or _abs(_PATHS.get("root"), EXP)
        or os.path.dirname(EXP))
# v2：按核查清单 C01 要求，封存 v1 结果，在独立目录重跑修正后的协议
V1_DIR = os.path.join(ROOT, "返修实验")
FINAL = _abs(_PATHS.get("final_factor_set"), EXP) or os.path.join(ROOT, "最终因子集")
POINTS = _abs(_PATHS.get("point_dataset"), EXP) or os.path.join(FINAL, "点数据集")
RASTERS = _abs(_PATHS.get("factor_rasters"), EXP) or os.path.join(FINAL, "因子栅格")
AUX_RASTERS = (_abs(_PATHS.get("aux_rasters"), EXP)
               or os.path.join(ROOT, "FactorPipeline_Output", "Rasters"))
DATA = os.path.join(EXP, "data")
RES = os.path.join(EXP, "results")
FIG = os.path.join(EXP, "figures")
LOG = os.path.join(EXP, "logs")
CFG = os.path.join(EXP, "config")

for d in (DATA, RES, FIG, LOG, CFG):
    os.makedirs(d, exist_ok=True)

# 解释器：优先环境变量 LSA_PY，否则用当前运行的解释器（迁移后自然正确）
PY_EXE = os.environ.get("LSA_PY") or sys.executable


def configured_path(key, required=False):
    value = os.environ.get("LSA_" + key.upper()) or _PATHS.get(key)
    path = _abs(value, EXP)
    if required and (not path or not os.path.exists(path)):
        raise FileNotFoundError(f"Configure {key!r} in config/paths.json; this is an optional raw-data rebuild input")
    return path

# ============================================================
# 因子定义
# ============================================================
# ------------------------------------------------------------
# 【v3 · R02】因子注册表 + 强断言
# 核查发现：P0_prcptot 的因子清单里写了 prcptot，但 CONTINUOUS 未登记，
# build_matrix 静默丢列，导致"有降雨"与"无降雨"矩阵完全相同（0.912047
# 实为无降雨对照）。此后所有请求的原始因子必须被恰好一个处理器接收，
# 未登记因子直接报错，不再静默忽略。
# ------------------------------------------------------------
CONTINUOUS = [
    # —— 15 个主因子中的连续项 ——
    "DEM", "slope", "plan", "profile", "NDVI", "roughness", "road",
    "river", "building", "rx5day", "raincv", "soil_clay", "soil_silt",
    # —— 消融/敏感性辅助因子（05/06/13/17）——
    "rainfall",          # 2019 年累计降雨（旧口径，G0）
    "temyear",           # 年平均气温（T1）
    "prcptot",           # 1990–2019 年平均年降水总量（插值版，仅存档）
    "prcptot_near",      # 同上，最近邻版（R03：与 rx5day/raincv 同一生产路径）
    "rx5day_krig", "raincv_krig",        # 克里金制图版（A1）
    "soil_sand", "soil_bd",
    # —— 降雨统计窗口变体（06）——
    "rx5day_W30", "raincv_W30", "rx5day_W10", "raincv_W10",
    "rx5day_W01", "raincv_W01",
    # —— 降雨候选指标（17 折内筛选）——
    "r99p", "rx1day", "sdii", "r95p", "r95frac", "pci", "cdd",
    "cwd", "raindays",
    # —— 年均温去高程残差（05 · T2）——
    "temp_resid",
]
CATEGORICAL = ["land_use"]
ASPECT = "aspect"

# 原始因子名 → 建模表列名（默认同名；此处显式登记便于审计）
RAW_COL = {f: f for f in CONTINUOUS + CATEGORICAL + [ASPECT]}

# 建模表中不参与建模的辅助列
NON_FACTOR_COLS = {
    "pointid", "Lon", "Lat", "X", "Y", "col", "row", "cell_id",
    "spatial_block_id", "fold", "fold_random", "valid_factors",
    "landslide", "possibility",
}


def assert_registry(columns=None):
    """
    注册表自检（R02）。
    ① 每个已登记因子必须恰好属于一个处理器类别（连续/分类/坡向派生）；
    ② 若给出 columns，则表内每个数值因子列都必须已登记，防止"表里有列、
       矩阵里没列"的静默丢列再次发生。
    """
    seen = {}
    for grp, names in (("continuous", CONTINUOUS),
                       ("categorical", CATEGORICAL),
                       ("aspect", [ASPECT])):
        for n in names:
            if n in seen and seen[n] != grp:
                raise ValueError(f"因子 {n} 同时被 {seen[n]} 与 {grp} 注册")
            seen[n] = grp
    if columns is not None:
        unknown = [c for c in columns
                   if c not in NON_FACTOR_COLS and c not in seen]
        if unknown:
            raise ValueError(f"建模表存在未登记因子列（会被静默丢弃）：{unknown}")
    return seen


def require_factors(factors):
    """断言请求的每个因子都已登记（R02：未知因子直接报错）。"""
    reg = assert_registry()
    unknown = [f for f in factors if f not in reg]
    if unknown:
        raise KeyError(f"未登记因子 {unknown}；请先在 CONTINUOUS/CATEGORICAL 注册")
    if len(set(factors)) != len(factors):
        dup = [f for f in set(factors) if list(factors).count(f) > 1]
        raise ValueError(f"因子清单重复：{dup}")
    return True

# 因子组
GROUPS = {
    "G0": ["DEM", "slope", "aspect", "plan", "profile", "NDVI", "land_use",
           "roughness", "road", "river", "rainfall"],
    "G1": ["DEM", "slope", "aspect", "plan", "profile", "NDVI", "land_use",
           "roughness", "road", "river", "rx5day", "raincv"],
    "G2": ["DEM", "slope", "aspect", "plan", "profile", "NDVI", "land_use",
           "roughness", "road", "river", "rx5day", "raincv",
           "soil_clay", "soil_silt"],
    "G3": ["DEM", "slope", "aspect", "plan", "profile", "NDVI", "land_use",
           "road", "river", "rx5day", "raincv",
           "soil_clay", "soil_silt"],
}
GROUP_DESC = {
    "G0": "基础因子 + 2019年累计降雨（旧口径对照）",
    "G1": "基础因子 + Rx5day + 日降雨CV（替换年累计降雨）",
    "G2": "G1 + 土壤黏粒/粉粒（30-60cm）= 最终14因子",
    "G3": "G2 去除 roughness（粗糙度敏感性对照）",
}

# 六类模型（与原论文 local run/ 下的脚本清单一致：RF/XGBoost/CatBoost/
# LogReg/MLP/SVM），LightGBM 作为附加对照（方案文档要求"至少包括"，且需
# 与 GBDT 家族对照），单独标注不属原论文六模型。
MODELS = ["RF", "XGBoost", "CatBoost", "LogReg", "MLP", "SVM", "LightGBM"]
MODELS_PAPER = ["RF", "XGBoost", "CatBoost", "LogReg", "MLP", "SVM"]
MODEL_FULLNAME = {
    "RF": "Random Forest",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "LightGBM": "LightGBM（附加对照）",
    "LogReg": "Logistic Regression",
    "MLP": "MLP (shallow neural network)",
    "SVM": "SVM (RBF / linear kernel)",
}


# ============================================================
# 工具
# ============================================================
def log(msg, f=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if f:
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def file_hash(path, algo="sha256", length=16):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# 数据加载（v3 · R01 统一数值规范）
# ============================================================
# 核查定位：v2 中滑坡行来自 CSV（float64），背景行由 float32 栅格读出后升为
# float64 再写表。两者地理点相同、float64 细节不同；标准化前的这层差异在
# 近共线因子上被树模型当成"标签纯净子组"的分类线索（NDVI 有 50 组同 float32
# 值在混合 float64 路径被拆开），使 2 km 方案 AUC 从 0.920 虚增到 0.972。
# 修复：所有入模连续因子一律先转 float32 再转 float64（与栅格原始精度一致），
# 在**构造矩阵这一唯一入口**统一执行，任何调用者都不可能绕过。
def canon_numeric(s):
    """把数值列归一到栅格原始精度（float32）后再升 float64。"""
    return np.asarray(s, dtype=np.float32).astype(np.float64)


def canonicalize(df, cols=None):
    """返回按统一规范处理过的副本（不修改入参）。"""
    out = df.copy()
    cols = [c for c in (cols or (CONTINUOUS + CATEGORICAL)) if c in out.columns]
    for c in cols:
        if c != "land_use":
            out[c] = canon_numeric(out[c].values)
    return out


def load_base(canon=False):
    """
    702 样本（含辅助因子列），附带 UTM 坐标、row/col、cell_id。

    注意：默认**不做**数值规范化，保持派生表原样。统一精度只在建模矩阵入口
    （build_matrix / matrix_for_categorical_model）执行，避免下游脚本
    （如 06 回写 modeling_base.csv）把规范化"顺手"固化进派生表，
    破坏"派生表逐字节可复现"这一验收条件。
    """
    path = os.path.join(DATA, "modeling_base.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("请先运行 00_prepare.py 生成 modeling_base.csv")
    df = pd.read_csv(path)
    assert_registry(df.columns)
    return canonicalize(df) if canon else df


def load_prediction(canon=False):
    path = os.path.join(DATA, "prediction_base.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("缺少 prediction_base.csv")
    df = pd.read_csv(path)
    assert_registry(df.columns)
    return canonicalize(df) if canon else df


# ============================================================
# 空间折叠（v2：C04 固定查找表，与样本无关）
# ============================================================
FOLD_TABLE_PATH = os.path.join(CFG, "fold_table.csv")


def build_fold_table(block_cells=20, n_splits=5, seed=42):
    """
    按核查清单 C04 建立**与背景样本无关**的 block→fold 查找表：
    以研究区 DEM 有效像元划分的网格块（block_cells × block_cells 像元）的
    **几何中心**作为聚类对象，KMeans 聚成 n_splits 个空间连续分区。
    这样 1/2/3 km 缓冲半径共享同一套 fold，滑坡点的分区不再随背景点改变。
    """
    import rasterio
    from sklearn.cluster import KMeans
    with rasterio.open(os.path.join(RASTERS, "dem.tif")) as s:
        dem = s.read(1).astype(float)
        nd = s.nodata
    valid = dem != nd
    rr, cc = np.where(valid)
    br = rr // block_cells
    bc = cc // block_cells
    blocks = np.unique(np.column_stack([br, bc]), axis=0)
    cen = np.column_stack([blocks[:, 0] * block_cells + block_cells / 2.0,
                           blocks[:, 1] * block_cells + block_cells / 2.0])
    n_cl = min(n_splits, len(blocks))
    km = KMeans(n_clusters=n_cl, random_state=seed, n_init=10).fit(cen)
    tbl = pd.DataFrame({"block_r": blocks[:, 0].astype(int),
                        "block_c": blocks[:, 1].astype(int),
                        "fold": km.labels_.astype(int)})
    L_tbl = {"block_cells": block_cells, "n_splits": n_splits, "seed": seed,
             "n_blocks": int(len(tbl)),
             "n_cells_valid": int(valid.sum())}
    save_json(L_tbl, os.path.join(CFG, "fold_table_meta.json"))
    return tbl


def get_fold_table(block_cells=20, n_splits=5, seed=42):
    if os.path.exists(FOLD_TABLE_PATH):
        return pd.read_csv(FOLD_TABLE_PATH)
    tbl = build_fold_table(block_cells, n_splits, seed)
    tbl.to_csv(FOLD_TABLE_PATH, index=False, encoding="utf-8-sig")
    return tbl


def assign_folds(df, block_cells=20, n_splits=5, seed=42):
    """按 (row, col) 查固定表得到 fold；样本集合变化不影响分区。"""
    tbl = get_fold_table(block_cells, n_splits, seed)
    key = pd.MultiIndex.from_arrays([df["row"].values // block_cells,
                                     df["col"].values // block_cells])
    lut = pd.Series(tbl["fold"].values,
                    index=pd.MultiIndex.from_arrays([tbl["block_r"], tbl["block_c"]]))
    f = lut.reindex(key).values
    if pd.isna(f).any():
        bad = int(pd.isna(f).sum())
        raise ValueError(f"{bad} 个样本的空间块不在 fold 查找表内（检查 row/col 是否越界）")
    return f.astype(int)


def make_spatial_folds(df, n_splits=5, seed=42, block_cells=20):
    """兼容旧调用：v2 起统一走固定查找表。"""
    return assign_folds(df, block_cells=block_cells, n_splits=n_splits, seed=seed)


def make_random_folds(df, n_splits=5, seed=42):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), dtype=int)
    for k, (_, te) in enumerate(skf.split(df, df["landslide"])):
        fold[te] = k
    return fold


# ============================================================
# 预处理（训练折内拟合）
# ============================================================
CONT_DERIVED = ["aspect_sin", "aspect_cos", "aspect_flat"]


def _check_consumed(factors, cont, has_aspect, has_cat):
    """【R02】确认每个请求因子恰好被一个处理器接收。"""
    require_factors(factors)
    got = list(cont)
    if has_aspect:
        got.append(ASPECT)
    if has_cat:
        got.append("land_use")
    if sorted(got) != sorted(factors):
        miss = sorted(set(factors) - set(got))
        extra = sorted(set(got) - set(factors))
        raise RuntimeError(f"因子未被完整消费：漏={miss} 多={extra}")
    return True


def _factor_map(cont, has_aspect, has_cat, enc):
    m = {f: [f] for f in cont}
    if has_aspect:
        m[ASPECT] = ["aspect_sin", "aspect_cos", "aspect_flat"]
    if has_cat:
        m["land_use"] = list(enc.get("dummy_cols", ["land_use_code"]))
    return m


def _matrix_hash(X):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X.values, dtype=np.float64).tobytes())
    return h.hexdigest()


def build_matrix(df, factors, enc=None, fit=True):
    """
    构造模型矩阵。fit=True 时学习标准化/编码参数并返回 enc。
    aspect → sin/cos/flat；land_use → one-hot（drop_first 由 enc 控制）。
    【v3】连续因子统一 float32 精度（R01）；未登记因子直接报错（R02）。
    """
    cont = [f for f in factors if f in CONTINUOUS]
    has_aspect = ASPECT in factors
    has_cat = "land_use" in factors
    _check_consumed(factors, cont, has_aspect, has_cat)

    # R01：唯一数值入口，先统一到栅格原始精度（保留原索引，避免 concat 错位）
    raw = pd.DataFrame({f: canon_numeric(df[f].values) for f in cont},
                       index=df.index)

    if fit:
        enc = {"cont": cont, "has_aspect": has_aspect, "has_cat": has_cat}
        mu = raw.mean()
        sd = raw.std(ddof=0).replace(0, 1.0)
        enc["mu"] = mu.to_dict()
        enc["sd"] = sd.to_dict()
    else:
        mu = pd.Series(enc["mu"])
        sd = pd.Series(enc["sd"])

    X = (raw - mu) / sd
    X = X.astype(float)

    if has_aspect:
        a = canon_numeric(df[ASPECT].values)
        flat = (a < 0).astype(float)
        ar = np.where(a < 0, 0.0, np.deg2rad(a))
        X["aspect_sin"] = np.sin(ar)
        X["aspect_cos"] = np.cos(ar)
        X["aspect_flat"] = flat

    if has_cat:
        lu = df["land_use"].astype(int)
        cats = sorted(lu.unique()) if fit else enc["lu_cats"]
        if fit:
            enc["lu_cats"] = cats
        dummies = pd.get_dummies(lu.astype("category").cat.set_categories(cats))
        dummies.columns = [f"lu_{int(c)}" for c in dummies.columns]
        dummies = dummies.astype(float)
        if fit:
            enc["dummy_cols"] = list(dummies.columns)
        dummies = dummies.reindex(columns=enc["dummy_cols"], fill_value=0.0)
        X = pd.concat([X, dummies], axis=1)

    if fit:
        enc["factor_map"] = _factor_map(cont, has_aspect, has_cat, enc)
        enc["matrix_cols"] = list(X.columns)
        enc["matrix_sha256"] = _matrix_hash(X)
        enc["numeric_spec"] = "float32-canonical"
    return X, enc


def matrix_for_categorical_model(df, factors, enc=None, fit=True):
    """CatBoost / LightGBM：land_use 保留整数类别码，其余同 build_matrix。"""
    cont = [f for f in factors if f in CONTINUOUS]
    has_aspect = ASPECT in factors
    has_cat = "land_use" in factors
    _check_consumed(factors, cont, has_aspect, has_cat)

    raw = pd.DataFrame({f: canon_numeric(df[f].values) for f in cont},
                       index=df.index)

    if fit:
        enc = {"cont": cont, "has_aspect": has_aspect, "has_cat": has_cat}
        mu = raw.mean()
        sd = raw.std(ddof=0).replace(0, 1.0)
        enc["mu"] = mu.to_dict()
        enc["sd"] = sd.to_dict()
        enc["lu_cats"] = sorted(df["land_use"].astype(int).unique())
        enc["cat_cols"] = ["land_use_code"] if has_cat else []
    else:
        mu = pd.Series(enc["mu"])
        sd = pd.Series(enc["sd"])

    X = (raw - mu) / sd
    X = X.astype(float)
    if has_aspect:
        a = canon_numeric(df[ASPECT].values)
        flat = (a < 0).astype(float)
        ar = np.where(a < 0, 0.0, np.deg2rad(a))
        X["aspect_sin"] = np.sin(ar)
        X["aspect_cos"] = np.cos(ar)
        X["aspect_flat"] = flat
    if has_cat:
        X["land_use_code"] = df["land_use"].astype(int).values
    if fit:
        enc["factor_map"] = _factor_map(cont, has_aspect, has_cat,
                                        {"dummy_cols": ["land_use_code"]})
        enc["matrix_cols"] = list(X.columns)
        enc["matrix_sha256"] = _matrix_hash(X)
        enc["numeric_spec"] = "float32-canonical"
    return X, enc


# ============================================================
# 模型与超参数网格
# ============================================================
def get_model_and_grid(name, seed, n_jobs=1):
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
        g = {"n_estimators": [400], "max_depth": [4, 8],
             "min_samples_leaf": [5, 10], "max_features": ["sqrt"]}
    elif name == "XGBoost":
        from xgboost import XGBClassifier
        m = XGBClassifier(random_state=seed, eval_metric="logloss",
                          n_jobs=n_jobs, tree_method="hist")
        g = {"n_estimators": [100, 200], "max_depth": [2, 3],
             "learning_rate": [0.05], "subsample": [0.8],
             "colsample_bytree": [0.8], "min_child_weight": [1, 5],
             "reg_lambda": [1.0, 10.0]}
        if os.environ.get("LSA_FAST_GRID") == "1":
            # Sensitivity runs only: depth and round count are searched, the two
            # regularisation parameters are fixed at the modal values of the main run.
            g = {"n_estimators": [100, 200], "max_depth": [2, 3],
                 "learning_rate": [0.05], "subsample": [0.8],
                 "colsample_bytree": [0.8], "min_child_weight": [1],
                 "reg_lambda": [1.0]}
    elif name == "CatBoost":
        from catboost import CatBoostClassifier
        # Budget-limited illustrative grid; no claim of exhaustive optimality.
        # l2_leaf_reg is the installed CatBoost default (3.0 for this version).
        m = CatBoostClassifier(random_state=seed, verbose=0,
                               allow_writing_files=False, thread_count=n_jobs)
        g = {"iterations": [100, 200], "depth": [3, 4],
             "learning_rate": [0.05], "l2_leaf_reg": [10]}
    elif name == "LightGBM":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1)
        g = {"n_estimators": [200], "num_leaves": [7, 15, 31],
             "learning_rate": [0.05], "min_child_samples": [20, 50]}
    elif name == "LogReg":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(random_state=seed, max_iter=2000, solver="lbfgs")
        g = {"C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}
    elif name == "MLP":
        from sklearn.neural_network import MLPClassifier
        # Frozen 18-setting operational grid, not an unsupported claim about a pilot.
        m = MLPClassifier(random_state=seed, max_iter=1000, early_stopping=True,
                          n_iter_no_change=20)
        g = {"hidden_layer_sizes": [(64,), (128, 64), (256, 128)],
             "alpha": [1e-3, 1e-2]}
    elif name == "SVM":
        from sklearn.svm import SVC
        # 与原论文 local run/SVM+param.py 的核函数设定一致（RBF/线性为主），
        # 去掉 poly/sigmoid 以控制计算成本（poly 单次拟合约慢 5-10 倍）。
        # probability=True 用于输出概率（AUC / Brier / 校准曲线 / 全区概率图）。
        m = SVC(random_state=seed, probability=True, cache_size=500)
        g = {"kernel": ["rbf", "linear"], "C": [0.1, 1, 10, 100],
             "gamma": ["scale", 0.1, 1]}
    else:
        raise ValueError(name)
    return m, g


def grid_size(name):
    _, g = get_model_and_grid(name, 0)
    n = 1
    for v in g.values():
        n *= len(v)
    return n, g


def instantiate(name, params, seed, n_jobs=1):
    m, _ = get_model_and_grid(name, seed, n_jobs)
    m.set_params(**params)
    return m


# ============================================================
# 指标
# ============================================================
def metrics_at(y, p, thr=0.5):
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 accuracy_score, precision_score, recall_score,
                                 confusion_matrix, brier_score_loss)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "f1": f1_score(y, yh, zero_division=0),
        "accuracy": accuracy_score(y, yh),
        "precision": precision_score(y, yh, zero_division=0),
        "recall": recall_score(y, yh, zero_division=0),
        "specificity": spec,
        "balanced_accuracy": 0.5 * (recall_score(y, yh, zero_division=0) + spec),
        "brier": brier_score_loss(y, p),
        "threshold": thr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def ci95(values, bounded=False):
    """
    外层折均值 ± t(0.975, n-1)·SE。
    【v2 · C17】对取值有界的指标（AUC/PR-AUC/F1/Brier 等 ∈[0,1]）可选择 logit
    变换后计算区间再逆变换，保证 CI 落在 (0,1) 内，**不做截断裁剪**。
    """
    from scipy import stats
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return (float(v[0]) if n else np.nan), np.nan, np.nan
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n))
    t = stats.t.ppf(0.975, n - 1)
    if bounded and 0.0 < m < 1.0 and se > 0:
        eps = 1e-9
        mm = min(max(m, eps), 1 - eps)
        g = np.log(mm / (1 - mm))
        se_g = se / (mm * (1 - mm))          # delta method
        lo = 1.0 / (1.0 + np.exp(-(g - t * se_g)))
        hi = 1.0 / (1.0 + np.exp(-(g + t * se_g)))
        return m, float(lo), float(hi)
    return m, m - t * se, m + t * se


def block_bootstrap(y, P, block_ids, n_boot=2000, seed=42, n_jobs=-1):
    """
    【v2 · C17】按**空间块**重采样的配对 bootstrap。
    与 v1 的按行独立重采样不同：这里以空间块为抽样单位，保留块内全部记录，
    从而反映空间依赖性（v1 的区间偏窄、p 值偏小）。
    y: (n,) 标签；P: (n,k) 各模型 OOF 概率；block_ids: (n,) 空间块编号。
    返回 list of ndarray，每个元素是一次重采样下各模型的 AUC。
    """
    from sklearn.metrics import roc_auc_score
    from joblib import Parallel, delayed
    y = np.asarray(y).astype(int)
    P = np.asarray(P, float)
    blk = np.asarray(block_ids)
    uniq = np.unique(blk)
    idx_by_block = {b: np.where(blk == b)[0] for b in uniq}
    rng = np.random.default_rng(seed)
    draws = [rng.choice(uniq, len(uniq), replace=True) for _ in range(n_boot)]

    def one(pick):
        idx = np.concatenate([idx_by_block[b] for b in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            return None
        return [roc_auc_score(yy, P[idx, j]) for j in range(P.shape[1])]

    out = Parallel(n_jobs=n_jobs)(delayed(one)(d) for d in draws)
    return [o for o in out if o is not None]


def paired_block_bootstrap(y, p_a, p_b, block_ids, n_boot=2000, seed=42,
                           n_jobs=-1):
    """
    【v3 · R07】两个**配对**预测（同一批样本、同一批空间块）的 AUC 差值区间。
    与分别做两次独立 bootstrap 再相减不同：这里每次重采样**同时**取两个模型
    在该重采样上的 AUC，差值分布反映配对结构，区间更窄也更正确。
    返回 dict：auc_a/auc_b 为原样本上的点估计。
    """
    from sklearn.metrics import roc_auc_score
    from joblib import Parallel, delayed
    y = np.asarray(y).astype(int)
    a = np.asarray(p_a, float)
    b = np.asarray(p_b, float)
    blk = np.asarray(block_ids)
    uniq = np.unique(blk)
    idx_by_block = {u: np.where(blk == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    draws = [rng.choice(uniq, len(uniq), replace=True) for _ in range(n_boot)]

    def one(pick):
        idx = np.concatenate([idx_by_block[u] for u in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            return None
        return (roc_auc_score(yy, a[idx]), roc_auc_score(yy, b[idx]))

    out = [o for o in Parallel(n_jobs=n_jobs)(delayed(one)(d) for d in draws)
           if o is not None]
    da = np.array([o[0] for o in out])
    db = np.array([o[1] for o in out])
    dif = da - db
    return {
        "auc_a": float(roc_auc_score(y, a)),
        "auc_b": float(roc_auc_score(y, b)),
        "delta": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "lo": float(np.percentile(dif, 2.5)),
        "hi": float(np.percentile(dif, 97.5)),
        "p_le0": float((dif <= 0).mean()),
        "p_ge0": float((dif >= 0).mean()),
        "n_valid": int(len(out)),
        "boot_mean": float(dif.mean()),
        "boot_sd": float(dif.std(ddof=1)),
    }


def threshold_from_oof(oof, y):
    """由训练折内层CV的OOF概率选最大化 F1 的阈值。"""
    from sklearn.metrics import f1_score
    grid = np.unique(np.round(np.linspace(0.05, 0.95, 91), 3))
    best, bt = -1, 0.5
    for t in grid:
        s = f1_score(y, (oof >= t).astype(int), zero_division=0)
        if s > best:
            best, bt = s, float(t)
    return float(bt)


def apply_transform(tf, d_fit, d_apply):
    """【v3 · R05】在 d_fit 上拟合折内派生变换，再应用到 d_apply。"""
    if tf is None:
        return d_fit, d_apply
    fitted = tf.fit(d_fit)
    return fitted.transform(d_fit), fitted.transform(d_apply)


def inner_select_raw(base, grid, dtr, ytr, factors, model_name, seed=42,
                     n_jobs=10, n_inner=5, transform=None):
    """
    【v2 · C05】训练折内网格搜索，**每个内层训练子集独立拟合预处理**。
    【v3 · R05】若给出 transform，则**每个内层训练子集也独立拟合派生变换**
    （如年均温对 DEM 的回归残差），内层验证子集不参与系数估计。
    返回 (best_params, threshold, best_inner_auc, inner_oof, tf_records)
    """
    from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from joblib import Parallel, delayed

    dtr = dtr.reset_index(drop=True)
    groups = dtr["cell_id"].values if "cell_id" in dtr.columns else np.arange(len(dtr))
    sgkf = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    splits = list(sgkf.split(dtr, ytr, groups))
    combos = list(ParameterGrid(grid))
    native = model_name in ("CatBoost", "LightGBM")

    # 每个内层训练子集独立拟合派生变换（与参数组合无关，先算一次）
    tf_list, tf_records = [None] * len(splits), []
    if transform is not None:
        for i, (a, b) in enumerate(splits):
            fit_df = dtr.iloc[a]
            f = transform.fit(fit_df)
            tf_list[i] = f
            tf_records.append({
                "inner_fold": i, "n_fit": int(len(fit_df)),
                "fit_ids_sha256": hashlib.sha256(
                    "|".join(sorted(map(str, dtr.iloc[a]["pointid"].values)))
                    .encode()).hexdigest(),
                "intercept": float(getattr(f, "intercept_", np.nan)),
                "coef_DEM": float(getattr(f, "coef_", np.nan)),
            })

    # The transform depends on the training subset, not model hyperparameters.
    # Fit once per inner split and cache only that split's matrices. No validation
    # data are used for fitting, and caches never cross an outer fold boundary.
    matrices = []
    for i, (a, b) in enumerate(splits):
        da, db = dtr.iloc[a], dtr.iloc[b]
        if transform is not None:
            da, db = tf_list[i].transform(da), tf_list[i].transform(db)
        if native:
            Xa, enc = matrix_for_categorical_model(da, factors, fit=True)
            Xb, _ = matrix_for_categorical_model(db, factors, enc=enc, fit=False)
            ci = [list(Xa.columns).index(c) for c in enc.get("cat_cols", [])]
            fk = ({"cat_features": ci} if (model_name == "CatBoost" and ci)
                  else ({"categorical_feature": enc.get("cat_cols", [])} if ci else {}))
        else:
            Xa, enc = build_matrix(da, factors, fit=True)
            Xb, _ = build_matrix(db, factors, enc=enc, fit=False)
            fk = {}
        matrices.append((a, b, Xa, Xb, fk))

    def work(params):
        oof = np.full(len(dtr), np.nan)
        for a, b, Xa, Xb, fk in matrices:
            m = clone(base)
            m.set_params(**params)
            m.fit(Xa, ytr[a], **fk)
            oof[b] = m.predict_proba(Xb)[:, 1]
        ok = ~np.isnan(oof)
        return roc_auc_score(ytr[ok], oof[ok]), oof

    out = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(work)(p) for p in combos)
    aucs = np.array([o[0] for o in out])
    i = int(np.argmax(aucs))
    best_oof = out[i][1]
    ok = ~np.isnan(best_oof)
    return (dict(combos[i]), threshold_from_oof(best_oof[ok], ytr[ok]),
            float(aucs[i]), best_oof, tf_records)


def metrics_with_fold_threshold(y, p, fold_ids, thr_by_fold):
    """
    【v2 · C18】合并 OOF 指标：用**逐折自己的阈值**判定 yhat，再合并混淆矩阵。
    避免 v1 用「各折阈值平均」当总体阈值造成口径不一致。
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    yhat = np.zeros(len(y), dtype=int)
    for k, thr in thr_by_fold.items():
        msk = fold_ids == k
        yhat[msk] = (p[msk] >= thr).astype(int)
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tp = int(((y == 1) & (yhat == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        "accuracy": (tp + tn) / len(y),
        "precision": prec, "recall": rec, "specificity": spec,
        "balanced_accuracy": 0.5 * (rec + spec),
        "brier": brier_score_loss(y, p),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold_scheme": "per_fold",
    }


# ============================================================
# 空间分块交叉验证主过程（v2）
# ============================================================
def run_spatial_cv(df, factors, model_name, seed=42, n_splits=5,
                   folds=None, inner_jobs=10, tune=True, fixed_params=None,
                   collect_extra=False, fold_transform=None, transform=None,
                   model_seed=None, fixed_params_by_fold=None,
                   fixed_params_thr=None):
    """
    外层空间分块CV（v2 协议）。
    v2 修正点：
      C05 内层每个训练子集独立拟合预处理，内层按 cell 分组；
      C18 合并 OOF 指标用逐折阈值；
      额外返回 block_ids 与逐折阈值，供 C17 空间块 bootstrap 使用。
    fold_transform(dtr, dte) -> (dtr, dte)：**已弃用**，改用 transform 对象。
    transform：可选折内派生变换对象，需实现 fit(df)->fitted 与
      transform(df)->df（如 R05 的年均温对 DEM 回归残差）。外层训练折拟合
      一次用于外层，内层每个训练子集再各自独立拟合。
    """
    from sklearn.base import clone
    if fold_transform is not None and transform is None:
        raise TypeError("fold_transform 已弃用，请传 transform 对象（R05）")

    y_all = df["landslide"].astype(int).values
    if folds is None:
        folds = df["fold"].values
    oof = np.zeros(len(df))
    fold_rows, best_params, train_rows = [], [], []
    extras = {}
    inner_aucs = []
    tf_records_all = []

    for k in range(n_splits):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]
        dtr, dte = df.iloc[tr].copy(), df.iloc[te].copy()
        if transform is not None:
            dtr, dte = apply_transform(transform, dtr, dte)
            extras.setdefault("outer_tf", []).append({
                "outer_fold": k, "n_fit": int(len(tr)),
                "fit_ids_sha256": hashlib.sha256(
                    "|".join(sorted(map(str, df.iloc[tr]["pointid"].values)))
                    .encode()).hexdigest(),
            })
        ytr, yte = dtr["landslide"].astype(int).values, dte["landslide"].astype(int).values

        native_cat = model_name in ("CatBoost", "LightGBM")
        base, grid = get_model_and_grid(model_name, seed, n_jobs=1)

        if tune and fixed_params is None:
            # 内层搜索：传原始 dtr，由函数内部对每个子集独立拟合预处理
            best, thr, iauc, _, tfr = inner_select_raw(
                base, grid, dtr, ytr, factors, model_name,
                seed=seed, n_jobs=inner_jobs, transform=transform)
            inner_aucs.append(iauc)
            for r in tfr:
                r["outer_fold"] = k
                tf_records_all.append(r)
        else:
            # 【v3 · R04】允许逐折给定固定参数（设计A 必须复现主实验配置）
            best = (dict(fixed_params_by_fold[k]) if fixed_params_by_fold
                    else (dict(fixed_params) if fixed_params else {}))
            thr = (float(fixed_params_thr[k]) if fixed_params_thr
                   else 0.5)

        # 用最优参数在完整外层训练折上重新拟合预处理 + 模型
        if native_cat:
            Xtr, enc = matrix_for_categorical_model(dtr, factors, fit=True)
            Xte, _ = matrix_for_categorical_model(dte, factors, enc=enc, fit=False)
            ci = [list(Xtr.columns).index(c) for c in enc.get("cat_cols", [])]
            fitkw = ({"cat_features": ci} if (model_name == "CatBoost" and ci)
                     else ({"categorical_feature": enc.get("cat_cols", [])}
                           if ci else {}))
        else:
            Xtr, enc = build_matrix(dtr, factors, fit=True)
            Xte, _ = build_matrix(dte, factors, enc=enc, fit=False)
            fitkw = {}

        model = clone(base)
        model.set_params(**best)
        if model_seed is not None:          # R04：只改模型随机源，不动切分
            model.set_params(random_state=model_seed)
        model.fit(Xtr, ytr, **fitkw)

        ptr = model.predict_proba(Xtr)[:, 1]
        pte = model.predict_proba(Xte)[:, 1]
        oof[te] = pte

        mte = metrics_at(yte, pte, thr)
        mtr = metrics_at(ytr, ptr, thr)
        mte.update({"fold": k, "model": model_name, "n_train": len(tr),
                    "n_test": len(te)})
        mtr.update({"fold": k, "model": model_name, "split": "train"})
        fold_rows.append(mte)
        train_rows.append(mtr)
        best_params.append({"fold": k, "model": model_name,
                            "threshold": thr,
                            "inner_auc": (inner_aucs[-1] if inner_aucs else np.nan),
                            "params": json.dumps(best, ensure_ascii=False)})

        if collect_extra and k == 0:
            extras["feature_names"] = list(Xtr.columns)

    thr_by_fold = {int(r["fold"]): float(r["threshold"]) for r in fold_rows}
    pooled = metrics_at(y_all, oof, 0.5)
    pooled_thr = metrics_with_fold_threshold(y_all, oof, folds, thr_by_fold)

    return {
        "fold_metrics": pd.DataFrame(fold_rows),
        "train_metrics": pd.DataFrame(train_rows),
        "pooled": pooled,
        "pooled_fold_thr": pooled_thr,
        "pooled_at_mean_thr": pooled_thr,   # 兼容旧字段名
        "oof": oof,
        "folds": np.asarray(folds),
        "thr_by_fold": thr_by_fold,
        "block_ids": (df["spatial_block_id"].values
                      if "spatial_block_id" in df.columns else None),
        "best_params": pd.DataFrame(best_params),
        "extras": extras,
        "tf_records": pd.DataFrame(tf_records_all),
        "outer_tf": extras.get("outer_tf", []),
    }


def summarize_cv(res, model_name, dataset=""):
    """Fold mean/SD and pooled OOF; overlapping CV folds are not independent replicates."""
    fm = res["fold_metrics"]
    row = {"dataset": dataset, "model": model_name, "n_folds": len(fm)}
    for key in ["roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall",
                "specificity", "balanced_accuracy", "brier"]:
        row[f"{key}_mean"] = float(fm[key].mean())
        row[f"{key}_std"] = float(np.std(fm[key].values, ddof=1))
    pl = res.get("pooled_fold_thr", res["pooled"])
    row["oof_roc_auc"] = res["pooled"]["roc_auc"]
    row["oof_pr_auc"] = res["pooled"]["pr_auc"]
    row["oof_brier"] = res["pooled"]["brier"]
    # 合并 OOF 的 F1/Acc 等用逐折阈值判定（C18）
    row["oof_f1_foldthr"] = pl["f1"]
    row["oof_acc_foldthr"] = pl["accuracy"]
    row["mean_threshold"] = float(np.mean(fm["threshold"].values))
    row["fold_mean_auc_minus_oof_auc"] = row["roc_auc_mean"] - row["oof_roc_auc"]
    tm = res["train_metrics"]
    row["train_roc_auc"] = float(tm["roc_auc"].mean())
    row["gap_roc_auc"] = row["train_roc_auc"] - row["roc_auc_mean"]
    row["gap_f1"] = float(tm["f1"].mean()) - row["f1_mean"]
    return row
