# -*- coding: utf-8 -*-
"""数据溯源与时间戳：记录清洗层的生成时间/来源/版本

用途:
    1. 清洗数据加入"生成时间戳"——知道这份数据是什么时候从 frozen 派生的
    2. 数据"可获得日期"（availability）——区分真实历史可用 vs 事后才发布
    3. 为 look-ahead（未来数据泄漏）检测提供元数据基础
"""
import json
import time
from datetime import datetime
from pathlib import Path

from .config import dir_of

META_NAME = "_meta.json"


def stamp_cleaned(meta: dict = None) -> Path:
    """为清洗层写入时间戳/来源元数据（每次重建自动调用）

    参数:
        meta: 附加元数据（如来源、清洗规则版本）

    写入:
        db/cleaned/daily_basic/_meta.json
    """
    info = {
        "layer": "cleaned",
        "recipe": "daily_basic",
        "source_layer": "frozen",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_timestamp": int(time.time()),
        "script": "scripts/rebuild_cleaned.py",
        **((meta or {}))
    }
    path = dir_of("daily") / META_NAME
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def get_stamp(dataset: str = "daily") -> dict:
    """读取数据集的时间戳元数据"""
    path = dir_of(dataset) / META_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def stamp_dataset(dataset: str, meta: dict) -> Path:
    """为任意数据集写入元数据（通用）"""
    info = {
        "dataset": dataset,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_timestamp": int(time.time()),
        **meta,
    }
    path = dir_of(dataset) / META_NAME
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ============================================================
# 数据指纹：判断"数据是否变过"，用于让派生缓存正确失效
# ============================================================
# 面板缓存（.cache/panel/<start>_<end>/）原先只以**区间**为键。数据被重建之后
# 缓存仍然命中，于是回测拿着旧数据算出结论 —— 这是一条会静默污染全部研究结论的
# 路径（本项目刚修完涨跌停价，就有 115 万行被改写，所有 2017–2020 的回测都受影响）。
# 因此缓存元数据里必须记下**所依赖数据集的指纹**，指纹变了就拒绝复用。
PANEL_DEPS = ("daily", "valuation", "adjust", "limit", "suspend")


def _fingerprint_one(dataset: str) -> dict:
    """单个数据集的指纹：文件数 + 最新 mtime + 生成时间戳

    用 os.scandir 逐层扫描（比 Path.glob 少建大量 Path 对象），
    7 万文件的数据集约 3 秒 —— 相对面板加载的 8 分钟可以忽略。
    """
    import os

    d = dir_of(dataset)
    if not d.exists():
        return {"exists": False}
    n = 0
    newest = 0.0
    try:
        years = [e for e in os.scandir(d) if e.is_dir() and e.name.startswith("year=")]
    except OSError:
        return {"exists": False}
    for y in years:
        try:
            for f in os.scandir(y.path):
                if not f.name.endswith(".parquet"):
                    continue
                n += 1
                try:
                    m = f.stat().st_mtime
                except OSError:
                    continue
                if m > newest:
                    newest = m
        except OSError:
            continue
    return {
        "exists": True,
        "n_files": n,
        "max_mtime": round(newest, 3),
        "generated": get_stamp(dataset).get("generation_timestamp"),
    }


def dataset_fingerprint(datasets=None) -> dict:
    """一组数据集的指纹（默认面板所依赖的全部数据集）

    只要任一数据集的文件数 / 最新修改时间 / 生成时间戳变了，指纹就变。
    比对时只比 `max_mtime` 与 `n_files`，`generated` 仅作展示。
    """
    return {ds: _fingerprint_one(ds) for ds in (datasets or PANEL_DEPS)}


def fingerprint_changed(old: dict, new: dict) -> list:
    """比较两个指纹，返回**发生变化的数据集名**列表（空 = 未变）"""
    changed = []
    for ds, cur in (new or {}).items():
        prev = (old or {}).get(ds)
        if prev is None:
            changed.append(ds)
            continue
        if (prev.get("n_files") != cur.get("n_files")
                or prev.get("max_mtime") != cur.get("max_mtime")):
            changed.append(ds)
    return changed



# ============================================================
# 数据可获得日期（availability）
# ============================================================
def attach_availability(df, trade_date_col="trade_date",
                        available_date_col=None):
    """为数据附加"_available"（该条数据的真实可获得日期）

    规则:
        - 价格/估值类（daily、valuation）: 可获得日 = 交易日
          （tushare 在收盘后发布，默认允许决策日使用当日数据）
        - 财务报表类（financial）: 可获得日 = 公告日 ann_date/f_ann_date
          （报告期 end_date 不等于发布日！用 end_date 判断 = 未来数据泄漏）

    参数:
        df:                 数据 DataFrame
        trade_date_col:     交易/报告期列名
        available_date_col: 可选，指定真实的发布日列（如 ann_date）
    """
    df = df.copy()
    if available_date_col and available_date_col in df.columns:
        df["_available"] = pd_to_datetime(df[available_date_col])
    else:
        df["_available"] = pd_to_datetime(df[trade_date_col])
    return df


import pandas as pd


def pd_to_datetime(series):
    return pd.to_datetime(series, errors="coerce")
