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

DB = Path(__file__).resolve().parent.parent / "db"
META_NAME = "_meta.json"


def stamp_cleaned(meta: dict = None) -> Path:
    """为清洗层写入时间戳/来源元数据（每次重建自动调用）

    参数:
        meta: 附加元数据（如来源、清洗规则版本）

    写入:
        db/daily/_meta.json
    """
    info = {
        "layer": "cleaned",
        "source_layer": "frozen",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_timestamp": int(time.time()),
        "script": "scripts/rebuild_cleaned.py",
        **((meta or {}))
    }
    path = DB / "daily" / META_NAME
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def get_stamp(dataset: str = "daily") -> dict:
    """读取数据集的时间戳元数据"""
    path = DB / dataset / META_NAME
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
    path = DB / dataset / META_NAME
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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
