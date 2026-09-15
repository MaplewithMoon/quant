# -*- coding: utf-8 -*-
"""全量重建涨跌停价（清洗配方 limit_price）

基于 db/cleaned/daily_basic（不复权原始价）重算，输出 db/cleaned/limit_price。

修正旧实现的两个缺陷
--------------------
1. **昨收口径**：旧实现用 `close.shift(1)`，导致
     - 每年首个交易日没有涨跌停价（年内分区内 shift 后 dropna）
     - **除权除息日涨跌停价错误**：交易所用的是除权后参考价，不是上一日收盘价
   改为使用 tushare 官方 `pre_close` 列（已按除权调整）。

2. **规则分叉**：旧实现有两套互相冲突的板块/ST 规则
     - database/downloader/meta.py ：有 ST 5%，无北交所
     - scripts/daily_update.py     ：有北交所 920，无 ST
   二者写同一个目录、互相覆盖。现在集中到本文件一处。

板块涨跌幅规则（2024 年口径）
    - 科创板 688/689            ±20%
    - 创业板 300/301            ±20%
    - 北交所 920/43/83/87/88    ±30%
    - 主板 ST/*ST               ±5%
    - 主板其它                  ±10%

用法:
    python scripts/rebuild_limit.py            # 全量重建
    python scripts/rebuild_limit.py --verify   # 重建后校验
"""
import sys, io, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

import argparse
import numpy as np
import pandas as pd

from database.config import FROZEN_ROOT, dir_of
from database.limit_rules import (BEIJING_PREFIXES, apply_limit_prices,
                                  base_pct as _base_pct,
                                  load_st_intervals as _load_st_intervals)

SRC = dir_of("daily")             # db/cleaned/daily_basic
DST = dir_of("limit")             # db/cleaned/limit_price
ST_DIR = FROZEN_ROOT / "st"       # db/frozen/st（只读）


def base_pct(code: str) -> float:
    """兼容旧调用：板块基础涨跌幅（唯一实现在 database/limit_rules.py）"""
    return _base_pct(code)


def load_st_intervals():
    """兼容旧调用：ST 区间（唯一实现在 database/limit_rules.py）"""
    return _load_st_intervals(ST_DIR)


def rebuild():
    if not SRC.exists():
        print(f"源目录不存在: {SRC}")
        return 1
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.glob("year=*/*.parquet"))
    st_map = load_st_intervals()
    print(f"源: {SRC}", flush=True)
    print(f"目标: {DST}", flush=True)
    print(f"ST 区间表: {len(st_map):,} 只股票带 ST 记录", flush=True)
    print(f"重建涨跌停价: {len(files):,} 文件", flush=True)

    total = 0
    for i, f in enumerate(files):
        code = f.stem
        year = f.parent.name.split("=")[1]
        df = pd.read_parquet(f)
        if df.empty or "pre_close" not in df.columns:
            continue

        d = df[["code", "trade_date", "pre_close"]].copy()
        d["trade_date"] = pd.to_datetime(d["trade_date"])
        d = d.sort_values("trade_date").reset_index(drop=True)

        # 交易所口径：涨跌停价按昨收（除权后参考价）× (1 ± 幅度)，四舍五入到分。
        # 规则（含 ST 5%、北交所 30%、创业板 2020-08-24 前 10%）统一在
        # database/limit_rules.py，这里不再自己实现一遍。
        out = apply_limit_prices(d, code, st_map)
        out = out[["code", "trade_date", "pre_close", "limit_up", "limit_down"]]

        part = DST / f"year={year}"
        part.mkdir(parents=True, exist_ok=True)
        out.to_parquet(part / f"{code}.parquet", index=False)
        total += len(out)
        if (i + 1) % 10000 == 0 or i + 1 == len(files):
            print(f"  进度 {i+1:,}/{len(files):,}", flush=True)

    print(f"完成: {len(files):,} 文件, {total:,} 行", flush=True)

    # 记录生成信息
    from database.provenance import stamp_dataset
    stamp_dataset("limit", {
        "recipe": "limit_price",
        "source": "cleaned/daily_basic (tushare 官方 pre_close，已按除权调整)",
        "rules": "科创板/创业板±20%, 北交所±30%, 主板ST±5%, 主板±10%",
        "script": "scripts/rebuild_limit.py",
        "rows": total,
    })
    print(f"已写入 {DST / '_meta.json'}")
    return 0


def verify():
    import duckdb
    from database.config import parquet_glob
    g = parquet_glob(DST)
    con = duckdb.connect()
    print("\n=== 涨跌停价校验 ===")
    r = con.execute(f"""
        SELECT count(*),
               sum(CASE WHEN limit_up <= limit_down THEN 1 ELSE 0 END),
               sum(CASE WHEN limit_up IS NULL OR limit_down IS NULL THEN 1 ELSE 0 END),
               sum(CASE WHEN pre_close IS NULL OR pre_close <= 0 THEN 1 ELSE 0 END)
        FROM read_parquet('{g}')
    """).fetchone()
    print(f"总行数: {r[0]:,}")
    print(f"limit_up<=limit_down 的异常行: {r[1]}")
    print(f"涨跌停价缺失行            : {r[2]}")
    print(f"pre_close 非法行          : {r[3]}")
    # 与日线交叉校验：真实成交价必须落在 [limit_down, limit_up] 区间内
    dg = parquet_glob(SRC)
    x = con.execute(f"""
        WITH d AS (SELECT code, trade_date, low, high FROM read_parquet('{dg}')),
             l AS (SELECT code, trade_date, limit_up, limit_down FROM read_parquet('{g}'))
        SELECT count(*) AS n,
               sum(CASE WHEN d.high > l.limit_up + 0.011 THEN 1 ELSE 0 END) AS over_up,
               sum(CASE WHEN d.low  < l.limit_down - 0.011 THEN 1 ELSE 0 END) AS under_dn
        FROM d JOIN l USING (code, trade_date)
    """).fetchone()
    print(f"\n与日线交叉校验（{x[0]:,} 个配对）:")
    print(f"  最高价 > 涨停价的异常行: {x[1]:,}")
    print(f"  最低价 < 跌停价的异常行: {x[2]:,}")
    ok = r[1] == 0 and r[2] == 0 and r[3] == 0 and x[1] == 0 and x[2] == 0
    print("结果:", "全部通过" if ok else "存在问题（见上）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    rc = rebuild()
    if rc == 0 and args.verify:
        verify()
    sys.exit(rc)
