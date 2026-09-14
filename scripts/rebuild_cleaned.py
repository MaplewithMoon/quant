# -*- coding: utf-8 -*-
"""基于 frozen 全量重建清洗层

- 读取 db/frozen/daily_raw（tushare 不复权原始价，可信源，只读）
- 清洗: 删幽灵K线(vol=0/NaN)、价格<=0、OHLC违规、VWAP三角违规
- 输出 db/cleaned/daily_basic（配方 daily_basic，按年/代码分区）
- frozen 层只读，本脚本不修改它

用法:
    python scripts/rebuild_cleaned.py            # 全量重建
    python scripts/rebuild_cleaned.py --verify   # 重建后校验
"""
import sys, io, os, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

import argparse
import pandas as pd

from database.config import FROZEN_ROOT, dir_of, parquet_glob

SRC = FROZEN_ROOT / "daily_raw"     # db/frozen/daily_raw
DST = dir_of("daily")               # db/cleaned/daily_basic


def clean_rows(df):
    """清洗单文件行: 返回(干净df, 删除行数)"""
    if df.empty:
        return df, 0
    bad = pd.Series(False, index=df.index)
    # 幽灵K线
    vol = pd.to_numeric(df["volume"], errors="coerce")
    bad |= vol.isna() | (vol == 0)
    # 价格
    for c in ["open", "high", "low", "close"]:
        v = pd.to_numeric(df[c], errors="coerce")
        bad |= v.isna() | (v <= 0)
    # OHLC
    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    cl = pd.to_numeric(df["close"], errors="coerce")
    valid = o.notna() & h.notna() & l.notna() & cl.notna()
    bad |= valid & ~(h >= l)
    bad |= valid & ~(l <= o)
    bad |= valid & ~(o <= h)
    bad |= valid & ~(l <= cl)
    bad |= valid & ~(cl <= h)
    # 三角校验（VWAP 落在 [low,high]）
    amt = pd.to_numeric(df["amount"], errors="coerce")
    vwap = amt / vol.replace(0, pd.NA)
    vok = vwap.notna() & valid
    bad |= vok & ~((vwap >= l * 0.95) & (vwap <= h * 1.05))
    return df[~bad], int(bad.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if not SRC.exists():
        print(f"源目录不存在: {SRC}", flush=True)
        return 1

    # 清空旧清洗层（只删 cleaned/ 下的配方目录，不碰 frozen）
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.glob("year=*/*.parquet"))
    total_removed = 0
    print(f"源: {SRC}")
    print(f"目标: {DST}")
    print(f"重建清洗层: {len(files)} 文件", flush=True)

    for i, f in enumerate(files):
        code = f.stem                       # 000001
        year = f.parent.name.split("=")[1]  # year=2024 -> 2024
        df = pd.read_parquet(f)
        clean, removed = clean_rows(df)
        total_removed += removed
        if clean.empty:
            continue
        part = DST / f"year={year}"
        part.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(part / f"{code}.parquet", index=False)
        if (i + 1) % 10000 == 0 or i + 1 == len(files):
            print(f"  进度 {i+1}/{len(files)}, 已清理 {total_removed} 行", flush=True)

    print(f"清洗层重建完成: {len(files)} 文件, 清理 {total_removed} 行", flush=True)

    # 写入时间戳/来源元数据（每次重建都会刷新）
    from database.provenance import stamp_cleaned
    stamp_cleaned(meta={"source": "frozen/daily_raw (tushare不复权)",
                        "cleaning_rules": "删幽灵K线/价格<=0/OHLC违规/三角违规",
                        "record_count": "见各year文件"})
    print(f"已写入清洗层时间戳 {DST / '_meta.json'}", flush=True)

    if args.verify:
        verify()
    return 0


def verify():
    import duckdb
    con = duckdb.connect()
    g = parquet_glob(DST)
    print("\n=== 清洗层校验 ===")
    print(f"glob: {g}")
    r = con.execute(f"""
        SELECT count(*),
               sum(CASE WHEN close<=0 THEN 1 ELSE 0 END),
               sum(CASE WHEN NOT (high>=low AND low<=open AND open<=high AND low<=close AND close<=high) THEN 1 ELSE 0 END),
               sum(CASE WHEN volume=0 OR volume IS NULL THEN 1 ELSE 0 END)
        FROM read_parquet('{g}')
    """).fetchone()
    print(f"总行数: {r[0]:,}")
    print(f"价格<=0: {r[1]}, OHLC违规: {r[2]}, 幽灵K线: {r[3]}")
    # 三角校验
    r2 = con.execute(f"""
        WITH t AS (SELECT amount/volume AS vwap, low, high
                   FROM read_parquet('{g}') WHERE volume>0)
        SELECT count(*), sum(CASE WHEN NOT (vwap>=low*0.95 AND vwap<=high*1.05) THEN 1 ELSE 0 END) FROM t
    """).fetchone()
    print(f"三角校验: 总{r2[0]:,}, 违规:{r2[1]}")
    ok = r[1] == 0 and r[2] == 0 and r[3] == 0 and r2[1] == 0
    print("结果:", "全部通过" if ok else "存在问题!")


if __name__ == "__main__":
    sys.exit(main())
