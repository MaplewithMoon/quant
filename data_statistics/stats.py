# -*- coding: utf-8 -*-
"""股票统计：最高价、最低价、涨跌幅、多股对比表格"""
import sys
import io
import os
import contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __name__ == "__main__":
    # 只在直接运行时切编码：模块顶层替换 sys.stdout 是有副作用的 import，
    # 会破坏 pytest 的输出捕获（详见 utils/console.py）
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

import pandas as pd
from data.dataset import DataSet
from data.source import RemoteSource, MockSource
from datetime import datetime, timedelta


def _stock_name(symbol: str, bs_session=None) -> str:
    if bs_session:
        try:
            code = f"sz.{symbol}" if symbol.startswith(("0", "3")) else f"sh.{symbol}"
            rs = bs_session.query_stock_basic(code)
            d = rs.get_data()
            if d is not None and not d.empty:
                return d.iloc[0]["code_name"]
        except Exception:
            pass
        return symbol

    with open(os.devnull, "w", encoding="utf-8") as dn:
        with contextlib.redirect_stdout(dn):
            try:
                import baostock as bs
                lg = bs.login()
                if lg.error_code != "0":
                    return symbol
                try:
                    code = f"sz.{symbol}" if symbol.startswith(("0", "3")) else f"sh.{symbol}"
                    rs = bs.query_stock_basic(code)
                    d = rs.get_data()
                    if d is not None and not d.empty:
                        return d.iloc[0]["code_name"]
                finally:
                    bs.logout()
            except Exception:
                pass
    return symbol


def _batch_fetch(symbols: list, start: str, end: str):
    """一次登录，批量拉取多只股票数据和名称"""
    import baostock as bs
    with open(os.devnull, "w", encoding="utf-8") as dn:
        with contextlib.redirect_stdout(dn):
            lg = bs.login()
            if lg.error_code != "0":
                raise ConnectionError("baostock login failed")

    try:
        names = {}
        for sym in symbols:
            names[sym] = _stock_name(sym, bs)

        dfs = {}
        for sym in symbols:
            code = f"sz.{sym}" if sym.startswith(("0", "3")) else f"sh.{sym}"
            rs = bs.query_history_k_data_plus(
                code, "date,open,high,low,close,volume,amount",
                start_date=start, end_date=end, frequency="d", adjustflag="2",
            )
            d = rs.get_data()
            if d is not None and not d.empty:
                d.rename(columns={"date": "datetime"}, inplace=True)
                d["datetime"] = pd.to_datetime(d["datetime"])
                d.set_index("datetime", inplace=True)
                for c in ["open", "high", "low", "close", "volume", "amount"]:
                    d[c] = d[c].astype(float)
                d.sort_index(inplace=True)
                dfs[sym] = DataSet(symbol=sym, data=d)
    finally:
        with open(os.devnull, "w", encoding="utf-8") as dn:
            with contextlib.redirect_stdout(dn):
                bs.logout()

    return names, dfs


def _fetch(symbol: str, start: str, end: str):
    backends = [("baostock", RemoteSource), ("akshare", RemoteSource), ("mock", MockSource)]
    for name, cls in backends:
        try:
            kwargs = {"backend": name} if name != "mock" else {}
            return DataSet.load(symbol, start, end, source=cls(**kwargs))
        except Exception:
            continue
    return None


def _disp_width(s) -> int:
    """字符串在终端中的显示宽度（中文占2，英文/数字占1）"""
    n = 0
    for c in str(s):
        if '\u4e00' <= c <= '\u9fff':
            n += 2
        else:
            n += 1
    return n


def _pad(s, width: int, align: str = "<") -> str:
    """按显示宽度填充字符串到指定宽度"""
    s = str(s)
    pad = width - _disp_width(s)
    if pad <= 0:
        return s
    return (" " * pad) + s if align == ">" else s + (" " * pad)


def _calc_stats(ds: DataSet):
    df = ds.data
    high = df["high"].max()
    low = df["low"].min()
    return {
        "最高价": round(high, 2),
        "最低价": round(low, 2),
        "涨跌幅": round((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 2),
        "最大相对跌幅": round((low / high - 1) * 100, 2),
        "交易日": len(df),
    }


def get_monthly_stats(symbol: str, months: int = 1,
                      start: str = "", end: str = ""):
    if not start or not end:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=months * 31)).strftime("%Y-%m-%d")

    ds = _fetch(symbol, start, end)
    if ds is None or len(ds) == 0:
        print(f"无法获取 {symbol} 的数据")
        return

    name = _stock_name(symbol)
    s = _calc_stats(ds)
    df = ds.data

    print(f"股票: {symbol}  {name}")
    print(f"区间: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"交易日: {s['交易日']} 天")
    print(f"最高价: {s['最高价']:.2f}")
    print(f"最低价: {s['最低价']:.2f}")
    print(f"涨跌幅: {s['涨跌幅']:+.2f}%")
    print(f"最大相对跌幅: {s['最大相对跌幅']:.2f}%")


def compare_stocks(symbols: list, start: str = "", end: str = "",
                   months: int = 1):
    if not start or not end:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=months * 31)).strftime("%Y-%m-%d")

    rows = []
    names_all, dfs = _batch_fetch(symbols, start, end)
    for sym in symbols:
        ds = dfs.get(sym)
        if ds is None or len(ds) == 0:
            rows.append([sym, names_all.get(sym, "N/A"), 0, 0, 0, 0, 0])
            continue
        s = _calc_stats(ds)
        rows.append([sym, names_all.get(sym, sym), s["最高价"], s["最低价"], s["涨跌幅"], s["最大相对跌幅"], s["交易日"]])

    headers = ["代码", "名称", "最高价", "最低价", "涨跌幅%", "最大相对跌幅%", "交易日"]
    align = ["<", "<", ">", ">", ">", ">", "<"]
    col_widths = [_disp_width(h) for h in headers]
    for r in rows:
        col_widths[0] = max(col_widths[0], _disp_width(r[0]))
        col_widths[1] = max(col_widths[1], _disp_width(r[1]))
        col_widths[2] = max(col_widths[2], _disp_width(f"{r[2]:.2f}"))
        col_widths[3] = max(col_widths[3], _disp_width(f"{r[3]:.2f}"))
        col_widths[4] = max(col_widths[4], _disp_width(f"{r[4]:+.2f}"))
        col_widths[5] = max(col_widths[5], _disp_width(f"{r[5]:.2f}"))
        col_widths[6] = max(col_widths[6], _disp_width(str(r[6])))
    gap = 1

    print(f"\n统计区间: {start} ~ {end}\n")
    sep = "-" * (sum(col_widths) + gap * (len(col_widths) - 1))
    print((" " * gap).join(_pad(h, col_widths[i], align[i]) for i, h in enumerate(headers)))
    print(sep)
    for r in rows:
        vals = [
            _pad(r[0], col_widths[0], "<"),
            _pad(r[1], col_widths[1], "<"),
            _pad(f"{r[2]:.2f}", col_widths[2], ">"),
            _pad(f"{r[3]:.2f}", col_widths[3], ">"),
            _pad(f"{r[4]:+.2f}", col_widths[4], ">"),
            _pad(f"{r[5]:.2f}", col_widths[5], ">"),
            _pad(str(r[6]), col_widths[6], "<"),
        ]
        print((" " * gap).join(vals))
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="股票统计")
    parser.add_argument("symbols", nargs="*", default=["000001"],
                        help="股票代码（多个用空格隔开）")
    parser.add_argument("--months", type=int, default=1, help="回溯月数")
    parser.add_argument("--start", default="", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--table", action="store_true", help="以表格模式输出多股对比")
    args = parser.parse_args()

    if args.table or len(args.symbols) > 1:
        compare_stocks(args.symbols, args.start, args.end, args.months)
    else:
        get_monthly_stats(args.symbols[0], args.months, args.start, args.end)
