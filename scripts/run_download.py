# -*- coding: utf-8 -*-
"""A股量化数据库主入口（tushare 版）

从新往旧拉取，磁盘剩余<5GB自动停止并记录断点，可随时续跑。

用法:
    python scripts/run_download.py                     # 按优先级全部跑
    python scripts/run_download.py --only daily        # 只跑日线
    python scripts/run_download.py --codes 000001,600519  # 指定股票
    python scripts/run_download.py --fresh             # 忽略断点重新下载
    python scripts/run_download.py --redownload valuation  # 强制重下某数据集
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

import argparse
import inspect
from database.config import setup_database
from database.token import load_token
from database.downloader import (
    CalendarDownloader, StockListDownloader, DailyDownloader,
    AdjustDownloader, DividendDownloader, ValuationDownloader,
    MetaDownloader, StStatusDownloader, SuspendDownloader,
    IndexConsDownloader, IndexDailyDownloader, IndustryDownloader,
    FinancialDownloader, MarginDownloader, NorthboundDownloader,
    HolderDownloader, FundDownloader, FuturesDownloader, OptionDownloader,
)


PRIORITY = {
    "calendar": CalendarDownloader,
    "stocks": StockListDownloader,
    "daily": DailyDownloader,
    "adjust": AdjustDownloader,
    "dividend": DividendDownloader,
    "valuation": ValuationDownloader,
    "meta": MetaDownloader,
    "st": StStatusDownloader,
    "suspend": SuspendDownloader,
    "indices_cons": IndexConsDownloader,
    "indices_daily": IndexDailyDownloader,
    "industry": IndustryDownloader,
    "financial": FinancialDownloader,
    "margin": MarginDownloader,
    "northbound": NorthboundDownloader,
    "holders": HolderDownloader,
    "etf": FundDownloader,
    "futures": FuturesDownloader,
    "options": OptionDownloader,
}


def main():
    parser = argparse.ArgumentParser(description="A股数据库下载")
    parser.add_argument("--only", choices=list(PRIORITY) + ["all"], default="all",
                        help="只跑某个数据集")
    parser.add_argument("--codes", default="", help="指定股票代码，逗号分隔")
    parser.add_argument("--fresh", action="store_true", help="忽略断点重新下载")
    parser.add_argument("--redownload", choices=list(PRIORITY), default="",
                        help="强制重下某数据集（清空断点）")
    args = parser.parse_args()

    if not load_token():
        print("警告: 未找到 Tushare token。请设置环境变量 TUSHARE_TOKEN "
              "或创建 database/tushare_token.txt")

    setup_database()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] or None

    if args.redownload:
        from database.storage import Storage
        Storage(args.redownload).save_checkpoint({})
        print(f"已清空 {args.redownload} 的断点，将强制重新下载")

    order = list(PRIORITY) if args.only == "all" else [args.only]
    for key in order:
        cls = PRIORITY[key]
        print(f"\n{'='*60}\n开始: {key}\n{'='*60}")
        try:
            dl = cls()
            sig = inspect.signature(dl.download)
            params = {}
            if "resume" in sig.parameters:
                params["resume"] = not args.fresh
            if "codes" in sig.parameters:
                params["codes"] = codes
            dl.download(**params)
        except Exception as e:
            print(f"[{key}] 失败: {e}")
            if "不足" in str(e):
                print("磁盘空间不足，已停止。下次运行自动续跑。")
                break
        # 每完成一个数据集检查磁盘
        from database.storage import Storage
        try:
            Storage("calendar").assert_disk_ok()
        except Exception as e:
            print(e)
            print("磁盘空间不足，已停止。下次运行自动续跑。")
            break

    print("\n全部完成（或遇断点停止）")


if __name__ == "__main__":
    main()
