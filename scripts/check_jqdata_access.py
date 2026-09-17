# -*- coding: utf-8 -*-
"""聚宽 JQData 访问权限试跑（**由你自己在本机运行，不需要把密码给别人**）

背景
----
B10 需要「指数成分的**历史时点**」（带 in_date/out_date）。已核实：
  - 中证官网只公开**最新**成分快照，文件每天被覆盖，历史补不回来；
  - tushare 的 `index_member` 是**申万行业成分**，不是中证指数成分；
  - 聚宽 `get_index_stocks(index_symbol, date=...)` **源码确认支持历史日期**，
    但它带 `@assert_auth`，需要账号认证。

关键未知项：**聚宽账号 ≠ JQData 权限**。JQData 是独立的数据服务，
普通回测账号未必有配额。所以先做一次最小调用探明，再决定要不要批量导出。

用法
----
    # 方式 1：环境变量（推荐）
    [Environment]::SetEnvironmentVariable('JQDATA_USER','你的手机号','Machine')
    [Environment]::SetEnvironmentVariable('JQDATA_PASSWORD','你的密码','Machine')

    # 方式 2：本地文件 database/jqdata_account.txt（已在 .gitignore）
    #   user=你的手机号
    #   password=你的密码

    python scripts/check_jqdata_access.py
    python scripts/check_jqdata_access.py --symbol 000905.XSHG --date 2015-01-05

它会依次报告：能否 import、能否 auth、能否取到历史成分、配额还剩多少。
任何一步失败都会打印原始错误，不做猜测。
"""
import argparse
import os
import sys

sys.path.insert(0, ".")
if __name__ == "__main__":
    from utils.console import force_utf8_stdout
    force_utf8_stdout()


def step(n, title):
    print()
    print(f"[{n}] {title}")
    print("-" * 68)


def main():
    ap = argparse.ArgumentParser(description="聚宽 JQData 权限试跑")
    ap.add_argument("--symbol", default="000300.XSHG",
                    help="指数代码（聚宽格式，如 000300.XSHG / 000905.XSHG）")
    ap.add_argument("--date", default="2015-01-05", help="历史日期 YYYY-MM-DD")
    args = ap.parse_args()

    step(1, "凭据")
    from database.token import load_jqdata_account
    user, pwd = load_jqdata_account()
    if not user or not pwd:
        print("  ✗ 未找到凭据。请设置环境变量 JQDATA_USER / JQDATA_PASSWORD，")
        print("    或写 database/jqdata_account.txt（user=... / password=...）")
        return 2
    print(f"  ✓ 读到账号 {user[:3]}***{user[-2:]}（密码长度 {len(pwd)}，不打印内容）")

    step(2, "SDK 是否可用")
    try:
        import jqdatasdk
    except ImportError:
        print("  ✗ 未安装 jqdatasdk。请先安装：")
        print("      python -m pip install jqdatasdk -i "
              "https://pypi.tuna.tsinghua.edu.cn/simple")
        return 3
    print(f"  ✓ jqdatasdk {getattr(jqdatasdk, '__version__', '?')}")

    step(3, "auth（这一步决定账号有没有 JQData 权限）")
    try:
        jqdatasdk.auth(user, pwd)
        print("  ✓ 认证通过")
    except Exception as e:
        print(f"  ✗ 认证失败: {type(e).__name__}: {e}")
        print("    常见原因：账号或密码错 / 该账号未开通 JQData / 试用已到期")
        return 4

    step(4, f"取历史成分 get_index_stocks({args.symbol}, '{args.date}')")
    try:
        codes = jqdatasdk.get_index_stocks(args.symbol, date=args.date)
        print(f"  ✓ 返回 {len(codes)} 只")
        print(f"    前 10 只: {list(codes)[:10]}")
        if not codes:
            print("  ⚠ 返回空 —— 可能是该日期无数据，或指数代码不对")
            return 5
    except Exception as e:
        print(f"  ✗ 失败: {type(e).__name__}: {e}")
        print("    这一步失败说明权限或配额不够（auth 能过不代表能取数）")
        return 5

    step(5, "配额")
    try:
        q = jqdatasdk.get_query_count()
        print(f"  {q}")
    except Exception as e:
        print(f"  （取配额失败，不影响使用: {type(e).__name__}）")

    print()
    print("=" * 68)
    print("结论：账号可用，可以着手做 B10 的历史成分导出。")
    print("下一步：按调整日批量取 000300/000905/000852，拼成 in_date/out_date")
    print("区间表落到 frozen/index_cons，再接进 universe/pool.py。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
