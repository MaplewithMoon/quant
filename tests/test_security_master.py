# -*- coding: utf-8 -*-
"""证券主表与时点正确可交易池测试

背景（为什么以前的股票池是错的）
-------------------------------
股票池是回测的地基，而以前这个地基是拿**当前快照**拼的：

  | 环节 | 旧做法 | 错在哪 |
  |---|---|---|
  | 判断"是否退市" | `name.str.contains("退")` | 既漏（改名/重组退市的不带"退"）又错（退市整理期带"退"但可交易） |
  | 判断"是否北交所/新三板" | **代码前缀** | 前缀是今天的编码约定，不是"当时挂在哪个交易所" |
  | 判断"该不该交易" | `close.notna()` | "有没有行情" ≠ "该不该交易"，交易所归属它根本表达不了 |

后果是**幸存者偏差**与**不可交易标的混入**，两个方向都会让回测结果失真。

正确做法是证券主表三层过滤（全是日期比较，不含"今天"）：
    ① exchange ∈ (SSE, SZSE)
    ② list_date <= t
    ③ delist_date 为空 或 delist_date > t

【本文件盯住的关键点】
  1. 退市边界必须**严格**：退市当日即不可交易（`delist_date > t`）。
  2. 退市股在退市**之前**必须可交易 —— 否则就是幸存者偏差。
  3. 主表缺字段时**不能静默放行**（那会悄悄引入偏差），要显式报出来。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _has_master() -> bool:
    try:
        from database.master import master_capabilities
        return bool(master_capabilities()["has_master"])
    except Exception:
        return False


def _fake_master() -> pd.DataFrame:
    """合成主表：覆盖各种边界"""
    return pd.DataFrame({
        "code": ["000001", "600000", "000002", "900001", "300750"],
        "name": ["甲的银行", "乙的银行", "退市股", "北交所股", "创业板股"],
        "exchange": ["SZSE", "SSE", "SZSE", "BSE", "SZSE"],
        "list_date": pd.to_datetime(["1991-04-03", "1999-11-10",
                                     "1990-12-10", "2021-11-15", "2018-06-11"]),
        "delist_date": pd.to_datetime([None, None, "2024-04-26", None, None]),
        "list_status": ["L", "L", "D", "L", "L"],
    })


# ============================================================
# 一、三层过滤（纯合成，不依赖真实主表）
# ============================================================
def test_delist_boundary_is_strict():
    """**核心回归**：退市当日即不可交易（delist_date > t，不是 >=）

    差一天就会让退市整理期最后一天变成"可交易"，或者反过来把退市前一天
    错杀掉。这条边界必须逐日钉死。
    """
    from database.master import tradable_mask

    m = _fake_master()
    dates = pd.to_datetime(["2024-04-24", "2024-04-25", "2024-04-26",
                            "2024-04-27"])
    mask = tradable_mask(dates, m["code"], master=m)
    col = "000002"
    assert bool(mask.loc["2024-04-25", col]) is True, "退市前一日应可交易"
    assert bool(mask.loc["2024-04-26", col]) is False, "退市当日就不可交易"
    assert bool(mask.loc["2024-04-27", col]) is False, "退市后不可交易"
    print("[OK] 退市边界严格：前一日可交易 / 当日及以后不可交易")


def test_delisted_stock_present_before_delist():
    """**幸存者偏差回归**：已退市股票在退市**之前**必须在池子里"""
    from database.master import tradable_mask

    m = _fake_master()
    dates = pd.to_datetime(["2005-06-30", "2023-06-30"])
    mask = tradable_mask(dates, ["000002"], master=m)
    assert bool(mask.loc["2005-06-30", "000002"]), \
        "退市股在 2005 年不在池子里 —— 幸存者偏差（历史收益会被高估）"
    assert bool(mask.loc["2023-06-30", "000002"])
    print("[OK] 退市股在退市前可交易（无幸存者偏差）")


def test_list_boundary():
    """上市前不可交易；上市首日可交易"""
    from database.master import tradable_mask

    m = _fake_master()
    dates = pd.to_datetime(["2018-06-10", "2018-06-11", "2018-06-12"])
    mask = tradable_mask(dates, ["300750"], master=m)
    assert bool(mask.loc["2018-06-10", "300750"]) is False, "上市前一日不该可交易"
    assert bool(mask.loc["2018-06-11", "300750"]) is True, "上市首日应可交易"
    assert bool(mask.loc["2018-06-12", "300750"]) is True
    print("[OK] 上市边界：上市前不可交易 / 首日起可交易")


def test_exchange_filter_excludes_bse():
    """**C1 的根治**：交易所过滤天然排除北交所/新三板（不靠代码前缀）"""
    from database.master import tradable_mask

    m = _fake_master()
    dates = pd.to_datetime(["2022-06-30"])
    only_sh_sz = tradable_mask(dates, m["code"], exchanges=("SSE", "SZSE"),
                               master=m)
    assert bool(only_sh_sz.loc["2022-06-30", "900001"]) is False, \
        "北交所标的没被交易所过滤排除"
    assert bool(only_sh_sz.loc["2022-06-30", "000001"]) is True

    # 显式纳入 BSE 时应可见（可配置，不是写死）
    with_bse = tradable_mask(dates, m["code"], exchanges=("SSE", "SZSE", "BSE"),
                             master=m)
    assert bool(with_bse.loc["2022-06-30", "900001"]) is True, \
        "显式纳入 BSE 后仍取不到，说明过滤写死了"
    # 空 exchanges 表示不过滤交易所
    no_ex = tradable_mask(dates, m["code"], exchanges=(), master=m)
    assert bool(no_ex.loc["2022-06-30", "900001"]) is True
    print("[OK] 交易所过滤：默认排除 BSE，可显式纳入（不是写死）")


def test_unknown_code_is_not_tradable():
    """主表里没有的代码一律 False —— "不知道"不等于"可以交易" """
    from database.master import tradable_mask

    m = _fake_master()
    dates = pd.to_datetime(["2020-06-30"])
    mask = tradable_mask(dates, ["999999"], master=m)
    assert bool(mask.loc["2020-06-30", "999999"]) is False, \
        "主表查不到的代码被放行了（应当宁可不交易）"
    print("[OK] 主表查不到的代码 -> 不可交易（不知道 ≠ 可以）")


def test_master_capabilities_detects_missing_fields():
    """**关键**：主表缺 exchange/delist_date 时必须报出来，不能静默放行

    旧版 `all.parquet` 只有 8 列（没有这两个字段），三层过滤做不了。
    如果那时静默放行，就会悄悄引入幸存者偏差 —— 比报错糟糕得多。
    """
    from database.master import master_capabilities

    old = pd.DataFrame({"code": ["000001"], "list_date": pd.to_datetime(["1991-04-03"])})
    cap = master_capabilities(old)
    assert cap["has_master"] is True
    assert cap["exchange"] is False and cap["delist"] is False
    assert cap["pit_ready"] is False, "缺字段却报告 pit_ready"

    full = _fake_master()
    cap2 = master_capabilities(full)
    assert cap2["pit_ready"] is True, f"字段齐全却报 pit_ready=False: {cap2}"
    print(f"[OK] 主表能力探测：旧表 pit_ready=False，新表 True（n={cap2['n']}）")


def test_live_codes_use_delist_date_not_name():
    """`live_codes` 必须用 delist_date，不能用"名字含退"

    合成一只"名字带『退』但仍在上市"的股票：按名字判据会被错误剔除，
    按 delist_date 判据则应保留。
    """
    from database.master import live_codes

    m = pd.DataFrame({
        "code": ["000777", "600888"],
        "name": ["某某退（其实还在上市）", "正常股"],
        "exchange": ["SZSE", "SSE"],
        "list_date": pd.to_datetime(["2000-01-01", "2000-01-01"]),
        "delist_date": pd.to_datetime([None, "2019-01-01"]),
    })
    out = live_codes(master=m)
    assert "000777" in out, "名字带『退』但未退市的股票被误剔除了"
    assert "600888" not in out, "已退市股票出现在 live_codes 里"
    print("[OK] live_codes 用 delist_date 判据（名字带『退』但未退市者保留）")


# ============================================================
# 二、真实主表
# ============================================================
def test_real_master_is_pit_ready():
    """真实主表应当字段齐全，且三种状态都拉到（避免幸存者偏差）"""
    if not _has_master():
        print("[SKIP] 无 frozen/stocks")
        return
    from database.master import (delisted_codes, load_master,
                                 master_capabilities)
    m = load_master()
    cap = master_capabilities(m)
    if not cap["pit_ready"]:
        print(f"[SKIP] 主表尚未补齐字段（{cap}）—— 需重跑 StockListDownloader")
        return
    assert "BSE" in set(m["exchange"].dropna()), "exchange 里没有 BSE？"
    n_del = len(delisted_codes())
    assert n_del > 100, f"只识别出 {n_del} 只退市股，疑似没拉 list_status='D'"
    print(f"[OK] 真实主表：{cap['n']:,} 只，退市 {n_del} 只，"
          f"交易所 {dict(m['exchange'].value_counts())}")


def test_real_pool_excludes_bse_by_default():
    """真实数据下：默认池不含 BSE，且退市股在退市前仍在池中"""
    if not _has_master():
        print("[SKIP] 无 frozen/stocks")
        return
    from database.master import load_master, master_capabilities, tradable_at
    if not master_capabilities()["pit_ready"]:
        print("[SKIP] 主表字段未补齐")
        return
    m = load_master()
    bse = m.loc[m["exchange"] == "BSE", "code"].tolist()
    if bse:
        got = tradable_at("2022-06-30", bse)
        assert not got, f"默认池里出现了 {len(got)} 只 BSE 标的"
    # 退市股在退市前必须在池中
    d = m[m["delist_date"].notna()]
    if not d.empty:
        row = d.sort_values("delist_date").iloc[0]
        before = row["delist_date"] - pd.Timedelta(days=30)
        assert row["code"] in tradable_at(before, [row["code"]]), \
            f"{row['code']} 在退市前 30 天不在池中（幸存者偏差）"
    print(f"[OK] 真实池：{len(bse)} 只 BSE 默认排除；退市股退市前在池中")


if __name__ == "__main__":
    test_delist_boundary_is_strict()
    test_delisted_stock_present_before_delist()
    test_list_boundary()
    test_exchange_filter_excludes_bse()
    test_unknown_code_is_not_tradable()
    test_master_capabilities_detects_missing_fields()
    test_live_codes_use_delist_date_not_name()
    test_real_master_is_pit_ready()
    test_real_pool_excludes_bse_by_default()
    print("\n全部证券主表 / 可交易池测试通过")
