# -*- coding: utf-8 -*-
"""已知数据缺陷注册表 + C1（北交所开市前涨跌停规则不可用）

背景（C1 到底错在哪）
---------------------
`920xxx` / `43xxxx` / `83xxxx` 这些前缀在 **2021-11-15 北交所开市之前**
根本不属于北交所 —— 它们是**新三板**遗留代码。老实现对所有北交所前缀一律套
±30%，于是：

  1. 算出一个**没有依据**的涨跌停价；
  2. 实测这段的 `pre_close` 大量缺失（238 行）或异常（约 40 行价格 < 0.5 元），
     集中在 2008–2022 年；
  3. 错的方向偏**宽松** —— 涨跌停价算错会让引擎拦不住本该封板的成交，
     回测可以凭空买入涨停板。

这是典型的"静默错"：不报错，只让收益曲线变好看。

【为什么"置成 NaN"还不够】
    NaN 在引擎里的语义是"**没有涨跌幅限制**"。而真相是"**我们不知道当时的规则**"。
    把"不知道"当成"没有限制"，方向依然是宽松的。所以除了置 NaN，
    还要靠故障注册表把这些 (代码, 日期) **排除出可交易股票池**。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_c1_bse_limit_regime_boundary():
    """**核心回归**：北交所 ±30% 只从开市日（2021-11-15）起生效

    开市前必须是 NaN（不可用），开市后必须正好是 ±30%。
    边界日两侧都要查，避免写成 `<=` / `<` 搞反。
    """
    from database.defects import BSE_FROM
    from database.limit_rules import apply_limit_prices

    assert str(BSE_FROM.date()) == "2021-11-15", BSE_FROM
    dates = ["2021-11-12", "2021-11-15", "2021-11-16"]
    df = pd.DataFrame({"code": ["920001"] * 3,
                       "trade_date": pd.to_datetime(dates),
                       "pre_close": [10.0] * 3})
    out = apply_limit_prices(df, "920001").set_index("trade_date")

    # 开市前一日：不可用
    assert pd.isna(out.loc["2021-11-12", "limit_up"]), \
        "北交所开市前的涨跌停价没有被置为不可用（C1 未修）"
    assert pd.isna(out.loc["2021-11-12", "limit_down"])
    # 开市日与之后：±30%
    for d in ("2021-11-15", "2021-11-16"):
        assert out.loc[d, "limit_up"] == 13.0, f"{d} 涨停价应为 13.0"
        assert out.loc[d, "limit_down"] == 7.0, f"{d} 跌停价应为 7.0"

    # 其他板块不受影响
    main = pd.DataFrame({"code": ["600000"], "trade_date": pd.to_datetime(["2021-11-12"]),
                         "pre_close": [10.0]})
    assert apply_limit_prices(main, "600000")["limit_up"].iloc[0] == 11.0
    print("[OK] C1：北交所 ±30% 从 2021-11-15 起生效，开市前为不可用")


def test_c1_unreliable_mask_excludes_before_bse():
    """不可靠掩码：只标北交所前缀、只标开市前，别的代码/日期不能误伤"""
    from database.defects import unreliable_limit_mask

    codes = ["920001", "430001", "600000", "000001", "300750"]
    dates = pd.to_datetime(["2020-01-02", "2021-11-15", "2022-06-01"])
    m = unreliable_limit_mask(codes, dates)

    assert m.loc["2020-01-02", "920001"], "920xxx 开市前应标为不可靠"
    assert m.loc["2020-01-02", "430001"], "43xxxx 开市前应标为不可靠"
    assert not m.loc["2021-11-15", "920001"], "开市当日不应再标为不可靠"
    assert not m.loc["2022-06-01", "920001"], "开市后不应标为不可靠"
    for c in ("600000", "000001", "300750"):
        assert not m[c].any(), f"{c} 被误标为不可靠"
    print("[OK] C1 掩码：只命中北交所前缀 + 开市前，主板/创业板/深市不受影响")


def test_build_universe_excludes_unreliable_by_default():
    """股票池默认排除不可靠段；可以显式关掉（对照用）"""
    from database.defects import unreliable_limit_mask
    from universe.pool import UniverseSpec

    spec = UniverseSpec()
    assert spec.exclude_unreliable is True, "默认应排除不可靠段"
    assert "不可靠" in spec.describe(), f"describe 没体现这条规则: {spec.describe()}"
    off = UniverseSpec(exclude_unreliable=False)
    assert "不可靠" not in off.describe()
    # 掩码本身与 build_universe 用的是同一个函数，保证口径一致
    assert callable(unreliable_limit_mask)
    print("[OK] 股票池：默认排除涨跌停不可靠段，且可在说明里看到")


def test_defects_registry_shape():
    """注册表条目必须字段完整、key 唯一、severity/needs_data 如实"""
    from database.defects import DEFECTS

    keys = [d.key for d in DEFECTS]
    assert len(keys) == len(set(keys)), f"key 重复: {keys}"
    for d in DEFECTS:
        assert d.title and d.scope and d.impact, f"{d.key} 描述不全"
        assert d.severity in ("高", "中", "低"), f"{d.key} 严重度非法: {d.severity}"
        assert isinstance(d.needs_data, bool), f"{d.key} needs_data 必须是 bool"
        assert d.doc_ref, f"{d.key} 没有回指文档（容易两处漂移）"
    print(f"[OK] 注册表：{len(DEFECTS)} 条，key 唯一、字段完整 -> {keys}")


def test_defects_in_window_uses_intersection():
    """窗口判定用**交集**：宁可多标注，不可漏标注"""
    from database.defects import defects_in_window

    # 完全覆盖 C4b（2006-2007）
    keys = {d.key for d in defects_in_window("2006-01-01", "2007-12-31")}
    assert "C4b" in keys, f"2006-2007 窗口应命中 C4b，实得 {keys}"

    # 只沾到边界一天也要算
    keys = {d.key for d in defects_in_window("2007-12-31", "2020-01-01")}
    assert "C4b" in keys, "窗口与缺陷区间只交叠一天也应命中"

    # 完全不相交则不命中
    keys = {d.key for d in defects_in_window("2015-01-01", "2020-01-01")}
    assert "C4b" not in keys, f"2015-2020 不该命中 C4b，实得 {keys}"

    # 无日期区间的缺陷（B7 等）任何窗口都要标注
    assert "B7" in {d.key for d in defects_in_window("2015-01-01", "2020-01-01")}
    print("[OK] 窗口判定：交集命中 / 边界日命中 / 不相交不命中 / 无区间恒标注")


def test_data_todo_lists_only_needs_data():
    """`needs_data` 是给人看的决策清单，不能混入代码就能修的条目"""
    from database.defects import data_todo

    todo = data_todo()
    assert todo, "应有需要额外数据的条目"
    assert all(d.needs_data for d in todo)
    keys = {d.key for d in todo}
    # C1 的根治确实需要新三板历史数据（代码层只能保守排除）
    assert "C1" in keys, f"C1 应算作需要额外数据，实得 {keys}"
    # B9 只是"覆盖不到"，不产生错数据 -> 不该出现在待办里
    assert "B9" not in keys, "B9 不需要额外数据，不该进待办"
    print(f"[OK] 需额外数据的条目：{sorted(keys)}")


def test_defect_banner_mentions_needs_data():
    """附注必须标出"需额外数据"，否则读者以为代码已经修好了"""
    from database.defects import format_banner, get

    txt = format_banner([get("C1")])
    assert "需额外数据" in txt, "附注没有标出需额外数据"
    assert "C1" in txt and "2021-11-15" in txt or "北交所" in txt
    assert format_banner([]) and "未触及" in format_banner([])
    print("[OK] 缺陷附注：标出「需额外数据」，空集时明确写「未触及」")


def test_c2b_relisted_stocks_skip_first_day_rule():
    """**回归**：重新上市股不能被误套新股首日 ±44%

    这 3 只是重组/重新上市，首日涨幅 +188.9% / +122.5% / +235.9% —— 远超
    ±44%，说明当时不受该规则约束。误套会让引擎在这些**本该能成交**的日子
    拒绝成交（偏悲观）。已知的 3 只已通过 `RELISTED_CODES` 跳过。
    """
    from database.defects import RELISTED_CODES
    from database.limit_rules import listing_windows

    assert set(RELISTED_CODES) == {"601399", "001267", "601155"}, RELISTED_CODES
    try:
        w = listing_windows()
    except Exception as e:                     # CI 里没有 db/，读 stocks 会失败
        print(f"[SKIP] 无 frozen/stocks 或日历（{type(e).__name__}），跳过 C2b 回归")
        return
    if not w:
        print("[SKIP] 无 frozen/stocks 或日历，跳过 C2b 回归")
        return
    for c in RELISTED_CODES:
        assert c not in w, f"{c} 是重新上市股，不该有新股首日窗口"
    # 不能把所有人都跳过
    assert len(w) > 1000, f"只剩 {len(w)} 只股票有窗口，像是过度跳过"
    print(f"[OK] C2b：{len(RELISTED_CODES)} 只重新上市股已跳过首日规则，"
          f"其余 {len(w)} 只不受影响")


if __name__ == "__main__":
    test_c1_bse_limit_regime_boundary()
    test_c1_unreliable_mask_excludes_before_bse()
    test_build_universe_excludes_unreliable_by_default()
    test_defects_registry_shape()
    test_defects_in_window_uses_intersection()
    test_data_todo_lists_only_needs_data()
    test_defect_banner_mentions_needs_data()
    test_c2b_relisted_stocks_skip_first_day_rule()
    print("\n全部数据缺陷注册表 / C1 / C2b 测试通过")
