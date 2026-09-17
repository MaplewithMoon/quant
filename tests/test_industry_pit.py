# -*- coding: utf-8 -*-
"""PIT 行业归属测试（B7）

背景（为什么这是个真问题）
-------------------------
行业中性化的做法是"减去同行业均值"。如果用的是**当前**行业归属，就等于拿
二十年后的分类去算当年的行业均值：

    000007 深信泰丰  1992 家电 -> 2006 房地产 -> 2013 有色金属
                    -> 2014 社会服务 -> 2017 综合 -> 2019 商贸零售

拿"商贸零售"去中性化它 2006 年的因子值，行业均值里就混进了未来信息。
全库有 **1,646 只**股票换过行业（最多 6 段），这不是边角情况。

原料是 tushare `index_member_all`（申万成分分级），带 `in_date`/`out_date`/`is_new`。

【本文件盯住的三件事】
  1. **空档处理**：原始记录有"退出 A 行业但很久以后才纳入 B"的空档
     （000007 中间空了 3 年多）。跳回当前快照是前视，必须"沿用上一段"。
  2. **PIT 分组去均值的正确性**：必须逐日按当天行业分组。这里有个性能陷阱 ——
     朴素逐日 groupby 在这个项目里实测 30 分钟跑不完，所以实现是"按行业配置
     快照分块 + 块内向量化"。正确性必须用手工逐日结果对照。
  3. **退化路径**：所有股票行业都不变时应走快路径且结果一致。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _has_pit() -> bool:
    try:
        from database.industry import has_pit_data
        return bool(has_pit_data())
    except Exception:
        return False


# ============================================================
# 一、PIT 分组去均值（纯合成，不依赖数据）
# ============================================================
def _panel(n_days=6, cols=("A", "B", "C", "D"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame(rng.normal(0, 1, (n_days, len(cols))),
                        index=idx, columns=list(cols))


def test_pit_demean_matches_per_day_grouping():
    """**核心回归**：分块向量化实现必须与"逐日手工分组去均值"逐元素一致

    实现用了"按行业配置快照分块"来避性能陷阱。分块边界只要错一天，
    就会用错行业的均值 —— 而且不会报错。所以必须和朴素的逐日结果对齐。
    """
    from factors.fundamental import _group_demean_pit

    r = _panel(8)
    idx = r.index
    p = pd.DataFrame("X", index=idx, columns=r.columns)
    p.loc[idx[3]:, ["C", "D"]] = "Y"          # 第 4 天起 C/D 换行业
    p.loc[idx[6]:, ["B"]] = "Z"               # 第 7 天起 B 再换一次

    got = _group_demean_pit(r, p)

    manual = r.copy()
    for d in idx:
        g = p.loc[d]
        for grp in g.unique():
            m = (g == grp).to_numpy()
            vals = r.loc[d].to_numpy()[m]
            manual.loc[d, g.index[m]] = vals - vals.mean()

    assert np.allclose(got.to_numpy(), manual.to_numpy()), \
        "PIT 去均值与逐日手工结果不一致（分块边界或分组错了）"
    print("[OK] PIT 去均值 == 逐日手工分组（含两次换行业的边界）")


def test_pit_demean_uses_then_current_industry():
    """换行业前后，同一列减掉的均值必须**不同**（否则等于没用 PIT）"""
    from factors.fundamental import _group_demean_pit

    idx = pd.bdate_range("2024-01-01", periods=6)
    r = pd.DataFrame({"A": [0.0, 6.0, 10.0, 14.0, 18.0, 22.0],
                      "B": [1.0, 7.0, 11.0, 15.0, 19.0, 23.0],
                      "C": [2.0, 8.0, 12.0, 16.0, 20.0, 24.0],
                      "D": [3.0, 9.0, 13.0, 17.0, 21.0, 25.0]}, index=idx)
    p = pd.DataFrame("X", index=idx, columns=r.columns)
    p.loc[idx[3]:, ["C", "D"]] = "Y"

    out = _group_demean_pit(r, p)
    # 第 1 天 A/B/C/D 同组(X)：0,1,2,3 -> 均值 1.5 -> C = 2 - 1.5 = +0.5
    assert np.isclose(out.loc[idx[0], "C"], 2.0 - 1.5), out.loc[idx[0], "C"]
    # 第 4 天 C/D 已在 Y 组：16,17 -> 均值 16.5 -> C = 16 - 16.5 = -0.5
    # （注意不是把后面几行一起平均 —— 去均值只在**当日横截面**内做）
    assert np.isclose(out.loc[idx[3], "C"], 16.0 - 16.5), out.loc[idx[3], "C"]
    print("[OK] 换行业后减的是**新行业**均值（C: +0.5 -> -0.5）")


def test_pit_demean_fast_path_when_industry_constant():
    """所有股票行业都不变时走快路径，结果必须与静态分组完全一致"""
    from factors.fundamental import _group_demean, _group_demean_pit

    r = _panel(5)
    p = pd.DataFrame("X", index=r.index, columns=r.columns)
    p["A"] = "Y"                              # 全程不变
    g = p.iloc[0]
    g.index = r.columns

    fast = _group_demean_pit(r, p)
    ref = _group_demean(r, g)
    assert np.allclose(fast.to_numpy(), ref.to_numpy()), "快路径与静态分组不一致"
    print("[OK] 行业恒定 -> 快路径，结果与静态分组一致")


def test_pit_demean_handles_unknown_group():
    """未分类必须被当成一个**组**处理，不能让这些股票变成 NaN"""
    from factors.fundamental import _group_demean_pit

    r = _panel(4)
    p = pd.DataFrame("未分类", index=r.index, columns=r.columns)
    p["A"] = "X"
    out = _group_demean_pit(r, p)
    assert out.notna().all().all(), "出现 NaN —— 未分类股票的权重会凭空消失"
    print("[OK] 未分类被当成一组正常去均值（不产生 NaN）")


def test_neutralized_score_accepts_pit_dataframe():
    """`neutralized_score` 的 ind_map 支持 DataFrame（PIT），不只是 Series"""
    from factors.fundamental import neutralized_score

    r = _panel(5, cols=("A", "B", "C"))
    factors = {"bp": r.copy()}                # bp 方向为 +1
    mask = pd.DataFrame(True, index=r.index, columns=r.columns)
    pit = pd.DataFrame("X", index=r.index, columns=r.columns)
    pit.loc[r.index[2]:, "C"] = "Y"

    out = neutralized_score(factors, ["bp"], mask, ind_map=pit)
    assert out.shape == r.shape and out.notna().all().all()
    # 与 Series（时不变）的结果应当不同 —— 否则说明 PIT 面板被忽略了
    static = pd.Series("X", index=r.columns)
    out2 = neutralized_score(factors, ["bp"], mask, ind_map=static)
    assert not np.allclose(out.to_numpy(), out2.to_numpy()), \
        "传入 PIT DataFrame 却和时不变 Series 结果一样 —— ind_map 被忽略了"
    print("[OK] neutralized_score 接受 PIT DataFrame，且结果与静态不同")


# ============================================================
# 二、区间构造：空档必须"沿用上一段"，不能跳回当前快照
# ============================================================
def test_intervals_carry_forward_bridges_gaps():
    """**核心回归**：区间必须首尾相接、不重叠（空档沿用上一段）

    真实案例：000007 在 2009-05-27 退出房地产、2013-07-01 才纳入有色金属，
    中间三年多 tushare 没有记录。跳回"当前快照"（商贸零售）是明确的前视 ——
    2010 年不可能知道这家公司 2019 年会变成商贸零售。
    """
    if not _has_pit():
        print("[SKIP] 无 sw_member 数据（跑 --only sw_member 后才有）")
        return
    from database.industry import industry_intervals

    iv = industry_intervals()
    assert iv, "区间为空"
    for code, recs in iv.items():
        assert recs, f"{code} 区间为空"
        # 按 in_date 升序
        starts = [r[0] for r in recs]
        assert starts == sorted(starts), f"{code} 区间未按 in_date 排序"
        # 首尾相接、不重叠：第 i 段的结束 == 第 i+1 段的开始
        for i in range(len(recs) - 1):
            assert recs[i][1] is not None, f"{code} 中间段的结束日不该为空"
            assert recs[i][1] == recs[i + 1][0], \
                f"{code} 第 {i} 段与第 {i+1} 段不衔接（空档没桥接）"
        assert recs[-1][1] is None, f"{code} 最后一段的结束日应为空（延续至今）"
    print(f"[OK] 区间构造：{len(iv):,} 只股票全部首尾相接、最后一段延续至今")


def test_industry_at_is_point_in_time():
    """同一只股票在不同日期必须返回**当时**的行业"""
    if not _has_pit():
        print("[SKIP] 无 sw_member 数据")
        return
    from database.industry import industry_at, industry_intervals

    iv = industry_intervals()
    # 找一只换过行业的股票
    changed = [c for c, v in iv.items() if len(v) > 1]
    assert changed, "没有换过行业的股票？数据可疑"
    code = "000007" if "000007" in iv else changed[0]
    recs = iv[code]

    seen = []
    for s, e, name in recs:
        got = industry_at(code, s, iv)
        assert got == name, f"{code} 在 {s.date()} 应为 {name}，实得 {got}"
        seen.append(name)
    # 至少两个不同行业，才说明确实在变
    assert len(set(seen)) > 1, f"{code} 的行业序列没有变化: {seen}"
    print(f"[OK] PIT 查询：{code} 行业随日期变化 {seen}")


def test_industry_at_unknown_code_returns_none():
    """查不到的代码返回 None（由调用方决定落到未分类），不抛异常"""
    if not _has_pit():
        print("[SKIP] 无 sw_member 数据")
        return
    from database.industry import industry_at

    assert industry_at("999999", "2020-06-30") is None
    print("[OK] 未知代码返回 None（不抛异常）")


def test_pit_panel_shape_and_attrs():
    if not _has_pit():
        print("[SKIP] 无 sw_member 数据")
        return
    from database.industry import industry_pit_panel

    dates = pd.to_datetime(["2005-06-30", "2015-06-30", "2025-06-30"])
    codes = ["000007", "600000", "000001"]
    p = industry_pit_panel(dates, codes)
    assert p.shape == (3, 3), p.shape
    assert list(p.index) == list(dates)
    assert list(p.columns) == codes
    assert p.notna().all().all()
    for k in ("pit_coverage", "snapshot_fallback_ratio", "source"):
        assert k in p.attrs, f"缺少 attrs[{k}]（调用方无法知道兜底比例）"
    # 000007 在 2010 前后应是不同行业（这条断言正是 B7 的意义）
    assert p.loc[dates[0], "000007"] != p.loc[dates[2], "000007"], \
        "000007 三个时点行业相同 —— PIT 没生效"
    print(f"[OK] PIT 面板：{p.shape}，覆盖率 "
          f"{p.attrs['pit_coverage']:.1%}，源={p.attrs['source'][:24]}...")


if __name__ == "__main__":
    test_pit_demean_matches_per_day_grouping()
    test_pit_demean_uses_then_current_industry()
    test_pit_demean_fast_path_when_industry_constant()
    test_pit_demean_handles_unknown_group()
    test_neutralized_score_accepts_pit_dataframe()
    test_intervals_carry_forward_bridges_gaps()
    test_industry_at_is_point_in_time()
    test_industry_at_unknown_code_returns_none()
    test_pit_panel_shape_and_attrs()
    print("\n全部 PIT 行业归属测试通过")
