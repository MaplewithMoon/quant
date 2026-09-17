# -*- coding: utf-8 -*-
"""市场级信号（factors/market.py）测试

为什么单独测这个模块
--------------------
`margin` / `northbound` / `futures` / `options` / `etf` 五个 frozen 数据集
此前**只有下载器、没有任何消费方**（B12）。本模块把它们接成**总仓位择时**
信号，这才第一次有了真正的读取方。

正因为它们是"择时信号"，两个错误模式的代价极高，必须测：
  1. **前视**（PIT）：两融、北向、期货结算、期权持仓都是**收盘后**才发布的，
     当天根本拿不到。若信号不做 lag，回测会拿"今晚才知道的数"决定"今天怎么交易"，
     收益曲线会凭空变好 —— 而且**看不出来**。
  2. **分位阈值前视**：`exposure_from_signal` 若用全样本分位，2018 年的仓位
     就"知道"了 2025 年的分布。这是很隐蔽的一种，回报同样虚高。

所以下面测的不是"函数能不能跑"，而是"它有没有偷偷看到未来"。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _has(ds: str) -> bool:
    from database.config import FROZEN_ROOT
    d = FROZEN_ROOT / ds
    return d.exists() and any(d.glob("year=*/*.parquet"))


# ============================================================
# 一、PIT：信号必须右移
# ============================================================
def test_market_signals_lag_is_exactly_a_shift():
    """lag=1 必须**恰好**等于 lag=0 右移一行 —— 不能是"差不多"

    这是前视的第一道闸门：只要 lag 真的生效，第 t 行就不含 t 日收盘后
    才发布的任何信息。
    """
    from factors.market import market_signals

    if not _has("futures"):
        print("[SKIP] 无 frozen/futures，跳过 lag 检验")
        return
    a = market_signals("2023-01-01", "2024-12-31", lag=0)
    b = market_signals("2023-01-01", "2024-12-31", lag=1)
    assert not a.empty, "取不到市场信号（futures 为空？）"
    exp = a.shift(1)
    pd.testing.assert_frame_equal(b, exp)
    # lag=0 与 lag=1 必须不同，否则说明 shift 被吃掉了
    assert not a.equals(b), "lag 参数没有生效"
    print(f"[OK] 市场信号 PIT：lag=1 严格等于 lag=0 右移一行（{a.shape[1]} 列）")


def test_signals_do_not_contain_future_dates():
    """信号的最大日期不得晚于原始数据 —— 右移只会让它更早，不会更晚"""
    from factors.market import load_futures_basis, market_signals

    if not _has("futures"):
        print("[SKIP] 无 frozen/futures，跳过未来日期检验")
        return
    raw = load_futures_basis(start="2023-01-01", end="2024-12-31")
    sig = market_signals("2023-01-01", "2024-12-31", lag=0)
    assert sig.index.max() <= raw.index.max(), "信号出现了原始数据没有的日期"
    print(f"[OK] 信号日期不超出原始数据（max={str(sig.index.max())[:10]}）")


# ============================================================
# 二、仓位映射：不得使用全样本分位
# ============================================================
def test_exposure_expanding_has_no_lookahead():
    """扩张窗口：截断未来数据后，历史部分的仓位必须**逐点不变**

    这是"分位阈值前视"的直接检验。用全样本分位时，截断会把 2018 年的阈值
    改掉，历史仓位随之变化 —— 这个测试就会红。
    """
    from factors.market import exposure_from_signal

    rng = np.random.default_rng(7)
    n = 1000
    s = pd.Series(rng.normal(0, 1, n).cumsum(),
                  index=pd.bdate_range("2018-01-01", periods=n))

    full = exposure_from_signal(s, expanding=True)
    # 截到 600 天：前 100 天之后的仓位必须与全样本完全一致
    cut = exposure_from_signal(s.iloc[:600], expanding=True)
    common = cut.index[100:]
    d = (full.reindex(common) - cut.reindex(common)).abs().max()
    assert d < 1e-12, \
        f"扩张窗口仍受未来数据影响（最大差 {d:.3e}）—— 用到了全样本分位，前视"

    # 反面对照：非扩张（全样本分位）**应当**受未来影响，否则测试没测到东西
    full_f = exposure_from_signal(s, expanding=False)
    cut_f = exposure_from_signal(s.iloc[:600], expanding=False)
    assert (full_f.reindex(common) - cut_f.reindex(common)).abs().max() > 1e-9, \
        "全样本分位竟然不受截断影响，说明这个对照失效了"
    print("[OK] 仓位映射：扩张窗口无前视（全样本分位作为对照确实会变）")


def test_exposure_bounds_and_monotonicity():
    """仓位必须落在 [min, max] 内，且随信号单调不减"""
    from factors.market import exposure_from_signal

    s = pd.Series(np.arange(200, dtype=float),
                  index=pd.bdate_range("2020-01-01", periods=200))
    e = exposure_from_signal(s, min_exposure=0.3, max_exposure=1.0)
    assert e.notna().all(), "仓位出现 NaN"
    assert e.min() >= 0.3 - 1e-12 and e.max() <= 1.0 + 1e-12, \
        f"仓位越界 [{e.min():.4f}, {e.max():.4f}]"
    assert (e.diff().dropna() >= -1e-12).all(), "信号递增时仓位却下降了"
    # 常数信号 -> 分位跨度为 0，应回落到中值而不是 NaN 或爆炸
    c = pd.Series(1.0, index=s.index)
    ec = exposure_from_signal(c)
    assert ec.notna().all() and ec.between(0.3, 1.0).all(), \
        f"常数信号处理错误：{ec.unique()[:5]}"
    print(f"[OK] 仓位映射：界内/单调/常数信号回落中值（{ec.iloc[0]:.2f}）")


def test_apply_exposure_keeps_nan_and_scales_only_rebalance_days():
    """缩放只作用于调仓日的非 NaN 权重，且不把 NaN 变成可交易"""
    from factors.market import apply_exposure

    idx = pd.bdate_range("2024-01-01", periods=6)
    w = pd.DataFrame(np.nan, index=idx, columns=["A", "B"])
    w.iloc[0] = [0.5, 0.5]          # 调仓日
    w.iloc[3] = [0.6, 0.4]          # 调仓日
    expo = pd.Series([0.5, 0.5, 0.5, 0.8, 0.8, 0.8], index=idx)

    out = apply_exposure(w, expo)
    assert np.allclose(out.iloc[0].values, [0.25, 0.25]), out.iloc[0].values
    assert np.allclose(out.iloc[3].values, [0.48, 0.32]), out.iloc[3].values
    assert out.iloc[1].isna().all() and out.iloc[2].isna().all(), \
        "非调仓日被缩放成了非 NaN —— 会凭空产生一次调仓"
    assert np.isclose(out.iloc[0].sum(), 0.5), "总仓位缩放后不等于 exposure"

    # 信号缺失的日期按 1.0 处理（不因为没信号就默认空仓）
    partial = pd.Series([0.5], index=idx[:1])
    out2 = apply_exposure(w, partial)
    assert np.allclose(out2.iloc[3].values, [0.6, 0.4]), \
        "信号缺失日被当成空仓，会无故清仓"
    assert np.allclose(out2.iloc[0].values, [0.25, 0.25])

    # renormalize：只调相对权重，总仓位回到 1
    out3 = apply_exposure(w, expo, renormalize=True)
    assert np.isclose(out3.iloc[0].sum(), 1.0)
    assert np.isclose(out3.iloc[3].sum(), 1.0)
    assert np.allclose(out3.iloc[0].values, [0.5, 0.5]), "归一化应还原相对权重"
    print("[OK] 仓位缩放：调仓日缩放 / 非调仓日保持 NaN / 缺信号按 1.0 / "
          "renormalize 还原")


def test_apply_exposure_clips_out_of_range():
    """越界的 exposure 要被夹住，不能放大杠杆"""
    from factors.market import apply_exposure

    idx = pd.bdate_range("2024-01-01", periods=2)
    w = pd.DataFrame(1.0, index=idx, columns=["A"])
    out = apply_exposure(w, pd.Series([2.0, -1.0], index=idx))
    assert np.isclose(out.iloc[0, 0], 1.0), ">1 的 exposure 应被夹到 1（不允许杠杆）"
    assert np.isclose(out.iloc[1, 0], 0.0), "<0 的 exposure 应被夹到 0"
    print("[OK] 仓位缩放：越界 exposure 被夹到 [0,1]")


# ============================================================
# 三、真实数据读取（数据缺失则 SKIP）
# ============================================================
def test_load_margin_shape_and_aggregation():
    """两融：按日聚合两个交易所，字段齐全且非负"""
    from factors.market import load_margin

    if not _has("margin"):
        print("[SKIP] 无 frozen/margin")
        return
    df = load_margin("2023-01-01", "2024-12-31")
    assert not df.empty, "两融读取为空"
    for c in ("rzye", "rzmre", "rzche", "rqye", "rzrqye"):
        assert c in df.columns, f"缺列 {c}"
    assert df.index.is_monotonic_increasing and df.index.is_unique, "日期索引不合法"
    assert (df["rzye"] > 0).all(), "融资余额出现非正值"
    # 按日聚合后同一交易日只能有一行（否则就是没 GROUP BY）
    assert df.index.is_unique
    print(f"[OK] 两融：{df.shape[0]} 交易日 × {df.shape[1]} 列，"
          f"融资余额中位 {df['rzye'].median()/1e8:,.0f} 亿")


def test_load_futures_basis_is_below_or_near_zero():
    """股指期货基差：列名对应、量级合理

    A 股股指期货长期**贴水**，所以中位数应为负；但也不会到 -20%
    （那说明把期货价和错误的现货配对，或量纲不一致）。
    """
    from factors.market import FUTURES_INDEX_PAIRS, load_futures_basis

    if not _has("futures") or not _has("index_daily"):
        print("[SKIP] 无 frozen/futures 或 index_daily")
        return
    df = load_futures_basis(start="2023-01-01", end="2024-12-31")
    assert not df.empty, "基差读取为空"
    for sym in FUTURES_INDEX_PAIRS:
        col = f"{sym}_basis"
        if col not in df.columns:
            continue
        v = df[col].dropna()
        assert len(v) > 100, f"{col} 样本过少（{len(v)}）"
        assert v.median() < 0.01, f"{col} 中位 {v.median():.3f} 不合理（现货配对错了？）"
        assert v.abs().max() < 0.20, f"{col} 极值 {v.abs().max():.3f} 过大"
    print(f"[OK] 股指期货基差：{df.shape[0]} 交易日 × {df.shape[1]} 列，"
          + ", ".join(f"{c}中位{df[c].median():+.2%}" for c in df.columns))


def test_load_northbound_is_covered_by_its_real_range():
    """北向：库里只有 2025-05 起的数据，早区间必须**返回空**而不是报错

    这条同时是文档承诺的回归：如果有人误以为北向有长历史，
    这个测试会立刻暴露真实覆盖范围。
    """
    from factors.market import load_northbound

    if not _has("northbound"):
        print("[SKIP] 无 frozen/northbound")
        return
    early = load_northbound("2018-01-01", "2019-12-31")
    assert early.empty, \
        f"2018-2019 竟然取到北向数据（{len(early)} 行）—— 覆盖范围与文档不符"
    late = load_northbound("2025-01-01", "2026-12-31")
    assert not late.empty, "2025 年之后应有北向数据"
    assert late.index.min() >= pd.Timestamp("2025-01-01")
    print(f"[OK] 北向：早期区间正确返回空，实际覆盖 {len(late)} 行，"
          f"起于 {str(late.index.min())[:10]}")


def test_load_option_pcr_ratio_semantics():
    """期权 PCR：put/call 之比，且必须标注 opt_basic 覆盖率"""
    from factors.market import load_option_pcr

    if not _has("options"):
        print("[SKIP] 无 frozen/options")
        return
    df = load_option_pcr("2023-01-01", "2024-12-31")
    if df.empty:
        print("[SKIP] options 区间内无数据")
        return
    assert {"pcr_vol", "pcr_oi"} <= set(df.columns), df.columns.tolist()
    v = df["pcr_vol"].dropna()
    assert (v > 0).all(), "PCR 出现非正值（分子分母搞反或含零）"
    assert 0.1 < v.median() < 10, f"PCR 中位 {v.median():.3f} 不在合理量级"
    # 覆盖率必须写进 attrs —— 只覆盖 SSE 时结果不能假装代表全市场
    assert "coverage" in df.attrs, "缺少 coverage 标注"
    assert df.attrs["coverage"] <= 1.0
    print(f"[OK] 期权 PCR：{len(df)} 交易日，PCR(量)中位 {v.median():.3f}，"
          f"opt_basic 覆盖 {df.attrs['coverage']:.1%}")


def test_load_etf_prices_uses_date_partition_layout():
    """ETF：分区是 `year=Y/{日期}.parquet`，与股票相反，必须能正常读出宽表

    ⚠️ 测试标的**不能**取 `fund_basic` 的前几只：那张表按代码排序，
    排在最前的是 2025/2026 年才成立的新 ETF（158031.SZ 之类），
    在任意历史区间里都没有行情 —— 一取 head(3) 测试就直接跳过，等于没测。
    改为从**目标区间实际有行情的**代码里挑，才真正覆盖到日期分区布局。
    """
    from database.config import FROZEN_ROOT, connect_duckdb, year_globs
    from factors.market import load_etf_basic, load_etf_prices

    if not _has("etf"):
        print("[SKIP] 无 frozen/etf")
        return
    b = load_etf_basic()
    assert not b.empty and "ts_code" in b.columns, "fund_basic 读取失败"
    start, end = "2024-01-01", "2024-03-31"

    con = connect_duckdb()
    try:
        g = year_globs(FROZEN_ROOT / "etf", 2024, 2024)
        codes = con.execute(f"""
            SELECT ts_code FROM read_parquet({g})
            WHERE trade_date >= DATE '{start}' AND trade_date <= DATE '{end}'
            GROUP BY 1 ORDER BY count(*) DESC LIMIT 3
        """).fetchdf()["ts_code"].astype(str).tolist()
    finally:
        con.close()
    assert codes, f"{start}~{end} 区间内没有任何 ETF 行情，无法验证读取"

    px = load_etf_prices(codes=codes, start=start, end=end)
    assert not px.empty, f"取不到 {codes} 的 ETF 行情（日期分区布局读取有问题）"
    assert isinstance(px.index, pd.DatetimeIndex), "索引不是交易日"
    assert set(px.columns) == set(codes), f"列不是 ts_code：{px.columns.tolist()}"
    assert px.shape[0] > 20, f"ETF 宽表行数过少 {px.shape}"
    assert (px.dropna(how="all") > 0).all().all(), "ETF 价格出现非正值"
    # 区间过滤必须生效（年份分区只是粗筛，区间才是精筛）
    assert px.index.min() >= pd.Timestamp(start) and px.index.max() <= pd.Timestamp(end), \
        f"区间过滤失效：[{px.index.min()}, {px.index.max()}]"
    print(f"[OK] ETF：fund_basic {b.shape[0]} 只；价格宽表 {px.shape[0]} 交易日 "
          f"× {px.shape[1]} 只（日期分区布局读取正确）")


def test_backtest_engine_accepts_exposure():
    """引擎真的接了 exposure —— 否则 factors/market.py 仍然只是"能跑没人用" """
    import inspect

    from backtest.multi_engine import PortfolioBacktestEngine

    sig = inspect.signature(PortfolioBacktestEngine.run)
    assert "exposure" in sig.parameters, \
        "回测引擎没有 exposure 参数，市场信号无法接入实际组合"
    assert sig.parameters["exposure"].default is None, "默认必须为 None（不改变原行为）"
    print("[OK] 组合引擎已接入 exposure 参数（市场信号可用于总仓位控制）")


# ============================================================
# 四、数据层回归：下载区间重叠导致整行重复
# ============================================================
def test_margin_quarter_ranges_do_not_overlap():
    """**回归**：季度下载区间必须闭合在季末，不能重叠

    旧写法 `end_date=f"{q[:4]}1231"`（年末）让 Q1 拉 1~12 月、Q2 拉 4~12 月……
    10 月以后被拉 4 遍，叠加"concat 后不去重"，把 `frozen/margin` 写成
    2.5 倍大小（实测 2023 年 1,803 行 vs 真实 726 行），且主键仍唯一、
    主键检查查不出来。这个测试直接盯住区间本身。
    """
    from datetime import date

    from database.downloader.other import QUARTER_ENDS, quarter_ranges

    rs = quarter_ranges(2023, 2023)
    assert len(rs) == 4, f"一年应有 4 个季度，实得 {len(rs)}"

    # ① 每个区间必须结束在季末，绝不能是年末
    ends = {m_end for _, m_end in QUARTER_ENDS}
    assert ends == {"0331", "0630", "0930", "1231"}, ends
    for s, e in rs:
        assert e[4:] != "1231" or s[4:] == "1001", \
            f"区间 {s}~{e} 的结束日不是该季度末（旧 bug：一律拉到年末）"

    # ② 任意两个区间不得重叠
    def span(a):
        return date(int(a[:4]), int(a[4:6]), int(a[6:]))
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            s1, e1 = rs[i]
            s2, e2 = rs[j]
            assert span(e1) < span(s2) or span(e2) < span(s1), \
                f"区间 {rs[i]} 与 {rs[j]} 重叠 —— 会产生整行重复"

    # ③ 全年必须被完整覆盖，且首尾相接（不漏数据）
    assert rs[0][0] == "20230101" and rs[-1][1] == "20231231"
    for k in range(len(rs) - 1):
        assert span(rs[k][1]) < span(rs[k + 1][0]), "相邻季度之间有空隙"

    # ④ 跨年时数量正确、年份正确
    multi = quarter_ranges(2023, 2025)
    assert len(multi) == 12, f"2023~2025 应有 12 个季度，实得 {len(multi)}"
    assert {s[:4] for s, _ in multi} == {"2023", "2024", "2025"}
    print(f"[OK] 两融下载区间：{len(rs)} 个季度互不重叠、首尾相接、"
          f"结束于季末（旧 bug 回归）")


def test_frozen_data_has_no_exact_duplicate_rows():
    """**回归**：frozen 批量数据集不得有整行完全重复

    这是"下载区间重叠 + concat 不去重"在**数据层面**的哨兵：margin 的重复
    让 `sum(rzye)` 变成真实值的 2.5 倍，而主键 `(trade_date, exchange_id)`
    仍然唯一，主键检查完全查不出来。

    ⚠️ 必须**逐文件**查，不能整库 `DISTINCT *`：
      - `margin` 各年分区 schema 并不完全一致（`rqyl` 在 2010-2017 是 INTEGER、
        2018+ 是 DOUBLE），整库 `SELECT DISTINCT *` 会抛 ConversionException
        —— 而如果就这么 `except: continue`，这个测试会**跳过它唯一要保护的数据集
        却仍然打印 OK**，是个假绿灯。
      - `options` 2,300 万行整库 DISTINCT 直接 OOM。
    逐文件查既避开 schema 合并，也不受数据量影响。每个数据集抽查 8 个文件
    （跨年份均匀取），并把**实际检查的个数**打出来 —— 沉默的跳过等于没测。
    """
    from database.config import FROZEN_ROOT, connect_duckdb

    if not (FROZEN_ROOT / "margin").exists():
        print("[SKIP] 无 frozen/margin")
        return
    con = connect_duckdb()
    bad, per_ds = [], {}
    try:
        for ds in ("margin", "northbound", "futures", "etf", "options"):
            d = FROZEN_ROOT / ds
            files = sorted(d.glob("year=*/*.parquet")) if d.exists() else []
            if not files:
                per_ds[ds] = 0
                continue
            step = max(1, len(files) // 8)
            picked = files[::step][:8]
            n_ok = 0
            for f in picked:
                p = f.as_posix()
                try:
                    n = con.execute(
                        f"SELECT count(*) FROM read_parquet('{p}')").fetchone()[0]
                    u = con.execute(
                        f"SELECT count(*) FROM (SELECT DISTINCT * "
                        f"FROM read_parquet('{p}'))").fetchone()[0]
                except Exception:
                    continue
                n_ok += 1
                if n != u:
                    bad.append((f"{ds}/{f.name}", n, u))
            per_ds[ds] = n_ok
    finally:
        con.close()

    assert per_ds.get("margin", 0) > 0, \
        "一个 margin 文件都没查成 —— 哨兵形同虚设（不要静默跳过）"
    assert not bad, ("frozen 存在整行重复（下游 sum/mean 会被放大）："
                     + "; ".join(f"{d} {n:,}->{u:,} (+{n - u:,})"
                                 for d, n, u in bad)
                     + "；用 `python scripts/validate_data.py --check duplicates "
                       "--repair-duplicates` 修复")
    print("[OK] frozen 无整行重复（逐文件抽查 "
          + "、".join(f"{k}:{v}" for k, v in per_ds.items())
          + " 个；margin 放大 2.5 倍 bug 的哨兵）")


if __name__ == "__main__":
    test_market_signals_lag_is_exactly_a_shift()
    test_signals_do_not_contain_future_dates()
    test_exposure_expanding_has_no_lookahead()
    test_exposure_bounds_and_monotonicity()
    test_apply_exposure_keeps_nan_and_scales_only_rebalance_days()
    test_apply_exposure_clips_out_of_range()
    test_load_margin_shape_and_aggregation()
    test_load_futures_basis_is_below_or_near_zero()
    test_load_northbound_is_covered_by_its_real_range()
    test_load_option_pcr_ratio_semantics()
    test_load_etf_prices_uses_date_partition_layout()
    test_backtest_engine_accepts_exposure()
    test_margin_quarter_ranges_do_not_overlap()
    test_frozen_data_has_no_exact_duplicate_rows()
    print("\n全部市场级信号测试通过")
