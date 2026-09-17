# -*- coding: utf-8 -*-
"""未来因子（look-ahead bias / 数据泄漏）检测

核心问题:
    回测策略在决策时刻，是否使用了"那时还不存在的数据"？
    经典错误：
      - 用财务报表的 end_date（报告期）而不是 ann_date（公告日）判断
        → 报告期数据在公告日前根本不可得，用了就是"偷看未来"
      - 用当日收盘价在当日开盘决策（收盘价当日收盘后才知道）
      - 幸存者偏差（用现在的股票名单回测历史）

本模块提供:
    1. assert_no_lookahead: 回测后断言——所有决策使用的数据均可获得
    2. 其他检测方式（见文档字符串底部）

用法:
    from backtest.lookahead import assert_no_lookahead, attach_availability

    # 回测完成后：
    violations = assert_no_lookahead(
        decision_dates=rebalance_dates,          # 每次决策的日期
        used_data={
            "pe_ttm": valuation_df,              # 策略实际使用的估值
            "close":  daily_df,                  # 策略实际使用的价格
        },
        raise_on_violation=True,                 # True=断言失败抛错
    )
"""
import pandas as pd


def attach_availability(df, trade_date_col="trade_date", available_date_col=None):
    """为数据附加"_available"列 = 该条数据真实可获得日期

    规则:
        - 默认: 可获得日 = 交易日（价格/估值当日收盘后发布）
        - 财务类: 传入 available_date_col="ann_date"（公告日），
          则用公告日而非报告期——这是防未来因子的关键
    """
    df = df.copy()
    if available_date_col and available_date_col in df.columns:
        df["_available"] = pd.to_datetime(df[available_date_col], errors="coerce")
    else:
        df["_available"] = pd.to_datetime(df[trade_date_col], errors="coerce")
    return df


def check_lookahead(decision_dates, used_data, trade_same_day_ok=True):
    """逐决策检查是否用了"决策日后才可获得"的数据

    参数:
        decision_dates:   决策日列表（每次重平衡/调仓的日期）
        used_data:        {数据名: DataFrame}，须已含 "_available" 列
                          （用 attach_availability 预处理）
        trade_same_day_ok: True=允许决策日使用当日收盘数据（收盘价撮合惯例）；
                           False=严格要求用前一日数据（更保守）

    返回:
        违规列表 [(决策日, 数据名, 标识, 可获得日), ...]
    """
    violations = []
    for d in decision_dates:
        d = pd.Timestamp(d)
        limit = d if trade_same_day_ok else d - pd.Timedelta(days=1)
        for name, df in used_data.items():
            if df is None or df.empty or "_available" not in df.columns:
                continue
            future = df[df["_available"] > limit]
            if not future.empty:
                for _, r in future.iterrows():
                    code = r.get("code", r.get("ts_code", "?"))
                    violations.append((d, name, code, r["_available"]))
    return violations


def verify_point_in_time(decision_pairs, trade_same_day_ok=True):
    """逐决策配对校验（Point-in-Time 正确姿势）

    每个决策只校验"该决策实际使用的数据"是否在决策日已可获得，
    避免把"后续决策才用到的数据"误判为前面决策的未来泄漏。

    参数:
        decision_pairs: [(decision_date, {数据名: 该决策实际使用的DataFrame}), ...]
                        DataFrame 须含 "_available" 列（attach_availability 预处理）
        trade_same_day_ok: True=允许决策日使用当日数据；False=严格前一日

    返回:
        违规列表 [(决策日, 数据名, 标识, 可获得日), ...]
    """
    violations = []
    for date, used in decision_pairs:
        date = pd.Timestamp(date)
        limit = date if trade_same_day_ok else date - pd.Timedelta(days=1)
        for name, df in used.items():
            if df is None or df.empty or "_available" not in df.columns:
                continue
            future = df[pd.to_datetime(df["_available"]) > limit]
            for _, r in future.iterrows():
                code = r.get("code", r.get("ts_code", "?"))
                violations.append((date, name, code, r["_available"]))
    return violations


def assert_no_lookahead(decision_dates, used_data, raise_on_violation=True,
                        trade_same_day_ok=True):
    """未来因子断言：回测完成后调用

    若检测到任一决策使用了"决策日之后才可获得"的数据，则报错（默认）
    或告警（raise_on_violation=False）。

    ⚠️ **`used_data` 必须是"每个决策各自实际使用的那一小片数据"，不能传全量面板。**
    传全量面板时，对除最后一个决策日以外的每一天，面板里都必然存在
    `_available > 决策日` 的行（因为面板覆盖到回测结束），于是会报出成千上万条
    假违规 —— 这个检查就废了。要传全量面板请用 `verify_point_in_time` 的
    逐决策配对形式，或直接用 `check_fill_timing` 做引擎层校验。
    本函数会在检测到"像全量面板"时直接抛错，避免静默产生垃圾结论。

    参数:
        decision_dates:      决策日列表
        used_data:          {数据名: DataFrame}，须含 "_available" 列
        raise_on_violation: True=抛 AssertionError；False=仅打印警告
        trade_same_day_ok:  True=允许决策日使用当日收盘数据（收盘撮合惯例）；
                            False=严格只用前一日数据

    返回:
        违规数量（0 = 通过）
    """
    _reject_full_panel(used_data, decision_dates)
    violations = check_lookahead(decision_dates, used_data,
                                 trade_same_day_ok=trade_same_day_ok)
    if violations:
        msg = f"\n[未来因子检测] 发现 {len(violations)} 处未来数据泄漏(look-ahead)！前10条：\n"
        for v in violations[:10]:
            msg += (f"  决策日 {v[0].date()} 使用了 {v[1]} 中 {v[2]} 的数据"
                    f"（该数据可获得日为 {v[3].date()}，晚于决策日）\n")
        msg += "提示：检查是否误用了报告期而非公告日、或当日收盘数据提前使用。"
        if raise_on_violation:
            raise AssertionError(msg)
        print("[警告]" + msg)
        return len(violations)
    print("[未来因子检测] 通过：所有决策使用的数据在决策日之前均已可获得，无未来数据泄漏。")
    return 0


def _reject_full_panel(used_data, decision_dates):
    """拦住"把全量面板当 used_data 传"这个致命误用

    判据：如果某个数据表里 `_available` 的最大值**远晚于**最后一个决策日，
    说明它覆盖了决策之后的时间 —— 那不是"某次决策用到的数据"，而是整段面板。
    此时 `check_lookahead` 会对几乎每个决策日都报违规，结论毫无意义。
    与其静默给出垃圾，不如直接报错并告诉调用方该用什么。
    """
    ds = [pd.Timestamp(d) for d in decision_dates]
    if not ds:
        return
    last = max(ds)
    for name, df in (used_data or {}).items():
        if df is None or getattr(df, "empty", True) or "_available" not in df.columns:
            continue
        avail = pd.to_datetime(df["_available"], errors="coerce").dropna()
        if len(avail) and avail.max() > last + pd.Timedelta(days=30):
            raise ValueError(
                f"used_data['{name}'] 看起来是**全量面板**（可获得日到 "
                f"{avail.max().date()}，而最后一个决策日是 {last.date()}）。"
                f"`check_lookahead` 只接受「每个决策各自实际使用的那一小片数据」；"
                f"传全量面板会对除最后一天外的每个决策日都报假违规。"
                f"要在回测后做引擎层校验请用 `check_fill_timing`，"
                f"要逐决策校验请用 `verify_point_in_time`。"
            )


# ============================================================
# 引擎层校验：成交时点不得早于决策时点
# ============================================================
def check_fill_timing(rebalance_dates, trades, fill_timing="next_open",
                      trade_date_col="timestamp"):
    """校验"成交不早于决策"，并识别同日成交的前视风险

    这是**引擎能自己保证**的那部分 PIT —— 不需要知道策略用了哪些数据，
    只需要知道"哪一天做的决策"和"成交发生在哪一天"。

    规则：
      - `next_open`（本项目默认）：T 日收盘后算权重、**T+1 开盘成交**。
        因此任何一笔成交都**不得**发生在它的决策日当天或之前。
        违反 = 用了当天收盘信息在当天成交，典型前视。
      - `same_close`：决策与成交同日、同用收盘价。**这是温和的前视**
        （收盘价要收盘后才知道，却按该价成交）。不是错误，但必须显式声明，
        这里作为"声明项"返回而不是静默放过。

    参数:
        rebalance_dates: 产生目标权重的决策日（升序）
        trades:          成交明细 DataFrame，含 trade_date_col
        fill_timing:     "next_open" / "same_close"

    返回:
        dict(report=str, violations=list, same_day_fills=int, checked=int)
    """
    reb = sorted(pd.Timestamp(d) for d in rebalance_dates)
    lines, violations = [], []
    same_day = 0
    checked = 0
    if trades is None or getattr(trades, "empty", True) or not reb:
        return {"report": "  成交时点校验: 无成交或无调仓，跳过",
                "violations": [], "same_day_fills": 0, "checked": 0}

    t = trades
    if trade_date_col not in t.columns:
        t = t.reset_index()
        if trade_date_col not in t.columns:
            trade_date_col = t.columns[0]
    dates = pd.to_datetime(t[trade_date_col], errors="coerce")

    for d in dates:
        if pd.isna(d):
            continue
        checked += 1
        # 该笔成交对应的最近一次决策（不晚于它的最后一次调仓）
        prior = [r for r in reb if r <= d]
        if not prior:
            violations.append((d, None, "成交早于任何一次调仓决策"))
            continue
        r = prior[-1]
        if d == r:
            same_day += 1
            if fill_timing != "same_close":
                violations.append((d, r, f"{fill_timing} 模式下不得在决策日当天成交"))
    if same_day and fill_timing == "same_close":
        lines.append(f"  ⚠ 同日成交 {same_day}/{checked} 笔（fill=same_close）："
                     f"这是**温和前视** —— 用当日收盘价决策并按同一价格成交。"
                     f"做敏感性请加 --fill open 对比")
    if violations:
        lines.append(f"  ✗ 成交时点违规 {len(violations)} 笔（前 5 条）：")
        for d, r, why in violations[:5]:
            lines.append(f"      成交日 {d.date()}  决策日 "
                         f"{r.date() if r is not None else '—'}  {why}")
    else:
        lines.append(f"  ✓ 成交时点校验通过：{checked} 笔成交均不早于其决策日"
                     f"（fill={fill_timing}）")
    return {"report": "\n".join(lines), "violations": violations,
            "same_day_fills": same_day, "checked": checked}



# ============================================================
# 其他未来因子检测方式（在策略开发中可选用）
# ============================================================
"""
除上述"决策日 vs 数据可获得日"断言外，常用检测方法：

1. Point-in-Time (PIT) 数据重放
   模拟每个历史时点"当时能看到的数据"，杜绝幸存者偏差和延迟发布的财务数据。
   实现：财务数据按 ann_date 延后生效，股票池用当时在市的股票。

2. 延迟敏感性分析（最实用的信号）
   把因子更新时间整体向后延迟 1、2、5 天重跑回测：
     - 若延迟后收益暴跌 → 说明策略依赖"最新数据"，存在 look-ahead
     - 若延迟后收益基本不变 → 因子稳健，无未来泄漏
   例：重平衡日从"用当日PE"改为"用3日前PE"，若结果差很多则有问题。

3. 随机化/置换检验
   将因子时间序列与收益序列打乱对齐后重跑，若策略"依然赚钱"，
   说明收益来自偶然噪声而非真实信号（无效策略）。

4. Walk-Forward 样本外测试
   前段优化参数、后段验证，防止用同一段数据既选参又评估（过拟合的 look-ahead）。

5. 幸存者偏差检查
   确认股票池包含当时上市、后来退市的股票（我们的 frozen/stocks 含退市股，
   选股时若只用"当前列表"就会引入幸存者偏差）。

6. 峰值/最终值复用检查
   统计中若用全样本均值/最大值标准化，会在每点都"提前知道未来"，
   应改用滚动窗口（expanding/rolling）。
"""
