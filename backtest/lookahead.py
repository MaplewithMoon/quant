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

    参数:
        decision_dates:      决策日列表
        used_data:          {数据名: DataFrame}，须含 "_available" 列
        raise_on_violation: True=抛 AssertionError；False=仅打印警告
        trade_same_day_ok:  True=允许决策日使用当日收盘数据（收盘撮合惯例）；
                            False=严格只用前一日数据

    返回:
        违规数量（0 = 通过）
    """
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
