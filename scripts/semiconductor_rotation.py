# -*- coding: utf-8 -*-
"""半导体行业低PE轮动策略（教学案例）

【策略逻辑】
    每隔10个交易日（重平衡日），在半导体行业中重新选股：
        1. 剔除 市盈率(pe_ttm) 缺失或为负 的股票（亏损股无意义）
        2. 剔除 市值 处于同行业倒数10% 的股票（排除小市值壳股）
        3. 在剩余股票中，按 pe_ttm 从小到大排序，选前 10 只
    选完后调整持仓：
        卖出 已不在新列表 的股票
        买入 新进入列表 的股票（等权配置：每只占总资金 1/10）

【运行】
    python scripts/semiconductor_rotation.py
"""
import sys
import io
sys.path.insert(0, ".")
if __name__ == "__main__":
    # 只在直接运行时切编码：模块顶层替换 sys.stdout 是有副作用的 import，
    # 会破坏 pytest 的输出捕获（详见 utils/console.py）
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

import glob
import numpy as np
import pandas as pd
from database.loader import load_daily, load_valuation, load_industry_map

# ============================================================
# 0. 参数配置（所有可调项集中在这里）
# ============================================================
INITIAL_CAPITAL = 100_000        # 初始资金（元）
REBALANCE_EVERY = 10             # 每隔多少个交易日重平衡一次
TOP_N = 10                       # 持有股票数量
INDUSTRY = "半导体"              # 目标行业
CAP_BOTTOM_PCT = 0.10            # 剔除市值倒数10%的股票
START, END = "2024-01-01", "2024-12-31"   # 回测区间
COMMISSION = 0.0001              # 佣金（万分之一）
MIN_COMMISSION = 5.0             # 单笔最低佣金
IMPACT_COEFF = 0.1               # 平方根滑点模型系数（impact = k × √(Q/V)）
RISK_FREE = 0.03                 # 无风险利率（用于夏普/alpha计算）
BENCHMARK = "000300.SH"          # 基准指数（沪深300）


def strip_code(ts_code: str) -> str:
    """'002049.SZ' → '002049'（去掉交易所后缀）"""
    return str(ts_code).split(".")[0]


def sqrt_slippage(shares: float, daily_volume: float) -> float:
    """平方根滑点模型

    市场冲击成本与下单量占日成交量的比例成平方根关系：
        impact_pct = IMPACT_COEFF × √(Q / V)
    其中:
        Q = 下单股数（本次交易规模）
        V = 该股当日成交量（流动性）

    含义:
        - 下单量越大、相对当日成交量占比越高 → 冲击成本越大
        - 流动越差（V 小）的股票 → 同样下单量的冲击更大
    返回: 滑点比例（如 0.003 = 0.3%），买入加价、卖出折价
    """
    if daily_volume <= 0:
        return 0.0
    return IMPACT_COEFF * np.sqrt(shares / daily_volume)


# ============================================================
# 1. 数据准备
# ============================================================
print("=" * 60)
print("第1步：准备数据")
print("=" * 60)

# 1.1 获取半导体行业股票列表
ind = load_industry_map()                       # 全部股票-行业映射
semi = ind[ind["industry"].astype(str) == INDUSTRY]
semi_codes = [strip_code(t) for t in semi["ts_code"]]
print(f"半导体行业股票: {len(semi_codes)} 只")

# 1.2 加载这些股票的日线（前复权）——只取回测区间，减少内存
#     用 dict 存 {代码: DataFrame}，按需读取
daily_dict = {}
for code in semi_codes:
    try:
        d = load_daily(code, START, END, adjust="qfq")
        if not d.empty:
            daily_dict[code] = d
    except Exception:
        continue    # 无日线数据（退市/新股未上市）则跳过
print(f"有日线数据的股票: {len(daily_dict)} 只")

# 1.3 加载这些股票的估值（pe_ttm、total_mv）
#     估值按 (代码, 日期) 组织，重平衡日直接查表
val_frames = []
for code in semi_codes:
    try:
        v = load_valuation(code, START, END)
        if not v.empty:
            v = v[["trade_date", "pe_ttm", "total_mv"]].copy()
            v["code"] = code
            val_frames.append(v)
    except Exception:
        continue
valuation = pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame()
print(f"估值数据: {len(valuation)} 行, 覆盖 {valuation['code'].nunique() if not valuation.empty else 0} 只")

# 1.4 交易日序列（用所有股票的交易日并集）
#     load_daily 返回的 trade_date 是"列"，需转成日期集合
all_dates = sorted(set().union(*[set(pd.to_datetime(d["trade_date"])) for d in daily_dict.values()]))
all_dates = [d for d in all_dates if pd.Timestamp(START) <= d <= pd.Timestamp(END)]
print(f"回测交易日数: {len(all_dates)}")

# 1.5 加载基准指数（沪深300）——用于对比策略相对市场的表现
try:
    import glob as _glob
    bench_files = _glob.glob("db/frozen/index_daily/year=*/*.parquet")
    bench = pd.concat([pd.read_parquet(f) for f in bench_files], ignore_index=True)
    bench = bench[bench["index_code"] == BENCHMARK].copy()
    bench["trade_date"] = pd.to_datetime(bench["trade_date"])
    bench = bench.sort_values("trade_date").set_index("trade_date")["close"]
    # 基准净值 = 指数点位归一化 × 初始资金（与策略同起点可比）
    bench_equity = bench.loc[bench.index.intersection(pd.to_datetime(all_dates))] / \
        bench.loc[bench.index.intersection(pd.to_datetime(all_dates))].iloc[0] * INITIAL_CAPITAL
    print(f"基准 {BENCHMARK}: {len(bench_equity)} 个交易日")
except Exception as e:
    bench_equity = pd.Series(dtype=float)
    print(f"基准加载失败: {e}")


# ============================================================
# 2. 选股逻辑
# ============================================================
def select_stocks(trade_date):
    """在指定日期选出 TOP_N 只股票

    逻辑:
        1. 取该日各股的 pe_ttm / total_mv（估值表中 <= 该日的最近一条）
        2. 剔除 pe_ttm 缺失 或 pe_ttm<=0（亏损股）
        3. 计算行业市值倒数10%阈值，剔除市值过小者
        4. 按 pe_ttm 升序取前 TOP_N

    参数:
        trade_date: 重平衡日

    返回:
        (选中的股票代码列表, 本次使用的估值数据)
        估值数据用于回测后的"未来因子断言"——确认用的都是该决策日可得的数据
    """
    # 2.1 取该日各股估值（用 merge_asof 向后匹配最近一条 <= trade_date）
    if valuation.empty:
        return [], pd.DataFrame()
    v = valuation[valuation["trade_date"] <= trade_date]
    latest = (v.sort_values("trade_date")
               .groupby("code")
               .tail(1))                       # 每只股票取最近一条
    latest["pe"] = pd.to_numeric(latest["pe_ttm"], errors="coerce")
    latest["mv"] = pd.to_numeric(latest["total_mv"], errors="coerce")

    # 2.2 剔除 PE 缺失或为负
    valid = latest.dropna(subset=["pe"]).query("pe > 0")
    if valid.empty:
        return [], latest

    # 2.3 剔除市值在同行业倒数10%（按该日所有半导体股市值排序）
    mv_threshold = valid["mv"].quantile(CAP_BOTTOM_PCT)
    valid = valid[valid["mv"] > mv_threshold]

    # 2.4 按 PE 升序取前 TOP_N
    chosen = valid.nsmallest(TOP_N, "pe")
    return chosen["code"].tolist(), latest


# ============================================================
# 3. 回测主循环（轮动执行）
# ============================================================
print("=" * 60)
print("第2步：执行回测（每10个交易日重平衡一次）")
print("=" * 60)

# 3.1 初始化账户
cash = INITIAL_CAPITAL          # 现金
positions = {}                  # {代码: 股数}
equity_curve = []               # [(日期, 总资产)]
trades_log = []                 # [(日期, 方向, 代码, 股数, 价格, 佣金)]
decision_dates = []             # 记录每次重平衡决策日（未来因子断言用）
used_valuation_records = []     # 记录每次决策实际使用的估值（未来因子断言用）


def get_price(code, date):
    """从该股日线取指定日期的收盘价（trade_date 是列，非索引）"""
    d = daily_dict.get(code)
    if d is None:
        return None
    row = d[d["trade_date"] == date]
    return float(row["close"].iloc[0]) if not row.empty else None


def get_volume(code, date):
    """取该股指定日期的成交量（用于平方根滑点模型的流动性因子 V）"""
    d = daily_dict.get(code)
    if d is None:
        return 0.0
    row = d[d["trade_date"] == date]
    return float(row["volume"].iloc[0]) if not row.empty else 0.0

   
for i, date in enumerate(all_dates):
    # 3.2 计算当日持仓市值（用当日收盘价）
    market_value = 0.0
    for code, shares in positions.items():
        price = get_price(code, date)
        if price is not None:
            market_value += shares * price
    total_equity = cash + market_value

    # 3.3 每 REBALANCE_EVERY 个交易日触发一次重平衡
    rebalance_day = (i % REBALANCE_EVERY == 0)

    # 3.4 最后一天无论是否到周期都强制清仓（便于计算最终收益）
    if i == len(all_dates) - 1:
        rebalance_day = True

    if rebalance_day:
        target, used_valuation = select_stocks(date)
        decision_dates.append(date)              # 记录决策日（用于未来因子断言）
        if not used_valuation.empty:
            used_valuation_records.append(used_valuation)
        print(f"  {str(date)[:10]} 重平衡 → 选中 {len(target)} 只: {target}")

        # 3.4.1 卖出已不在目标列表的股票
        for code in list(positions.keys()):
            if code not in target:
                price = get_price(code, date)
                if price is not None:
                    shares = positions.pop(code)
                    # 平方根滑点：卖出按折价成交
                    volume = get_volume(code, date)
                    slip = sqrt_slippage(shares, volume)
                    exec_price = price * (1 - slip)
                    fee = max(shares * exec_price * COMMISSION, MIN_COMMISSION)
                    cash += shares * exec_price - fee
                    trades_log.append((date, "卖出", code, shares, exec_price, fee))

        # 3.4.2 计算当前应持有的等权目标仓位（总资产 / TOP_N）
        # 先补算剩余持仓市值，保证等权分母准确
        mv_now = 0.0
        for c, s in positions.items():
            p = get_price(c, date)
            if p is not None:
                mv_now += p * s
        equity_now = cash + mv_now
        target_value_per_stock = equity_now / TOP_N

        # 3.4.3 买入新进入列表的股票 / 调整已有持仓到目标仓位
        for code in target:
            price = get_price(code, date)
            if price is None:
                continue   # 该股当日无数据（停牌）则跳过
            desired_shares = int(target_value_per_stock // price)  # 手数向下取整

            if code in positions:
                # 已有持仓：若不足则补买（简化：只补不卖，避免频繁交易）
                diff = desired_shares - positions[code]
                if diff > 0:
                    # 平方根滑点：买入按加价成交
                    volume = get_volume(code, date)
                    slip = sqrt_slippage(diff, volume)
                    exec_price = price * (1 + slip)
                    fee = max(diff * exec_price * COMMISSION, MIN_COMMISSION)
                    if cash >= diff * exec_price + fee:
                        cash -= diff * exec_price + fee
                        positions[code] += diff
                        trades_log.append((date, "买入", code, diff, exec_price, fee))
            else:
                # 新进入：全量买入（同样适用平方根滑点加价）
                if desired_shares > 0:
                    volume = get_volume(code, date)
                    slip = sqrt_slippage(desired_shares, volume)
                    exec_price = price * (1 + slip)
                    fee = max(desired_shares * exec_price * COMMISSION, MIN_COMMISSION)
                    if cash >= desired_shares * exec_price + fee:
                        cash -= desired_shares * exec_price + fee
                        positions[code] = desired_shares
                        trades_log.append((date, "买入", code, desired_shares, exec_price, fee))

    # 3.5 记录当日总资产
    equity_curve.append((date, total_equity))

# 3.6 组装净值曲线
equity_df = pd.DataFrame(equity_curve, columns=["trade_date", "equity"])
equity_df["trade_date"] = pd.to_datetime(equity_df["trade_date"])
equity_df = equity_df.set_index("trade_date")["equity"]

trades_df = pd.DataFrame(trades_log, columns=["trade_date", "action", "code", "size", "price", "fee"])


# ============================================================
# 3.7 未来因子断言（look-ahead 检测）
#     确认每次重平衡决策只用了"该决策日之前或当日"可获得的数据
# ============================================================
print("=" * 60)
print("未来因子断言（检查是否用了未来的数据）")
print("=" * 60)
from backtest.lookahead import verify_point_in_time, attach_availability

# 3.7.1 逐决策配对校验：每次重平衡实际使用的估值，必须在该决策日已可获得
#      select_stocks 已保证用 <= 决策日的估值，这里做复核（防未来因子）
decision_pairs = []
for date, used in zip(decision_dates, used_valuation_records):
    if used is None or used.empty:
        continue
    used = attach_availability(used, trade_date_col="trade_date")
    decision_pairs.append((date, {"pe_ttm/市值(估值)": used}))

violations = verify_point_in_time(decision_pairs, trade_same_day_ok=True)

if violations:
    # 有未来泄漏 → 默认终止（raise_on_violation 语义）
    msg = f"\n[未来因子检测] 发现 {len(violations)} 处未来数据泄漏！前5条：\n"
    for v in violations[:5]:
        msg += (f"  决策日 {v[0].date()} 使用了 {v[1]} 中 {v[2]} 的数据"
                f"（可获得日 {v[3].date()}，晚于决策日）\n")
    raise AssertionError(msg)
print("=> [通过] 每次重平衡使用的估值均在决策日已可获得，无未来数据泄漏\n")

# 3.7.2 价格数据校验：策略在重平衡日用"当日收盘价"撮合（trade_date==决策日）
#      不会引用未来价格，此处确认无跨越决策日的价格引用
price_pairs = []
for date in decision_dates:
    used_price = pd.concat(
        [d[d["trade_date"] == date] for d in daily_dict.values()
         if not d[d["trade_date"] == date].empty],
        ignore_index=True,
    )
    if not used_price.empty:
        price_pairs.append((date, {"收盘价(日线)": attach_availability(used_price)}))
violations_p = verify_point_in_time(price_pairs, trade_same_day_ok=True)
if violations_p:
    raise AssertionError(f"价格引用存在未来泄漏: {len(violations_p)} 处")
print("=> [通过] 重平衡日引用的收盘价均不晚于决策日\n")


# ============================================================
# 4. 绩效评估（完整指标集：收益/风险/相对市场表现）
# ============================================================
print("=" * 60)
print("第3步：绩效评估")
print("=" * 60)


# 4.1 日收益率序列
strategy_ret = equity_df.pct_change().dropna()          # 策略日收益
bench_ret = bench_equity.pct_change().dropna()          # 基准日收益（沪深300）

# 4.2 对齐两者的交易日，计算相对市场的指标
merged_ret = pd.concat([strategy_ret.rename("strategy"), bench_ret.rename("bench")],
                       axis=1).dropna()
s_ret, b_ret = merged_ret["strategy"], merged_ret["bench"]

# 4.3 基础收益/风险指标
total_return = equity_df.iloc[-1] / INITIAL_CAPITAL - 1
n_days = len(s_ret)
annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
annual_vol = s_ret.std() * np.sqrt(252)                 # 年化波动率
sharpe = (annual_return - RISK_FREE) / annual_vol if annual_vol > 0 else 0

# 4.4 相对市场指标：alpha / beta
#     beta  = Cov(策略, 基准) / Var(基准)
#     alpha = 年化(策略收益 - 无风险) - beta × 年化(基准收益 - 无风险)
bench_ann = (1 + (b_ret + 1).prod() - 1) ** (252 / len(b_ret)) - 1  # 基准年化
beta = s_ret.cov(b_ret) / b_ret.var() if b_ret.var() > 0 else 0
alpha = (annual_return - RISK_FREE) - beta * (bench_ann - RISK_FREE)

# 4.5 下行风险指标
downside = s_ret[s_ret < 0]
downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252) if len(downside) > 0 else 0
sortino = (annual_return - RISK_FREE) / downside_dev if downside_dev > 0 else 0

# 4.6 跟踪误差 / 信息比率（衡量相对基准的稳定性）
tracking_error = (s_ret - b_ret).std() * np.sqrt(252)
excess_ret = annual_return - bench_ann
info_ratio = excess_ret / tracking_error if tracking_error > 0 else 0

# 4.7 回撤类指标（复用 Metrics）
from backtest.metrics import Metrics
metrics = Metrics.compute(equity_df, trades_df)
max_dd = metrics.max_drawdown
calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

# 4.8 输出全部指标
final_equity = equity_df.iloc[-1]
print(f"\n策略总收益:     {total_return:+.2%}     基准收益:  {bench_ann:+.2%}")
print(f"超额收益:       {excess_ret:+.2%}")
print(f"年化收益:       {annual_return:+.2%}")
print(f"年化波动率:     {annual_vol:.2%}")
print(f"夏普比率:       {sharpe:.2f}")
print(f"Sortino比率:    {sortino:.2f}")
print(f"最大回撤:       {max_dd:.2%}")
print(f"Calmar比率:     {calmar:.2f}")
print(f"Alpha(年化):    {alpha:+.2%}")
print(f"Beta:           {beta:.2f}")
print(f"跟踪误差:       {tracking_error:.2%}")
print(f"信息比率:       {info_ratio:.2f}")
print(f"交易次数:       {len(trades_df)} 笔   期末资产: {final_equity:,.2f} 元")

# 4.9 展示每次重平衡的选股变化
print("\n【各重平衡日的持仓变化】")
for date in equity_df.index[::REBALANCE_EVERY]:
    day_trades = trades_df[trades_df["trade_date"] == date]
    if not day_trades.empty:
        buys = day_trades[day_trades["action"] == "买入"]["code"].tolist()
        sells = day_trades[day_trades["action"] == "卖出"]["code"].tolist()
        print(f"  {str(date)[:10]}  买入: {buys}  卖出: {sells}")

# 4.10 可视化：策略 vs 沪深300 基准 + 完整指标表
import matplotlib
#matplotlib.use("Agg")   # 非交互后端，不弹窗
matplotlib.use('TkAgg') # 交互后端，弹窗
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(16, 10))
# --- 上：净值曲线（策略 vs 基准）---
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(equity_df.index, equity_df.values, label="策略净值", color="#1f77b4", linewidth=1.8)
ax1.plot(bench_equity.index, bench_equity.values, label="沪深300", color="#d62728",
         linewidth=1.2, linestyle="--")
ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle=":", linewidth=0.8)
ax1.set_title("半导体低PE轮动 vs 沪深300 基准", fontsize=14)
ax1.set_ylabel("资产(元)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# --- 右：指标表（指标名称 + 数值）---
# 用文本框把全部指标绘制在图中
metrics_text = (
    f"收益类\n"
    f"  总收益      {total_return:+.2%}\n"
    f"  基准收益    {bench_ann:+.2%}\n"
    f"  超额收益    {excess_ret:+.2%}\n"
    f"  年化收益    {annual_return:+.2%}\n"
    f"风险类\n"
    f"  年化波动率  {annual_vol:.2%}\n"
    f"  最大回撤    {max_dd:.2%}\n"
    f"  下行波动率  {downside_dev:.2%}\n"
    f"风险调整收益\n"
    f"  夏普比率    {sharpe:.2f}\n"
    f"  Sortino     {sortino:.2f}\n"
    f"  Calmar      {calmar:.2f}\n"
    f"相对市场\n"
    f"  Alpha       {alpha:+.2%}\n"
    f"  Beta        {beta:.2f}\n"
    f"  信息比率    {info_ratio:.2f}\n"
    f"  跟踪误差    {tracking_error:.2%}\n"
    f"交易\n"
    f"  交易次数    {len(trades_df)}\n"
    f"  期末资产    {final_equity:,.0f} 元"
)
ax1.text(0.99, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment="top", horizontalalignment="right",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

plt.tight_layout()
plt.savefig("半导体轮动_净值曲线.png", dpi=120)
print("\n已保存: 半导体轮动_净值曲线.png")
plt.plot()
plt.show()

# 4.11 保存结果
equity_df.to_csv("半导体轮动_净值曲线.csv")
trades_df.to_csv("半导体轮动_交易记录.csv")
print("结果已保存: 半导体轮动_净值曲线.csv / 半导体轮动_交易记录.csv")
