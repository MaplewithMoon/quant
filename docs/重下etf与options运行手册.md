# =============================================================================
#  重下 frozen/etf 与 frozen/options —— 运行手册
#
#  背景（详见 docs/已知问题与待办汇总.md 的 B12/B17 与 2.11）
#    旧下载器三个错叠加，把这两个数据集写坏了：按"每月 1 号"取数（1 号常非
#    交易日）、空响应也标记完成（永不再试）、save_by_year 不传 code 导致每年
#    都写同一个 data.parquet 被后一天覆盖。实测 2020~2026 每年只剩 1 天数据。
#    另外 opt_daily 按 trade_date 一把取恒定 15,000 行（单次上限），
#    按交易所拆分后是 26,566 行 —— 不拆就每天丢 44%。
#
#  代码已修好，并且做过单日实跑验证：
#    fund_daily(20260915) -> 2,139 行
#    opt_daily(20260915) 按 8 个交易所拆 -> 26,566 行（无重叠）
#
#  估计耗时：etf 1,628 次调用 @60/分 ≈ 27 分钟
#            options 13,024 次调用 @100/分 ≈ 2.2 小时
#            合计 ≈ 2.6 小时（可中断，断点续跑）
# =============================================================================

# ---- 0) 准备环境（每个新 shell 都要）----
$env:TUSHARE_TOKEN = [Environment]::GetEnvironmentVariable('TUSHARE_TOKEN','Machine')
$env:PYTHONUTF8 = 1
$env:PYTHONIOENCODING = 'utf-8'
Set-Location D:\quant

# ---- 1) 先看要下多少（不写数据）----
python -c "from database.calendar import trading_days as t; d=t(start='2020-01-01'); print(f'交易日 {len(d)} 个 -> etf {len(d)} 次调用(~{len(d)/60:.0f}分), options {len(d)*8:,} 次调用(~{len(d)*8/100/60:.1f}小时)')"

# ---- 2) 重下 etf（约 30 分钟）----
#   --fresh --only etf 只清 etf 自己（已修：早期写法会连带删掉 adjust/st/valuation）
python scripts/download_frozen_tushare.py --only etf --fresh

# ---- 3) 重下 options（约 2.2 小时）----
#   建议放后台并留日志，中途 Ctrl-C 也安全：已完成的日子记在断点里，重跑会跳过
python scripts/download_frozen_tushare.py --only options --fresh *> results\download_options.log

# 或者用后台任务（不占着终端）：
#   Start-Job -ScriptBlock { Set-Location D:\quant; python scripts/download_frozen_tushare.py --only options --fresh *> results\download_options.log }

# ---- 4) 下完后的校验 ----
python scripts/validate_data.py --check checkpoint --backfill-checkpoint
python scripts/validate_data.py --check freshness
#   期望：etf/options 的"确证缺失"为 0；新鲜度不再超硬上限
#   还会提示"哪些数据集没有消费方"—— etf/options 仍会列在那里（这是事实）

# ---- 5) 看结果 ----
python -c "
from pathlib import Path
from database.config import connect_duckdb
con = connect_duckdb()
for ds in ('etf','options'):
    g = f'db/frozen/{ds}/year=*/*.parquet'
    n, a, b, days = con.execute(f\"SELECT count(*), min(trade_date), max(trade_date), count(DISTINCT trade_date) FROM read_parquet('{g}')\").fetchone()
    print(f'{ds:<8} {n:>9,} 行  {str(a)[:10]} ~ {str(b)[:10]}  {days:,} 个交易日')
"

# =============================================================================
#  可能遇到的情况
#  - "有 N 个交易日返回空 —— 未标记完成"：正常（当天数据还没发布等），
#    再跑一次就会补上，不会丢数据。
#  - "有 N 次调用触到单次上限 15000"：说明还需要再细分维度，把这条日志发我。
#  - 被限频打断：api_call 自带指数退避重试；持续失败的那天不会标完成，
#    重跑即续。
#  - 想中途看进度：Get-Content results\download_options.log -Tail 5
# =============================================================================
