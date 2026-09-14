# 量化交易系统框架

## 文档导航

| 文档 | 内容 |
|---|---|
| [`docs/快速入门.md`](docs/快速入门.md) | 5分钟上手：环境/回测/写策略 |
| [`docs/架构图.md`](docs/架构图.md) | 分层架构 + 模块依赖图 + 调用链 |
| [`docs/数据字段说明.md`](docs/数据字段说明.md) | 16个数据集字段说明 + 读取案例 |

## 快速开始

```bash
pip install -r requirements.txt
python scripts/backtest.py --symbol 300750 --strategy MA   # 单策略回测
python scripts/backtest.py --symbol 300750 --compare       # 多策略对比
python scripts/backtest_demo.py                            # 6个教学案例
python scripts/run_tests.py                                # 单元测试
```

## 数据模型（三层：frozen 原始层 / cleaned 加工层 / _legacy 归档）

**分层架构：frozen 只读可信源（数据源原样），cleaned 按"清洗配方"分目录且可重建**

```
db/
├── _MANIFEST.json          全库说明（层级 / 规则 / 只读约定）
│
├── frozen/                 ① 原始层 —— 只读，仅下载器可写
│   ├── daily_raw/          tushare 不复权原始价（可信源）
│   ├── adjust/             tushare 原始复权因子 (adj_factor)
│   ├── valuation/          tushare 每日估值（原始单位：万元 / turnover_rate）
│   ├── st/  dividend/  suspend/  stocks/  calendar/
│   ├── financial/  holders/  margin/  northbound/  industry/
│   ├── index_cons/  index_daily/  etf/  futures/  options/
│   └── _MANIFEST.json
│
├── cleaned/                ② 加工层 —— 按清洗配方分子目录，可从 frozen 重建
│   ├── _RECIPES.json       配方登记表（来源 / 规则 / 产出脚本）
│   ├── daily_basic/        配方 daily_basic：清洗后日线（原始价 + 原始量额）
│   └── limit_price/        配方 limit_price：涨跌停价（由 daily_basic 派生）
│
└── _legacy/                ③ 历史归档 —— frozen 层建立前的旧管线产物，不再更新
    ├── _README.md
    ├── valuation_norm/     旧估值（市值已换算成亿元、turnover_rate→turnover）
    ├── st_flagged/         旧 ST（多一列派生 is_st）
    └── etf_akshare/        旧 ETF 行情（akshare 源，中文列名）
```

**为什么按"清洗配方"分目录**：同一份原始数据用不同方式加工，口径完全不同
（估值有"万元原样"和"亿元归一"、日线有"不复权"和"前复权"）。混在一个目录里
就无法判断读到的是哪一种——本项目历史上 `db/daily` 就同时被写入过前复权价和
不复权价，导致 `load_daily(adjust='qfq')` 出现双重复权。现在一种清洗方式独占一个
目录，目录名 = 配方名。

**frozen 层规则**（由代码强制，不只是约定）：
- 禁止修改/删除已有记录，只允许增量追加新交易日
- 清洗/转换必须输出到 `cleaned/` 下的配方目录
- 分析/回测代码写 frozen 会直接抛 `FrozenWriteError`
  （只有下载器用 `Storage(..., allow_frozen=True)` 才能写）

**新增一种清洗方式**：在 `database/config.py` 的 `RECIPES` 里登记一条，
产出目录自动变成 `db/cleaned/<配方名>/`，详见 [`docs/数据结构说明.md`](docs/数据结构说明.md)。

**下载脚本**：
```bash
python scripts/download_tushare_raw.py        # 全市场不复权原始价 → frozen/daily_raw
python scripts/download_frozen_tushare.py     # adjust/st/valuation/etf/options 原始数据
python scripts/download_frozen_tushare.py --only adjust
```

**前复权计算**（从 frozen 原始价 + 因子现场算）：
```python
from database.loader import load_daily
raw = load_daily('600519', '2024-01-01', '2024-06-30', adjust='')    # 原始价
qfq = load_daily('600519', '2024-01-01', '2024-06-30', adjust='qfq') # 前复权
frozen = load_daily_qfq('600519')  # 读 frozen 原始（load_daily_qfq 现指 frozen/daily_raw）
```

或通过框架 DataSet：
```python
from data.dataset import DataSet
ds = DataSet.from_db('600519', '2024-01-01', '2024-06-30')  # 默认前复权
```

**数据校验工具**：
```bash
python scripts/clean_data.py --scan    # 扫描逻辑问题（OHLC/非负/三角/复权因子跳变）
python scripts/clean_data.py --clean   # 清理脏行
python scripts/validate_data.py        # 主键唯一/Schema/覆盖率
```

**缺失值处理策略**（已实施）：

| 缺失类型 | 处理 | 当前状态 |
|---|---|---|
| 停牌导致缺失 | 不填充，停牌日无K线（删除了占位K线）；停牌事件在 `db/frozen/suspend/` | ✅ 全天停牌无K线 |
| 上市前/退市后 | 无记录，用NaN而非0 | ✅ 全库价格0值=0 |
| 因子/财务真缺失 | 保留NaN（亏损股PE无意义），`fill_factor_nan()` 可按行业中位数填充 | ✅ 24.5% PE NaN 保留 |
| 单根K线缺失 | 无插值（缺失即无记录），不伪造数据 | ✅ 天然符合 |

**加载时缺失处理助手**：
```python
from database.loader import load_daily, mark_suspended, fill_factor_nan, load_valuation

df = load_daily('600519', '2024-01-01', '2024-06-30', adjust='qfq')
df = mark_suspended(df, '600519')          # 加 suspended 停牌标记列
v = load_valuation('600519', '2024-01-01')  # 估值数据
```

**历史遗留**：`db/daily_qfq/` 保留旧的前复权+原始量额混合数据，仅作参考，新数据请用 `db/daily/`。

---

## 目录结构

```
quant/
├── main.py                       # 程序入口 — 回测demo入口
│
├── database/                     # 量化数据库（Parquet + DuckDB）
│   ├── config.py                 # 路径/磁盘阈值/数据集目录
│   ├── storage.py                # Parquet分区存储 + 断点记录
│   ├── query.py                  # DuckDB 查询层
│   └── downloader/               # 各数据集下载器
│       ├── base.py               # 基类（重试/限流/断点/磁盘监控）
│       ├── calendar.py           # 交易日历
│       ├── stocks.py             # 股票列表
│       ├── daily.py              # 日线（双源交叉校验）
│       ├── adjust.py             # 复权因子 + 分红送转
│       ├── valuation.py          # 每日估值（tushare/百度）
│       ├── meta.py               # 停牌/涨跌停价
│       ├── indices.py            # 指数成分/日K/行业分类
│       ├── financial.py          # 三大财务报表
│       └── other.py              # 两融/北向/股东/ETF/期货/期权
│
├── scripts/
│   └── run_download.py           # 数据库下载主入口
│
├── data/                         # 数据层
│   ├── __init__.py
│   ├── source.py                 # DataSource 抽象 + Local/Remote/Mock 实现
│   ├── dataset.py                # DataSet 核心数据容器（OHLCV + 衍生属性）
│   └── preprocessor.py           # 预处理流水线 + 11个技术指标 + help_indicators()
│
├── strategy/                     # 策略层
│   ├── __init__.py
│   ├── base.py                   # Strategy 基类 + Signal/Action 定义
│   ├── examples.py               # 内置示例：双均线交叉、均值回归
│   ├── classic.py                # 经典策略：海龟/布林带/MACD/双均线(量过滤)
│   └── modern.py                 # 现代策略：DualThrust/网格/RSI背离/突破回踩/自适应均线
│
├── backtest/                     # 回测层
│   ├── __init__.py
│   ├── engine.py                 # 回测引擎（逐bar驱动 + 风控集成）
│   └── metrics.py                # 绩效评估（14项指标）+ help_metrics()
│
├── execution/                    # 执行层
│   ├── __init__.py
│   ├── order.py                  # 订单模型（类型/方向/状态机）
│   └── broker.py                 # Broker 抽象 + SimulatedBroker（滑点/佣金）
│
├── risk/                         # 风控层
│   ├── __init__.py
│   └── manager.py                # 风控规则链（最大回撤/仓位限制）
│
├── portfolio/                    # 投资组合
│   ├── __init__.py
│   └── portfolio.py              # 持仓管理/现金/PnL/历史峰值回撤
│
├── analytics/                    # 分析可视化
│   ├── __init__.py
│   └── visualizer.py             # 净值曲线/回撤图/收益分布/指标表
│
├── data_statistics/              # 数据统计
│   ├── __init__.py
│   └── stats.py                  # 单股/多股对比统计表格
│
├── config/                       # 配置
│   ├── __init__.py
│   └── settings.py               # 全局参数（资金/手续费/滑点/风控阈值）
│
├── utils/                        # 工具
│   ├── __init__.py
│   ├── logger.py                 # 日志（控制台 + 文件）
│   └── helpers.py                # help_all() 统一指标帮助 + 工具函数
│
└── requirements.txt              # 依赖库
```

---

## 完成进度

### ✅ 已完成

| 模块 | 状态 | 备注 |
|------|------|------|
| `data/source.py` | ✅ | `DataSource` 抽象 + `RemoteSource`(akshare) + `MockSource`(模拟) + `LocalSource`(本地csv) |
| `data/dataset.py` | ✅ | OHLCV 容器 + 重采样 + 特征添加 + 工厂加载方法 |
| `data/preprocessor.py` | ✅ | 链式流水线 + fillna/去噪/SMA/EMA/MACD/RSI/布林带/ATR |
| `strategy/base.py` | ✅ | `Strategy` 基类 + `Signal`/`Action` 定义 |
| `strategy/examples.py` | ✅ | 双均线交叉 + 均值回归 |
| `strategy/classic.py` | ✅ | 海龟交易法 + 布林带回归 + MACD背离 + 量过滤均线 |
| `backtest/engine.py` | ✅ | 逐bar回测，集成风控，含交易记录 |
| `backtest/metrics.py` | ✅ | 收益/夏普/回撤/胜率/盈亏比 |
| `execution/order.py` | ✅ | 订单模型 + 状态机 |
| `execution/broker.py` | ✅ | 模拟券商（滑点/佣金） |
| `risk/manager.py` | ✅ | 最大回撤 + 仓位限制规则链 |
| `portfolio/portfolio.py` | ✅ | 多标的持仓 + 现金 + 历史峰值回撤 |
| `analytics/visualizer.py` | ✅ | matplotlib 可视化（4种图表） |
| `config/settings.py` | ✅ | 全局配置数据类 |
| `utils/logger.py` | ✅ | 控制台+文件日志 |
| `utils/helpers.py` | ✅ | 年化/时间转换/格式化 |
| `main.py` | ✅ | 端到端demo（远端优先→Mock回退） |

### 📋 待完成（按优先级）

#### P0 — 立刻可以做

| # | 任务 | 说明 | 参考文件 |
|---|------|------|---------|
| 1 | **写自己的策略** | 继承 `Strategy` 实现 `on_bar()` | `strategy/classic.py` |
| 2 | **换真实数据源** | 解决网络问题 / 改用 baostock / 或加代理 | `data/source.py` |

#### P1 — 完善框架

| # | 任务 | 说明 | 预计文件 |
|---|------|------|---------|
| 3 | **数据缓存层** | 首次远程拉取 → 本地 csv 落盘，下次优先读本地 | `data/cache.py` |
| 4 | **多标的回测** | 支持多股票组合/轮动 | `backtest/engine.py` |
| 5 | **单元测试** | pytest 覆盖每个模块 | `tests/` |
| 6 | **参数优化器** | 网格搜索/随机搜索策略参数 | `optimizer/` |

#### P2 — 进阶功能

| # | 任务 | 说明 |
|---|------|------|
| 7 | **实盘/模拟交易** | 定时 + 券商 API 对接 |
| 8 | **因子库** | 统一因子计算/存储/IC 分析 |
| 9 | **回测报告 HTML** | 导出静态报告 |
| 10 | **事件驱动重构** | tick/分钟级事件循环 |

---

## 量化数据库

### 启动下载

```bash
# 全量按优先级下载（断点续跑，磁盘<5GB自动停止）
python scripts/run_download.py

# 只跑日线
python scripts/run_download.py --only daily

# 指定股票
python scripts/run_download.py --only daily --codes "000001,600519"   # PowerShell下需引号

# 忽略断点重下
python scripts/run_download.py --fresh
```

### 数据源说明

| 数据集 | 主源 | 次源 | 说明 |
|---|---|---|---|
| 交易日历/股票列表 | tushare | — | `trade_cal` / `stock_basic` |
| 个股日线(不复权) | tushare | — | `daily`，落 `frozen/daily_raw`（量=股 / 额=元） |
| 复权因子/分红 | tushare | — | `adj_factor` / `dividend` |
| 每日估值 | tushare | akshare百度 | `daily_basic`；无 TUSHARE_TOKEN 时仅 PE/PB/总市值 |
| 财务报表 | tushare | — | `balancesheet`/`income`/`cashflow` |
| 指数/行业 | tushare | — | 成分权重/日K/申万分类 |
| 涨跌停价 | 由清洗层派生 | — | 配方 `limit_price`，落 `cleaned/limit_price` |

### 查询示例

```python
from database.query import QueryEngine
from database.config import dir_of, parquet_glob

qe = QueryEngine()

# 单只股票某年（清洗后日线）
df = qe.query(f"""
    SELECT trade_date, open, high, low, close
    FROM '{parquet_glob(dir_of('daily') / 'year=2024')}'
""")

# 跨年查询全部
df = qe.query(f"""
    SELECT code, count(*) AS n
    FROM read_parquet('{parquet_glob(dir_of('daily'))}')
    GROUP BY code
""")

# 估值 + 行情 表连接
df = qe.query(f"""
    SELECT d.trade_date, d.close, v.pe_ttm, v.total_mv
    FROM '{parquet_glob(dir_of('daily') / 'year=2024')}' d
    JOIN '{parquet_glob(dir_of('valuation') / 'year=2024')}' v
      ON d.trade_date = v.trade_date
""")
```

> 注意：**不要手写 `'db/xxx/...'` 路径字符串**，一律用
> `database.config.dir_of()` 取目录、`parquet_glob()` 生成 DuckDB glob。
> 这样以后调整目录结构时只需改 `config.py` 一处。

### 存储结构

```
db/
├── calendar/year=YYYY.parquet
├── stocks/year=2005/data.parquet
├── daily_qfq/year={YYYY}/{code}.parquet      # 前复权日线
├── valuation/year={YYYY}/{code}.parquet
├── adjust/year={YYYY}/{code}.parquet         # 复权因子
├── financial/{sheet}/year={YYYY}/{sheet}_{code}.parquet
└── ...每个目录下有 _checkpoint.json 记录断点
```

---

## 回测使用

**CLI 入口**（`scripts/backtest.py`，数据从数据库取）：
```bash
# 单策略回测（默认前复权）
python scripts/backtest.py --symbol 300750 --strategy MA --params '{"fast":5,"slow":20}'
python scripts/backtest.py --symbol 600519 --strategy Turtle --start 2020-01-01 --end 2024-12-31

# 多策略对比
python scripts/backtest.py --symbol 300750 --compare

# 风控 + 可视化
python scripts/backtest.py --symbol 300750 --strategy MA --risk --plot

# JSON 输出
python scripts/backtest.py --symbol 000001 --strategy BBand --json
```

**试用案例**（6个完整示例）：
```bash
python scripts/backtest_demo.py
```

**可用策略**（`STRATEGIES` 注册表）：

| 名称 | 策略 | 默认参数 |
|---|---|---|
| `MA` | 双均线交叉 | fast=5, slow=20 |
| `MAVol` | 量过滤双均线 | fast=10, slow=30 |
| `MeanRev` | 均值回归 | window=20, entry_std=2 |
| `BBand` | 布林带 | window=20, num_std=2 |
| `MACD` | MACD金叉死叉 | signal_period=9 |
| `Turtle` | 海龟交易法 | entry_window=20, exit_window=10 |
| `DualThrust` | 通道突破 | k1=0.5, k2=0.5 |
| `RSIDiv` | RSI背离 | period=14, oversold=30, overbought=70 |
| `Pullback` | 突破回踩 | lookback=20, ma_period=10 |
| `AMA` | 自适应均线 | fast=2, slow=30 |
| `Grid` | 网格交易 | grids=10 |

**编程式调用**：
```python
from data.dataset import DataSet
from data.preprocessor import Preprocessor, fillna, add_technical_indicators
from backtest.engine import BacktestEngine
from backtest.metrics import Metrics

ds = DataSet.from_db("300750", "2024-01-01", "2024-12-31", adjust="qfq")
ds = Preprocessor().add(fillna()).add(add_technical_indicators).run(ds)
engine = BacktestEngine(initial_capital=1_000_000)
trades = engine.run(ds, MyStrategy())
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行回测demo（网络不通则自动用模拟数据）
python main.py

# 3. 跑多策略对比（在 main.py 中取消 run_comparison() 注释）
# 4. 在 strategy/ 下编写自己的策略
# 5. 修改 main.py 中的 strategy 和 symbol 进行回测
```

---

## 新增策略模板

```python
# strategy/my_strategy.py
from .base import Strategy, Signal, Action

class MyStrategy(Strategy):
    def __init__(self, param=20):
        super().__init__("MyStrategy")
        self.param = param

    def on_bar(self, data, idx):
        if idx < self.param + 1:
            return Signal(Action.HOLD)
        # 你的择时逻辑
        if ...:
            return Signal(Action.BUY, reason="buy")
        elif ...:
            return Signal(Action.SELL, reason="sell")
        return Signal(Action.HOLD)
```
