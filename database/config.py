# -*- coding: utf-8 -*-
"""数据库路径与配置：分层结构 + 清洗配方注册表

目录布局
--------
    db/
    ├── _MANIFEST.json          全库说明（层级 / 规则 / 只读约定）
    ├── frozen/                 ① 原始层：数据源原样落盘，**只读**
    │   └── <dataset>/year=YYYY/<code>.parquet
    ├── cleaned/                ② 加工层：可从 frozen 重建，按"清洗配方"分子目录
    │   ├── _RECIPES.json
    │   └── <recipe>/year=YYYY/<code>.parquet
    └── _legacy/                ③ 历史产物（frozen 层建立之前的旧管线），只归档不再更新

为什么按"清洗配方"分目录
------------------------
同一份原始数据用不同方式加工，结果口径完全不同：
  - 估值：frozen 是"万元原样 / turnover_rate"，旧管线产出的是"亿元归一 / turnover"
  - 日线：可以是"不复权原始价"，也可以是"前复权价"
  - 涨跌停：可以由清洗层派生，也可以由其它规则派生

这些口径混在同一个目录里，读取方就无法判断拿到的是哪一种——本项目历史上
`db/daily` 就同时被写入过前复权价和不复权价，导致 `load_daily(adjust='qfq')`
出现"双重复权"。因此规定：**一种清洗方式独占一个目录，目录名 = 配方名**，
并在 `RECIPES` 里登记它的来源、规则和产出脚本。

只读约定
--------
`frozen/` 只允许"下载器"追加写入（`Storage(..., allow_frozen=True)`），
清洗 / 分析 / 回测代码一律只能读。`Storage` 会在写入前做强制校验。
"""
from pathlib import Path

# ============================================================
# 根路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_ROOT = PROJECT_ROOT / "db"

FROZEN_ROOT = DB_ROOT / "frozen"      # ① 只读原始层
CLEANED_ROOT = DB_ROOT / "cleaned"    # ② 加工层（按配方分目录）
LEGACY_ROOT = DB_ROOT / "_legacy"     # ③ 历史归档

DB_DUCKDB = DB_ROOT / "quant.duckdb"

# 磁盘空间保护：剩余 < 该阈值则停止下载（默认 5GB）
DISK_MIN_FREE = 5 * 1024 ** 3

# 目标数据起始年份（从新往旧拉取到这一年）
DATA_START_YEAR = 2005


# ============================================================
# frozen 层：原始数据集（只读）
# ============================================================
FROZEN_DATASETS = {
    "calendar":    "tushare 交易日历",
    "stocks":      "tushare 股票列表（含退市）",
    "daily_raw":   "tushare 不复权原始价（可信源，量=股 额=元）",
    "adjust":      "tushare 复权因子 adj_factor（原始）",
    "valuation":   "tushare 每日估值（原始单位：市值=万元，列名 turnover_rate）",
    "dividend":    "tushare 分红送转",
    "suspend":     "tushare 停牌记录",
    "st":          "tushare ST 名称变更",
    "financial":   "tushare 三大财务报表",
    "holders":     "tushare 股东户数",
    "margin":      "tushare 两融余额",
    "northbound":  "tushare 北向资金",
    "industry":    "tushare 行业分类（申万）",
    "index_cons":  "tushare 指数成分及权重",
    "index_daily": "tushare 指数日K",
    "etf":         "tushare 基金列表（fund_basic）",
    "futures":     "tushare 期货主连",
    "options":     "tushare 期权合约",
}


# ============================================================
# cleaned 层：清洗配方注册表
#   键 = 配方名（同时是 cleaned/ 下的目录名）
#   新增一种清洗方式时，在这里加一条，并让产出脚本用 Storage(recipe_dir=...)
# ============================================================
RECIPES = {
    "daily_basic": {
        "dataset": "daily",
        "desc": "基础清洗后的个股日线：原始价 + 原始量额（口径一致，三角校验通过）",
        "source": "frozen/daily_raw",
        "rules": ("丢弃 volume=0/NaN（幽灵K线）、价格≤0、OHLC 关系违规、"
                  "VWAP 三角校验越界（low*0.95 ~ high*1.05）"),
        "producer": "scripts/rebuild_cleaned.py, scripts/daily_update.py",
    },
    "limit_price": {
        "dataset": "limit",
        "desc": "涨跌停价（由清洗层日线派生）",
        "source": "cleaned/daily_basic",
        "rules": "主板±10% / 创业板·科创板±20% / 北交所±30% / ST±5%",
        "producer": "database/downloader/meta.py, scripts/daily_update.py",
    },
    # 已废弃的 akshare 日线下载器（DailyDownloader）的输出。
    # 单独开一个配方目录，避免它的前复权数据混进 daily_basic 造成双重复权。
    "daily_akshare_legacy": {
        "dataset": "daily_akshare",
        "desc": "【已废弃】akshare/baostock 双源日线，勿用于回测",
        "source": "(网络) akshare 新浪 / baostock",
        "rules": "双源交叉校验（close 差异 >0.5% 仅告警，仍取新浪）",
        "producer": "database/downloader/daily.py（已废弃）",
    },
}

# dataset 名 -> 配方名（便于 dir_of("daily") 这类旧写法继续工作）
DATASET_TO_RECIPE = {r["dataset"]: name for name, r in RECIPES.items()}


# ============================================================
# 路径解析：所有读写都必须走这里，禁止再手写 "db/xxx" 字符串
# ============================================================
class FrozenWriteError(RuntimeError):
    """试图写入只读的 frozen 层"""


def frozen_dir(dataset: str) -> Path:
    """frozen 层某个数据集的目录"""
    return FROZEN_ROOT / dataset


def recipe_dir(recipe: str) -> Path:
    """cleaned 层某个清洗配方的目录"""
    if recipe not in RECIPES:
        raise KeyError(f"未注册的清洗配方 {recipe!r}，可用: {list(RECIPES)}")
    return CLEANED_ROOT / recipe


def dir_of(dataset: str, recipe: str = None) -> Path:
    """统一路径路由（旧接口兼容）

    优先级: 显式 recipe > frozen 数据集名 > 配方产出的 dataset 名
    未知名字一律落到 cleaned/ 下，避免误写进只读的 frozen 层。
    """
    if recipe:
        return recipe_dir(recipe)
    if dataset in FROZEN_DATASETS:
        return FROZEN_ROOT / dataset
    if dataset in DATASET_TO_RECIPE:
        return CLEANED_ROOT / DATASET_TO_RECIPE[dataset]
    return CLEANED_ROOT / dataset


def is_frozen(path) -> bool:
    """判断路径是否位于只读的 frozen 层"""
    try:
        return FROZEN_ROOT in Path(path).resolve().parents
    except (OSError, ValueError):
        return False


def parquet_glob(path) -> str:
    """生成 DuckDB read_parquet() 用的 glob，统一走正斜杠

    例：parquet_glob(dir_of("daily")) -> 'D:/quant/db/cleaned/daily_basic/year=*/*.parquet'
    """
    p = Path(path)
    if p.is_absolute():
        base = p
    else:
        base = PROJECT_ROOT / p
    return (base / "year=*" / "*.parquet").as_posix()


def setup_database():
    """创建数据库目录结构"""
    DB_ROOT.mkdir(parents=True, exist_ok=True)
    FROZEN_ROOT.mkdir(parents=True, exist_ok=True)
    CLEANED_ROOT.mkdir(parents=True, exist_ok=True)
    LEGACY_ROOT.mkdir(parents=True, exist_ok=True)
    for name in FROZEN_DATASETS:
        frozen_dir(name).mkdir(parents=True, exist_ok=True)
    for name in RECIPES:
        recipe_dir(name).mkdir(parents=True, exist_ok=True)
    return DB_ROOT
