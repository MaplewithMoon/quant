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
# 非年度数据集的占位分区
# ============================================================
# 有些数据集天然不是"按年切片"的（股票列表、ST 名称变更、停牌记录、行业分类、
# 指数成分、三张报表……），历史上统一塞进 `year=2005` 这个占位分区。
#
# ⚠️ 这个约定本身没问题，但**读取方不能按回测区间去拼年份路径**：
#     用 `year=2024` 去 glob `frozen/suspend` 会拼出一个不存在的路径，
#     DuckDB 抛 IOException；而调用方往往 `except Exception: 返回空`，
#     于是**停牌数据在每一次组合回测里都是空的**（本项目中真实发生过）。
# 统一用 `year_globs()` 生成路径，它会先看磁盘上真实存在哪些分区。
NON_ANNUAL_YEAR = 2005


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
    "fund_daily":  "tushare 基金/ETF 日K（与 etf 的月度快照分开存，避免粒度混用）",
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
        "source": "cleaned/daily_basic（tushare 官方 pre_close，已按除权调整）",
        "rules": ("科创板/创业板±20% / 北交所±30% / 主板ST±5% / 主板±10%；"
                  "创业板 2020-08-24 前 ±10%；"
                  "上市初期特殊规则：科创板、创业板(2020-08-24 起)、"
                  "主板(2023-04-10 起) 前 5 个交易日**不设涨跌幅**（limit 留空），"
                  "主板 2013-12-13~2023-04-09 首日 ×1.44/×0.64（基准=发行价），"
                  "主板与创业板开板初期首日不设限；"
                  "按昨收（除权后参考价）四舍五入到分"),
        "producer": "scripts/rebuild_limit.py（全量）、scripts/daily_update.py（增量）",
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

    若传入的路径**已经**是某个年份分区（含 `year=`），则只补 `*.parquet`——
    否则会拼出 `year=2025/year=*/*.parquet` 这种不存在的路径，DuckDB 直接抛
    `IOException: No files found that match the pattern`。
    """
    p = Path(path)
    if p.is_absolute():
        base = p
    else:
        base = PROJECT_ROOT / p
    if any(part.startswith("year=") for part in base.parts):
        return (base / "*.parquet").as_posix()
    return (base / "year=*" / "*.parquet").as_posix()


def year_globs(path, y0: int = None, y1: int = None) -> str:
    """生成 DuckDB `read_parquet()` 用的分区 glob 列表，**只包含磁盘上真实存在的分区**

    为什么不能直接按区间拼路径
    --------------------------
    1. **非年度数据集**（stocks / st / suspend / financial / industry / index_cons /
       dividend / holders …）只有占位分区 `year=2005`。按回测区间去拼会得到
       `year=2024/*.parquet` —— 不存在，DuckDB 抛 `IOException: No files found`；
       调用方若用 `except Exception` 吞掉，数据就**静默变空**。
    2. **年度数据集起始年份可能晚于区间起点**（fund_daily 从 2013、margin 从 2010、
       northbound 只有 2025+）。缺失的年份同样会让**整条**查询失败，
       而不是"少几个分区"。

    规则:
        - 只有占位分区 `year=2005` -> 视为静态数据集，忽略区间，直接用它
        - 否则 -> 取落在 [y0, y1] 内**且存在**的分区
        - 区间内一个都没有 -> 回退到全部分区（让 SQL 的日期条件给出空结果，
          而不是抛异常）
    """
    import os

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if any(part.startswith("year=") for part in p.parts):
        return f"['{(p / '*.parquet').as_posix()}']"
    if not p.exists():
        return "[]"
    years = []
    for e in os.scandir(p):
        if e.is_dir() and e.name.startswith("year="):
            try:
                years.append(int(e.name[5:]))
            except ValueError:
                continue
    if not years:
        return "[]"
    years.sort()
    if years == [NON_ANNUAL_YEAR]:
        sel = years                       # 静态数据集，与区间无关
    else:
        sel = [y for y in years if (y0 is None or y >= y0)
               and (y1 is None or y <= y1)]
        if not sel:
            sel = years                   # 区间内没有分区 -> 交给 WHERE 过滤
    return "[" + ", ".join(
        f"'{(p / f'year={y}' / '*.parquet').as_posix()}'" for y in sel) + "]"


def connect_duckdb(db_path=None, threads: int = 4, **kwargs):
    """统一的 DuckDB 连接工厂

    存在的意义：`SET enable_progress_bar=false` 漏在任何一个连接上，扫描
    parquet 时就会往 stderr 刷进度条，把日志/CI 输出冲得没法看。与其在每个
    `duckdb.connect()` 后面记得补一句，不如只留这一个入口。

    参数:
        db_path: None = 内存库（只读查询用）；否则打开/创建该文件
        threads: 扫描线程数
    """
    import duckdb

    con = duckdb.connect(str(db_path)) if db_path else duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={int(threads)}")
    except Exception:
        pass
    con.execute("SET enable_progress_bar=false")
    for k, v in kwargs.items():
        con.execute(f"SET {k}={v}")
    return con


def write_manifest() -> Path:
    """把 `db/_MANIFEST.json` **从本文件生成**（不要手写）

    原先这个文件是手写的一次性说明，既没有生成方也没有读取方，内容很快就和
    `RECIPES` 漂移了 —— 它的 `limit_price.rules` 还停在"主板±10%/创业板±20%/
    ST±5%"，缺少创业板 2020-08-24 前 ±10%、上市初期规则、ST 只对主板生效等
    三条；`producer` 也和 `RECIPES` 不一致。读到它的人会拿到**错的规则**。
    改成生成之后就不可能再漂移。
    """
    import json
    from datetime import datetime
    manifest = {
        "generated_by": "database/config.py::write_manifest()",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warning": "本文件由代码生成，请勿手改；要改规则请改 database/config.py",
        "layer_rules": {
            "frozen": "只读原始层：数据源原样落盘，仅下载器可写（Storage allow_frozen=True）",
            "cleaned": "加工层：可从 frozen 重建，按清洗配方分子目录",
            "_legacy": "历史产物（frozen 层建立之前的旧管线），只归档不再更新，可安全删除",
        },
        "frozen_datasets": FROZEN_DATASETS,
        "cleaned_recipes": RECIPES,
        "dataset_to_recipe": DATASET_TO_RECIPE,
        "non_annual_year": NON_ANNUAL_YEAR,
    }
    path = DB_ROOT / "_MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def setup_database():
    """创建数据库目录结构（并把 _MANIFEST.json 从本文件重新生成）"""
    DB_ROOT.mkdir(parents=True, exist_ok=True)
    FROZEN_ROOT.mkdir(parents=True, exist_ok=True)
    CLEANED_ROOT.mkdir(parents=True, exist_ok=True)
    LEGACY_ROOT.mkdir(parents=True, exist_ok=True)
    for name in FROZEN_DATASETS:
        frozen_dir(name).mkdir(parents=True, exist_ok=True)
    for name in RECIPES:
        recipe_dir(name).mkdir(parents=True, exist_ok=True)
    write_manifest()
    return DB_ROOT
