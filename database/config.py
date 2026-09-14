"""数据库路径与配置"""
import os
from pathlib import Path

# 数据库根目录（工程内）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_ROOT = PROJECT_ROOT / "db"
DB_DUCKDB = DB_ROOT / "quant.duckdb"

# 磁盘空间保护：剩余 < 该阈值则停止下载（默认 5GB）
DISK_MIN_FREE = 5 * 1024 ** 3

# 目标数据起始年份（从新往旧拉取到这一年）
DATA_START_YEAR = 2005

# 各数据集的存放目录
# frozen = 冻结原始层（只读可信源，未加工/tushare原样），工作层为清洗/派生数据
FROZEN = DB_ROOT / "frozen"
DIRS = {
    "calendar": FROZEN / "calendar",            # frozen: tushare交易日历
    "stocks": FROZEN / "stocks",                # frozen: tushare股票列表
    "daily": DB_ROOT / "daily",                 # 清洗后: 个股原始价日线（量额一致）
    "daily_raw": FROZEN / "daily_raw",          # frozen: tushare不复权原始价（可信源）
    "valuation": FROZEN / "valuation",          # frozen: tushare每日估值
    "adjust": FROZEN / "adjust",                # frozen: tushare复权因子（原始）
    "dividend": FROZEN / "dividend",            # frozen: tushare分红送转
    "suspend": FROZEN / "suspend",              # frozen: tushare停牌
    "st": FROZEN / "st",                        # frozen: tushare ST名称变更
    "limit": DB_ROOT / "limit",                 # 派生: 涨跌停价
    "index_cons": FROZEN / "index_cons",        # frozen: tushare指数成分及权重
    "index_daily": FROZEN / "index_daily",      # frozen: tushare指数日K
    "industry": FROZEN / "industry",            # frozen: tushare行业分类
    "financial": FROZEN / "financial",          # frozen: tushare财务报表
    "margin": FROZEN / "margin",                # frozen: tushare两融
    "northbound": FROZEN / "northbound",        # frozen: tushare北向
    "holders": FROZEN / "holders",              # frozen: tushare股东户数
    "etf": FROZEN / "etf",                      # frozen: ETF/LOF
    "futures": FROZEN / "futures",              # frozen: 期货主连
    "options": FROZEN / "options",              # frozen: ETF期权
}


def setup_database():
    """创建数据库目录结构"""
    DB_ROOT.mkdir(parents=True, exist_ok=True)
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    return DB_ROOT


def dir_of(dataset: str) -> Path:
    return DIRS.get(dataset, DB_ROOT / dataset)
