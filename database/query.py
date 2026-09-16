"""DuckDB 查询层：直接对 Parquet 分区做 SQL 查询

⚠️ **本模块目前没有任何调用方（2026-09 审计确认）**，保留仅为兼容历史代码。

    新代码请直接用 `database.config.connect_duckdb()` +
    `read_parquet(year_globs(...))`，不要用这里的方法，原因：
      1. 它自己 `duckdb.connect()`，没有 `SET enable_progress_bar=false`，
         扫 parquet 时会把进度条刷进 stderr（`connect_duckdb()` 就是为此存在）
      2. `create_views()` 用 `read_parquet('{root}/**/*.parquet')` 全目录递归，
         对 `frozen/stocks`、`frozen/industry` 这类**同目录多 schema** 的数据集
         会直接抛 schema mismatch；而它用 `except Exception: continue` 吞掉，
         结果是"视图建了但没建全"且不报错
      3. `query_files()` 拼 SQL 字符串，`where` 参数未做转义

"""
import duckdb
import pandas as pd
from .config import DB_DUCKDB


class QueryEngine:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_DUCKDB)

    def query(self, sql: str, params=None):
        """对数据库执行 SQL。

        支持 glob 语法直接查 Parquet：
            SELECT * FROM 'db/cleaned/daily_basic/year=2024/*.parquet'
        路径请用 database.config.parquet_glob() 生成，不要手写字符串。
        """
        con = duckdb.connect(self.db_path)
        try:
            if params:
                return con.execute(sql, params).fetchdf()
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    def query_files(self, glob_pattern: str, where: str = "", limit: int = 1000):
        """快捷查询：指定 parquet glob 模式 + where 条件"""
        sql = f"SELECT * FROM '{glob_pattern}'"
        if where:
            sql += f" WHERE {where}"
        if limit:
            sql += f" LIMIT {limit}"
        return self.query(sql)

    def create_views(self):
        """创建数据库视图，把各分区表注册为逻辑表"""
        import pandas as pd
        from .storage import Storage

        con = duckdb.connect(self.db_path)
        try:
            datasets = ["daily", "valuation", "adjust", "dividend",
                        "index_daily", "industry", "financial", "calendar"]
            for ds in datasets:
                store = Storage(ds)
                try:
                    con.execute(f"CREATE VIEW IF NOT EXISTS {ds} AS "
                                f"SELECT * FROM read_parquet('{store.root}/**/*.parquet')")
                except Exception:
                    continue
            con.close()
            return True
        except Exception:
            con.close()
            return False

    def tables(self):
        con = duckdb.connect(self.db_path)
        try:
            return con.execute("SHOW TABLES").fetchdf()
        finally:
            con.close()


def q(sql: str, params=None) -> pd.DataFrame:
    """快速查询入口"""
    return QueryEngine().query(sql, params)
