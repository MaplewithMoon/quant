"""DuckDB 查询层：直接对 Parquet 分区做 SQL 查询"""
import duckdb
import pandas as pd
from .config import DB_DUCKDB


class QueryEngine:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_DUCKDB)

    def query(self, sql: str, params=None):
        """对数据库执行 SQL。
        支持 glob 语法直接查 Parquet：SELECT * FROM 'db/daily/year=2024/*.parquet'
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
