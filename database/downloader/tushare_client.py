"""Tushare 客户端封装：token加载 + 限流 + 重试"""
import time
import random
import pandas as pd
from ..token import load_token


class TushareClient:
    def __init__(self, calls_per_min: int = 200):
        self.calls_per_min = calls_per_min
        self._timestamps = []
        self._pro = None

    def get_pro(self):
        if self._pro is None:
            token = load_token()
            if not token:
                raise RuntimeError("未找到 Tushare token，请设置环境变量或 database/tushare_token.txt")
            import tushare as ts
            ts.set_token(token)
            self._pro = ts.pro_api()
        return self._pro

    def _wait(self):
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.calls_per_min:
            time.sleep(60 - (now - self._timestamps[0]) + 0.5)
            self._timestamps = [t for t in self._timestamps if now - t < 60]
        self._timestamps.append(now)

    def call(self, api_name: str, max_retries: int = 3, **params) -> pd.DataFrame:
        """调用 tushare 接口（限流+重试）"""
        pro = self.get_pro()
        self._wait()
        for attempt in range(max_retries + 1):
            try:
                fn = getattr(pro, api_name)
                df = fn(**params)
                return df if df is not None else pd.DataFrame()
            except Exception as e:
                msg = str(e)
                if "最多" in msg or "每分钟" in msg or "频率" in msg:
                    time.sleep(5)
                    continue
                if attempt < max_retries:
                    time.sleep(2 + random.uniform(0, 1))
                else:
                    raise

    # 高频封装：常用接口
    def daily_basic(self, ts_code, start, end):
        return self.call("daily_basic", ts_code=ts_code,
                         start_date=start, end_date=end)

    def daily(self, ts_code, start, end):
        return self.call("daily", ts_code=ts_code,
                         start_date=start, end_date=end)

    def adj_factor(self, ts_code, start, end):
        return self.call("adj_factor", ts_code=ts_code,
                         start_date=start, end_date=end)

    def income(self, ts_code, start, end):
        return self.call("income", ts_code=ts_code, start_date=start, end_date=end)

    def balancesheet(self, ts_code, start, end):
        return self.call("balancesheet", ts_code=ts_code, start_date=start, end_date=end)

    def cashflow(self, ts_code, start, end):
        return self.call("cashflow", ts_code=ts_code, start_date=start, end_date=end)


_tc = None


def get_client(calls_per_min: int = 200) -> TushareClient:
    global _tc
    if _tc is None:
        _tc = TushareClient(calls_per_min)
    return _tc
