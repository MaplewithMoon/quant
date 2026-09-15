"""Tushare 客户端封装：token加载 + 限流 + 重试"""
import time
import random
import pandas as pd
from ..token import load_token


class TushareCallError(RuntimeError):
    """tushare 接口在重试耗尽后仍失败

    专门用一个异常类型，便于调用方区分"接口失败（应重试/跳过，不写断点）"
    与"接口正常返回空（确实没有数据，可以标记完成）"。
    """


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
            # ⚠️ 不要用 ts.set_token(token)：它会把 token 写进 C:\Users\<user>\tk.csv，
            # 在没有家目录写权限的环境（受限沙箱 / 服务账号）会直接 PermissionError，
            # 整个下载流程起不来。pro_api 支持直接传 token，语义完全一样。
            self._pro = ts.pro_api(token)
        return self._pro

    def _wait(self):
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.calls_per_min:
            time.sleep(60 - (now - self._timestamps[0]) + 0.5)
            self._timestamps = [t for t in self._timestamps if now - t < 60]
        self._timestamps.append(now)

    def call(self, api_name: str, max_retries: int = 3, **params) -> pd.DataFrame:
        """调用 tushare 接口（限流 + 重试）

        修正要点
        --------
        旧实现里限频分支是 `time.sleep(5); continue`：若重试次数耗尽，
        `for` 循环自然结束 → **隐式返回 None**。调用方普遍写成
            if df is None or df.empty: self.storage.mark_done(code)
        于是"网络抖动导致重试耗尽"被当成"这只股票没有数据"，
        **被永久标记为已完成，再也不会重下**。

        现在：重试真正耗尽时**抛异常**；调用方的 except 会跳过该股票且不写断点，
        下次运行还会再试。只有 API 正常返回空 DataFrame 才算"确实没有数据"。

        另外：每次尝试都重新限流，旧实现只在进循环前 wait() 一次，
        重试是绕过限流的。
        """
        pro = self.get_pro()
        last_err = None
        for attempt in range(max_retries + 1):
            self._wait()                       # 每次尝试都限流
            try:
                fn = getattr(pro, api_name)
                df = fn(**params)
                return df if df is not None else pd.DataFrame()
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    break
                msg = str(e)
                if "最多" in msg or "每分钟" in msg or "频率" in msg:
                    time.sleep(5 + random.uniform(0, 1))
                else:
                    time.sleep(2 + random.uniform(0, 1))
        raise TushareCallError(
            f"tushare {api_name} 重试 {max_retries} 次仍失败: {last_err}"
        ) from last_err

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
