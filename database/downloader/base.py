"""下载器基类：重试、限流、断点、磁盘监控"""
import time
import random
import pandas as pd
from functools import wraps
from ..storage import Storage, DiskFullError
from ..config import DATA_START_YEAR
from utils.logger import setup_logger

# baostock 依赖 pandas<2.0 的 DataFrame.append，这里打兼容补丁
if not hasattr(pd.DataFrame, "append"):
    def _df_append(self, other, ignore_index=False):
        return pd.concat([self, other], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append


class RateLimiter:
    """简易滑动窗口限流器"""
    def __init__(self, calls_per_min: int = 100):
        self.calls_per_min = calls_per_min
        self._timestamps = []

    def wait(self):
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.calls_per_min:
            sleep_time = 60 - (now - self._timestamps[0]) + 0.5
            time.sleep(sleep_time)
            self._timestamps = [t for t in self._timestamps if now - t < 60]
        self._timestamps.append(now)


def retry_call(func, max_retries=3, base_delay=1.5, backoff=2.0):
    """带指数退避的重试"""
    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return func()
        except DiskFullError:
            raise
        except Exception as e:
            if attempt >= max_retries:
                raise ConnectionError(f"{func.__name__} 重试{max_retries}次仍失败: {e}")
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= backoff


class BaseDownloader:
    """所有下载器的基类"""

    def __init__(self, dataset: str, name: str = "", calls_per_min: int = 60):
        # 下载器是唯一被允许写入只读 frozen 层的角色
        self.storage = Storage(dataset, allow_frozen=True)
        self.limiter = RateLimiter(calls_per_min)
        self.logger = setup_logger(name or dataset)
        self.start_year = DATA_START_YEAR

    # ---------- 公共工具 ----------
    def fetch(self, func, *args, max_retries=3, **kwargs):
        """限流 + 重试的接口调用"""
        self.limiter.wait()
        self.storage.assert_disk_ok()
        return retry_call(lambda: func(*args, **kwargs), max_retries=max_retries)

    def save_year(self, df, year, code=None, force=False):
        self.storage.save(df, year=year, code=code, force=force)

    # ---------- 子类需实现 ----------
    def download(self, *args, **kwargs):
        raise NotImplementedError

    def resume(self):
        """断点续跑：默认调用 download()，子类可覆盖实现增量逻辑"""
        return self.download()
