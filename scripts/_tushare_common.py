# -*- coding: utf-8 -*-
"""tushare 下载共用工具：限流、重试、代码转换（供各下载脚本复用）"""
import time
import shutil

DISK_MIN_FREE = 5 * 1024 ** 3
CALLS_PER_MIN = 190


class RateLimiter:
    """限流器：控制调用间隔，保证每分钟调用数低于限频"""
    def __init__(self, calls_per_min=CALLS_PER_MIN):
        self.min_interval = 60.0 / calls_per_min
        self._last = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.time()


def api_call(pro, api_name, limiter, max_retries=5, **params):
    """带限流 + 频率超限退避重试的 tushare 调用
    - 限流器保证基础间隔
    - 频率超限/瞬时"不存在"错误时指数退避重试，不跳过数据
    """
    for attempt in range(max_retries):
        limiter.wait()
        try:
            return getattr(pro, api_name)(**params)
        except Exception as e:
            msg = str(e)
            if any(k in msg for k in ("频率", "超限", "积分", "最多", "不存在")):
                wait_s = min(30, 3 * (2 ** attempt))  # 3,6,12,24,30,30
                if "不存在" in msg:
                    wait_s = min(wait_s, 5)
                print(f"  [{api_name}] {msg[:30]}... 等待{wait_s}s重试(第{attempt+1}次)", flush=True)
                time.sleep(wait_s)
                continue
            raise  # 其他错误直接抛出
    raise RuntimeError(f"{api_name} 持续失败，放弃本条（下次运行会重试）")


def ts_code_of(code: str) -> str:
    """6位代码 → tushare ts_code（北交所920 → BJ，0/3开头 → SZ，其余 → SH）"""
    if code.startswith("920"):
        return f"{code}.BJ"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    return f"{code}.SH"


def check_disk() -> None:
    """磁盘保护：D盘剩余不足则抛异常停止"""
    free = shutil.disk_usage("D:\\").free
    if free < DISK_MIN_FREE:
        raise RuntimeError(f"D盘剩余 {free/2**30:.1f}GB < {DISK_MIN_FREE/2**30:.0f}GB，停止")
