# -*- coding: utf-8 -*-
"""运行全部单元测试（也可用 pytest）"""
import sys
import io
import os
sys.path.insert(0, ".")
sys.path.insert(0, "tests")
if __name__ == "__main__":
    # 只在直接运行时切编码：模块顶层替换 sys.stdout 是有副作用的 import，
    # 会破坏 pytest 的输出捕获（详见 utils/console.py）
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

import subprocess
import sys

TESTS = [
    "tests/test_loader.py",
    "tests/test_preprocessor.py",
    "tests/test_engine.py",
    "tests/test_ledger.py",
    "tests/test_market_rules.py",
    "tests/test_impact.py",
    "tests/test_optimizer.py",
    "tests/test_storage.py",
    "tests/test_factors.py",
    "tests/test_universe.py",
    "tests/test_portfolio_backtest.py",
    "tests/test_attribution.py",
    "tests/test_performance.py",
    "tests/test_sector_rotation.py",
    "tests/test_fundamental.py",
    "tests/test_joinquant.py",
    "tests/test_limit_rules.py",
    "tests/test_data_provenance.py",
]


def main():
    print("=" * 60)
    print("运行全部单元测试")
    print("=" * 60)
    failures = 0
    for t in TESTS:
        print(f"\n>>> {t}")
        r = subprocess.run([sys.executable, t], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr[-800:])
            failures += 1
    print("=" * 60)
    if failures:
        print(f"{failures} 个测试文件失败")
        sys.exit(1)
    print("全部测试通过")


if __name__ == "__main__":
    main()
