from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    initial_capital: float = 1_000_000
    commission: float = 0.0001      # 佣金费率：万分之一
    min_commission: float = 5.0     # 单笔最低佣金：5元
    slippage: float = 0.001

    data_backend: str = "akshare"
    data_cache_dir: str = "data/storage"

    log_level: str = "INFO"
    log_file: str = ""

    strategy_params: dict = field(default_factory=lambda: {
        "default": {"fast_ma": 5, "slow_ma": 20},
    })

    risk_max_drawdown: float = 0.20
    risk_max_position_pct: float = 0.25

    symbols: list = field(default_factory=lambda: ["000001"])


settings = Settings()
