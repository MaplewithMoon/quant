# -*- coding: utf-8 -*-
"""Tushare token 加载工具
读取优先级: 环境变量 TUSHARE_TOKEN > 本地文件 tushare_token.txt
"""


def load_token() -> str:
    import os
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token

    # 本地 token 文件（不上传 git）
    candidates = [
        os.path.join(os.path.dirname(__file__), "tushare_token.txt"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "tushare_token.txt"),
    ]
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                t = f.read().strip()
                if t:
                    return t
        except OSError:
            continue
    return ""


def ensure_token() -> str:
    token = load_token()
    if not token:
        raise RuntimeError(
            "未找到 Tushare token。请将 token 保存到 database/tushare_token.txt"
            " 或设置环境变量 TUSHARE_TOKEN"
        )
    return token
