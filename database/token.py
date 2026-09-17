# -*- coding: utf-8 -*-
"""凭据加载工具

读取优先级（两者都是）：**环境变量 > 本地未跟踪文件**

    tushare : TUSHARE_TOKEN                    > tushare_token.txt
    jqdata  : JQDATA_USER / JQDATA_PASSWORD    > jqdata_account.txt

⚠️ 聚宽存的是**登录密码**，不是 token。明文落盘有风险：
   - 优先用环境变量；
   - 用完建议改密码，或用一次性密码；
   - `jqdata_account.txt` 已加入 .gitignore，**不要提交**。
"""
import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _read_kv_fallback(filenames, keys):
    """按「环境变量 > 本地文件」取值；返回与 keys 等长的列表

    文件支持两种写法：`user=xxx` / `password=yyy`，或一行一个值。
    """
    out = [_env(k) for k in keys]
    if all(out):
        return out
    for path in ([os.path.join(_HERE, f) for f in filenames]
                 + [os.path.join(_ROOT, f) for f in filenames]):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = [ln.strip() for ln in fh
                       if ln.strip() and not ln.strip().startswith("#")]
        except OSError:
            continue
        kv, bare = {}, []
        for ln in raw:
            if "=" in ln:
                k, _, v = ln.partition("=")
                kv[k.strip().lower()] = v.strip()
            else:
                bare.append(ln)
        for i, (name, fname) in enumerate(zip(keys, filenames)):
            if out[i]:
                continue
            short = name.split("_")[-1].lower()
            val = kv.get(short, "") or kv.get(fname.split(".")[0].lower(), "")
            if not val and i < len(bare):
                val = bare[i]
            out[i] = val
        if all(out):
            break
    return out


def load_token() -> str:
    """Tushare token"""
    vals = _read_kv_fallback(["tushare_token.txt"], ["TUSHARE_TOKEN"])
    return vals[0]


def load_jqdata_account():
    """聚宽账号 -> (user, password)；缺失返回 ('', '')

    ⚠️ 这是**登录密码**。建议设机器级环境变量而不是写文件：
        [Environment]::SetEnvironmentVariable('JQDATA_USER','手机号','Machine')
        [Environment]::SetEnvironmentVariable('JQDATA_PASSWORD','密码','Machine')
    """
    u, p = _read_kv_fallback(["jqdata_account.txt"],
                             ["JQDATA_USER", "JQDATA_PASSWORD"])
    return u, p


def ensure_token() -> str:
    token = load_token()
    if not token:
        raise RuntimeError(
            "未找到 Tushare token。请将 token 保存到 database/tushare_token.txt"
            " 或设置环境变量 TUSHARE_TOKEN"
        )
    return token
