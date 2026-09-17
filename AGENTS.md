# 协作约定（AGENTS.md）

给在这个仓库里工作的自动化助手（以及新加入的人）看的操作约定。
**这些不是建议，是踩过坑之后定下来的规矩。**

---

## 1. 不要擅自跑全局数据校验

`python scripts/validate_data.py --check all` 是**几十分钟**量级，单项也不便宜。
2026-09 实测（Windows，本机）：

| 检查项 | 实测耗时 | 说明 |
|---|---|---|
| `parquet` | 1~25 秒 | 只查每个数据集最新一年并跳过占位年份；`--parquet-all` 要 5 分钟以上 |
| `calendar` | ~18 秒 | |
| `duplicates` | ~40 秒 | 批量数据集全扫 + 按代码数据集抽样 |
| `freshness` | ~107 秒 | 每个数据集都要取最大日期 |
| `limit` | > 180 秒 | 抽样 300 只算涨跌停 |
| `checkpoint` | ~294 秒 | 逐 key 比对磁盘 |
| `schema` / `coverage` / `completeness` | **最慢** | 要扫 7 万+ 文件 / 5,889 只股票 |

**规矩：跑之前先问。** 说清楚要查什么、为什么需要全量、预计多久，
由人来决定值不值得。日常开发用**针对性**的单项目检查就够了，例如
`--check parquet`、`--check duplicates`、`--check freshness`；只在数据层有
实质改动、或准备发布结论时才考虑全量。

---

## 2. 环境准备（每个 shell 都是新进程）

PATH 必须手动刷新，否则找不到 Python：

```powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
```

- Python 3.14；**没有 scipy**（需要 Spearman 就复用
  `factors/evaluation.py::_corr` 的"先排名再 Pearson"写法，别引入新依赖）。
- `TUSHARE_TOKEN` 只存在**机器级**环境变量里，需要时注入：
  `$env:TUSHARE_TOKEN = [Environment]::GetEnvironmentVariable('TUSHARE_TOKEN','Machine')`
- PowerShell 的 `*>` 写出来是 **UTF-16LE**，会把中文弄乱；要用
  `python -u ... *> file` 再以 `encoding='utf-16'` 读。
- 单条命令超过约 600 秒会被执行器杀掉，长任务用后台作业。

---

## 3. 数据层的规矩

- **写 parquet 一律用 `database.storage.atomic_to_parquet`**，不要直接
  `df.to_parquet(最终路径)`。非原子写被中断会留下**半截文件**（头魔数
  `PAR1` 还在、尾部是零字节），而读取方常见的 `except Exception: continue`
  会把它当成"这只标的没数据"跳过 —— 坏文件于是**永远不会被重写**，
  一直烂到某次全量重建才炸。`daily_update.py` 已经因此挂过一次（B19）。
  该函数写完会校验 footer，校验失败时**不替换最终文件**，旧数据还在。
- **发现分片损坏时，绝不能"当作它不存在"**。那会把这次增量窗口当成全部
  历史写进去，标的的多年数据被静默删除 —— 比留一个坏文件糟糕得多。
  正确做法：拒绝写入 + 记录 + 报错，然后用
  `python scripts/daily_update.py --repair-corrupt` 隔离并**重下整年**。
- **检查文件完整性要同时看头尾魔数**。只看头会漏掉截断文件，那正是坏文件
  的实际形态（`is_valid_parquet` 已经这么做了）。
- **frozen 是原始层**：只读。唯一的例外是**修复损坏数据**（去掉下载器
  造成的整行重复、隔离半截文件），因为那不是"改原始数据"，是**还原**成
  API 真实返回的样子。
- **派生层缓存按数据指纹失效**。改了 frozen 数据后，如果该数据集在
  `database.provenance.PANEL_DEPS` 里，面板缓存会自动作废；不在里面的
  （margin / northbound / futures / options / etf）不受影响。
- **PIT 是底线**：所有财务数据按 `ann_date` 对齐，不是 `end_date`；
  市场级数据（两融/北向/期货结算/期权持仓）都是收盘后发布，必须 `lag≥1`。
- 新增读取方后，`--check freshness` 的"读取方"一列应该能看到它。
  排在 `config.py` 里只是**登记**，`downloader/` 与 `download_*.py` 是
  **写入方**，都不算消费方。
- **不要用 `ts.set_token()`**：它会往 `C:\Users\<user>\tk.csv` 写文件，
  受限环境直接 PermissionError。用 `ts.pro_api(token)`，语义一样。
- `db/cleaned/` 的重建是**就地覆盖**，中途失败会留下**半新半旧**的层
  （按文件名排序，前面的股票是新数据、后面的是旧数据，而文件数完全正常）。
  所以重建前必须备份，失败必须回滚。

---

## 4. 测试与提交

- 测试文件放 `tests/test_*.py`，会被 `scripts/run_tests.py` **自动发现**
  （以前是硬编码清单，漏过文件，已改）。
- 测试**不能依赖 `db/` 或网络**（CI 里没有 `db/`）。读真实数据的测试要
  用存在性守卫，数据不在时干净跳过。
- 也别用 pytest 的 `tmp_path`：它在受限环境会 PermissionError；
  用 `tests/test_data_provenance.py::_tmpdir()` 那套项目内临时目录。
- **不要 push**。提交到本地即可，推送由人来做。
- commit message 用 UTF-8 文件 + `git commit -F`，别直接写在命令行里
  （中文会被编码搞乱）。
- `ruff check .` 是 CI 门禁，提交前跑一下。
- 仓库里有几个**用户自己的未跟踪文件**（`*Clone.py`、`新建 文本文档.txt`），
  不要动、也不要提交。

---

## 5. 结论要诚实

这个项目已经因为数据错误得出过错误的研究结论。写结论时：

- 说清楚**样本内外**、**样本量**、**是否擦线通过**。
- 不要用"显著"这种词掩盖 `|IC| = 0.051`、`t = -2.20` 这种擦线值。
- 数据有已知缺陷时（北向只有 300 行、opt_basic 只覆盖 SSE、
  `suspend_d` 不含暂停上市），**在结论旁边写出来**，不要只写在别处。
- 证伪的结论和证实的结论一样有价值，照实写。
