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
- **说「某数据拿不到」之前，先把相关接口都列一遍。** B7 曾被判成"tushare 无
  历史行业归属接口"，实际只看了一个（`stock_basic.industry`，当前快照），
  而 `index_member_all` 一直就有 `in_date`/`out_date`。类似地
  `index_member`（带进出日期）无权限，但申万成分分级是另一回事。
- **tushare 的单次调用上限会伪装成"数据就这么少"。** 已经踩过两次：
  `opt_daily` 一把取恒定 15,000 行、`index_member_all` 默认恒定 3,000 行
  （且只返回 `is_new='Y'`，历史一条没有）。**看到"刚好整千整万"就怀疑截断**，
  用 `limit`/`offset` 翻页验证。
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

## 5. 回测结果可靠性（T0 护栏）

**排序原则：先保证"对任意策略，回测结果都可靠"，策略编写放最后。**
判据是「这件事是否影响**任意**策略」—— 是则优先，只影响某个策略的归策略阶段。

### 三道护栏必须成对存在于两条路径

项目有 `backtest/engine.py`（单标的）与 `backtest/multi_engine.py`（组合）
两条路径。**所有真实策略都走组合**，所以护栏必须装在组合上：

| 护栏 | 单标的 | 组合 |
|---|---|---|
| 冲击成本模型 | ✅ | ✅ 已接（默认 `fixed` + 1bp，别再改成 0） |
| 风控 | `risk.RiskManager`（信号级） | `risk.PortfolioRiskManager`（权重级） |
| 前视自检 | 有 | `run(audit_lookahead=True)` 默认开 |

- **新增回测脚本时必须显式传 `--impact-model` 与滑点**。零滑点会让结果偏乐观，
  而且掩盖容量问题（`--capital` 调大也没反应）。
- **引擎只能保证"成交不早于决策"**。策略实际用了哪些数据引擎不知道 ——
  要自己用 `backtest.lookahead.verify_point_in_time` 逐决策校验，
  `scripts/semiconductor_rotation.py` 是正确范例（记录每次决策用的数据切片）。
  别把全量面板传给 `check_lookahead`，它会报成千上万条假违规（已加拦截）。

### 组合层风控的两条硬约束

1. **只能降风险**：`PortfolioRiskManager` 最后统一 clip，规则写错方向也放不大仓位。
   新规则请遵守这个约定，不要绕过。
2. **触发要留痕**：写进 `risk_events`，否则事后看不出"这次减仓是策略决定的
   还是风控砍的"。

⚠️ `PortfolioRiskManager` 有 `_prev_weights` 状态。跨回测段复用一个实例会让状态
串味（测试集第一次调仓会拿训练集末尾的持仓当上期持仓）——**传配置、每段现建**。

### 已知数据缺陷

`database/defects.py` 是**唯一登记处**。文档讲为什么，它讲"是什么、影响哪段"，
每条带 `doc_ref` 回指文档（避免两处漂移）。

- 遇到"数据本身就不对、但代码没错"的区间，**登记进去**，别只写在文档里 ——
  跑回测的人看不到文档。
- `needs_data=True` 表示**代码层面改不动、必须补数据源**。这类要单独交给
  人决定，不要默默绕过。
- 回测脚本应当在报告里输出 `defects_in_window()` 的附注。
- **"我们不知道" ≠ "没有限制"**。C1 的处理方式是范例：只把涨跌停置 NaN 会变成
  "随便交易"（偏宽松），所以还要**把这些 (代码, 日期) 排除出股票池**。

### 时点正确性：任何"用今天决定历史"的做法都是前视

这是这个项目**反复**踩的一类坑，而且每次都伪装成"保守"或"合理"：

| 错误做法 | 看起来像 | 实际是 |
|---|---|---|
| 用当前 `stock_basic.industry` 做历史行业中性化 | "行业归属本来就是慢变量" | 前视（1,646 只股票换过行业） |
| 用**代码前缀**排除北交所/新三板 | "保守，宁可少交易" | 把前视从涨跌停挪到了**池子成员资格** |
| 用 `name` 含"退"判断退市 | "大概能猜出来" | 既漏（改名退市的不带"退"）又错（退市整理期带"退"但可交易） |
| 用 `close.notna()` 当可交易判据 | "没行情自然不能交易" | 表达不了交易所归属，且掩盖幸存者偏差 |

**正确的地基只有一种：时点正确的证券主表（`database/master.py`）**，
三层过滤全是日期比较：

```python
master[(master['exchange'].isin(['SSE','SZSE']))          # ① 交易所归属
       & (master['list_date'] <= t)                        # ② 已上市
       & (master['delist_date'].isna() | (master['delist_date'] > t))]  # ③ 未退市
```

- 主表必须同时拉 `list_status` 的 **L / D / P** 三种状态，否则有**幸存者偏差**
  （"当时存在、后来退市"的股票不在池子里 -> 历史收益被系统性高估）。
- 退市边界**严格**：`delist_date > t`，退市当日即不可交易。
- 主表查不到的代码一律视为不可交易 ——「不知道」不等于「可以交易」。
- 主表缺 `exchange`/`delist_date` 时**不能静默放行**（见 `master_capabilities`）。

---

## 6. 结论要诚实

这个项目已经因为数据错误得出过错误的研究结论。写结论时：

- 说清楚**样本内外**、**样本量**、**是否擦线通过**。
- 不要用"显著"这种词掩盖 `|IC| = 0.051`、`t = -2.20` 这种擦线值。
- 数据有已知缺陷时（北向只有 300 行、opt_basic 只覆盖 SSE、
  `suspend_d` 不含暂停上市），**在结论旁边写出来**，不要只写在别处。
- 证伪的结论和证实的结论一样有价值，照实写。
