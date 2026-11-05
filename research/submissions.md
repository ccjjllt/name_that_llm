# Kaggle 提交记录

## 2026-11-04：公开 0.668 方案复现

- 原方案：[Omer Faruk Merey — 0.668 | Verifier + Matched Rows + Pool Prompts](https://www.kaggle.com/code/omerfarukmerey/0-668-verifier-matched-rows-pool-prompts)，源 Notebook v2。
- 选择依据：当次 Kaggle MCP `search_notebooks(ScoreDescending)` 排名第一；`search_content` 返回 `kernel_document.best_public_score=0.668`。该分数属于作者，不是本账号复现成绩。
- 本账号私有副本：[ccjjllt/name-that-llm-public-0668-reproduction](https://www.kaggle.com/code/ccjjllt/name-that-llm-public-0668-reproduction)，kernel ID `142709450`，正式提交使用 v6。
- 保留原代码、16 个 dataset、8 个 notebook output、8 个 model 输入；Open Pool 改为下文记录的官方 v13 原样私有快照，对应 190,588 行。使用源 Docker 镜像，T4 ×2，禁用网络，最长运行 9 小时。
- 原方案包括多模型/LoRA 特征、外部数据与 n-gram 分类器、互补集成、ATLAS、soft Sinkhorn、文风残差和 pairwise verifier。本次不调整其建模配方。

2026-11-05 使用 `ccjjllt` 授权查询该比赛的提交列表，返回一条记录。提交 `57987571` 的状态为 **COMPLETE**，公开榜 **0.668**，与源方案的公开分数一致。

- 提交账号与队伍：`ccjjllt`。
- 提交时间：2026-11-04 22:17:40 北京时间（API：`2026-11-04T14:17:40.507Z`）。
- 提交文件：v6 生成的 `submission.parquet`，通过 Code Competition 提交接口发起隐藏测试重跑。
- Kaggle CLI 返回的本账号 `publicScore` 为 `0.668`，`privateScore` 为空。
- v6 普通 commit 已 COMPLETE，session ID `365647412`，耗时 435.085 秒。日志确认 `svc pool = open pool 190588` 与 `external rows 24443`，输出包含 `submission.parquet`；9 个代码 cell 与原方案逐字一致。

准备阶段记录：MCP 创建的 v1、v2 因所有输入未挂载而失败；v3、v4 仅 QuickSave 尝试修复参数，也未挂载成功。v5 通过 CLI 解决挂载并运行成功（414.834 秒），但服务器忽略输入名里的 `/13`，实际使用 211,369 行新池，因此没有提交 v5。上述版本均未进行比赛提交。

v6 的历史数据快照：

- 从[官方 Open Pool v13](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/13)下载 `ntl_open_pool.parquet`，保留原文件字节及逐行来源，不进行转换。
- 文件大小 96,388,050 bytes；总行数与原代码筛选后行数均为 190,588。
- SHA-256：`38e48b40e1db0ab1bf6e4f29d3697e37307c49d1ae5020a7af3cd59c484657d5`。
- 快照为[ccjjllt/name-that-llm-open-pool-v13-frozen](https://www.kaggle.com/datasets/ccjjllt/name-that-llm-open-pool-v13-frozen)，私有，dataset ID `13087503`，version 1；保留 `CC-BY-SA-4.0` 许可、原作者归属与来源说明。已读回确认 Ready、私有状态及文件字节数。
- v6 只替换此输入来源，原方案代码不变。不要更新此冻结数据集；需要其他数据实验时使用不同数据集。

原因与下一步：

- 已实证 Kaggle MCP `save_notebook` 会丢弃四种 data-source 数组，普通字段、`...Setter`、二者同时以及 snake_case 变体均无效。这与官方仓库的[已知问题 #1140](https://github.com/Kaggle/kaggle-cli/issues/1140)一致；官方 CLI/REST 是可用替代路径。
- 用户已明确选择 `ccjjllt`。本机 CLI 的默认账号 `qingyu1232` 保持不变；本次复用已有 Kaggle MCP 授权，仅传入独立 CLI 子进程，并通过 `kernels list --mine` 实际核实为 `ccjjllt`。没有显示或另存凭据。
- CLI 上传材料位于 `runs/public_0668_20261104/cli/`，指向当前 `ccjjllt` 私有副本和冻结的历史数据。官方 CLI 已返回 `Kernel version 6 successfully pushed`；v5 的旧配置另存为 `kernel-metadata-v5.json`。
- Notebook 文件自身的 `metadata.kaggle.dataSources` 与当前 API 元数据数量不一致，可能是上游旧记录，不能将这些旧 numeric IDs 当作已核实的当前输入版本。

公开测试只有 3 行，源方案会在普通 commit 中输出占位 `submission.parquet`；完整推理发生在正式提交后的隐藏测试重跑，不能把普通 commit 成功当成分数复现。

本地原始源码、搜索证据、保存请求与响应位于 `runs/public_0668_20261104/`，包含 `source.ipynb`、`source-metadata.json`、`search-score-descending.json`、`source-response.json`、`save-request.json`、`save-response.json`。这些文件留在本地，不提交 Git。
