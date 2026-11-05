# Name That LLM

用于 [Kaggle Name That LLM](https://www.kaggle.com/competitions/name-that-llm) 的研究工作区。重点是快速验证假设、积累可靠的实验结果。保持简单的代码和配置，不设置 CI、审批门控或工程测试套件。

## 开始实验

在项目根目录执行，`uv` 会使用 Python 3.12 和本项目的 `.venv`：

```bash
uv sync
uv run python -m ntllm.baseline --config configs/tfidf.toml
```

默认运行 5 折按 `prompt_id` 分组的交叉验证，计算 balanced accuracy。每折独立拟合 TF-IDF，用词级与字符级 n-gram 训练逻辑回归。原始文本的大小写和标点默认保留。

需要同时训练全量模型并对本地示例测试集生成预测时：

```bash
uv run python -m ntllm.baseline --config configs/tfidf.toml --fit-full
```

公开 `test.csv` 只有 3 行，此处的 `submission.parquet` 仅用于检查本地预测流程。正式提交需要把代码与依赖准备到 Kaggle Notebook，在禁用网络的环境中对隐藏测试重新推理；本命令不会上传或提交。

可直接复制 TOML 配置开展新实验，或用 CLI 改变数据目录、输出目录、折数和种子：

```bash
uv run python -m ntllm.baseline --config configs/tfidf.toml --folds 3 --seed 123
uv run python -m ntllm.baseline --config configs/tfidf.toml --data-dir data --output-dir runs
```

## 目录与结果

```text
AGENTS.md             研究协作约定
configs/              实验配置
data/                 官方原始数据与数据核查记录（本地保留）
src/ntllm/            可复用的实验代码
notebooks/            临时分析、可视化与探索
research/             比赛事实、实验假设与研究结论
runs/                 每次实验独立目录（本地保留）
```

每次运行保存配置、整体/分折指标、各类召回率、长度分桶分析、OOF 预测概率和混淆矩阵。`--fit-full` 额外保存模型与示例预测。新实验使用独立目录，不覆盖旧结果。

数据已通过 Kaggle MCP 下载：训练集 2,699 条、150 个 prompt、18 个模型；`test.csv` 与 `sample_submission.parquet` 各 3 行。`data/data_audit.json` 保存实测统计、来源和文件校验值。原始数据和生成产物由 `.gitignore` 排除，`uv.lock` 用于记录依赖版本。

需要 Notebook 内核和绘图时再安装可选依赖：

```bash
uv sync --group explore
```

在编辑器中选择 `.venv/bin/python` 作为 Notebook 内核即可。

## 研究起点

- 首次实测：5 折 OOF balanced accuracy **0.31225**。详见[实验记录](research/experiments.md)，包含配置、长度分桶、弱类别和产物链接。
- [比赛与数据事实](research/competition.md)：官方信息、关键讨论和仍未澄清的事项。
- [实验假设](research/ideas.md)：从文本基线到冻结特征、外部数据与 LoRA 的研究顺序。
- [协作约定](AGENTS.md)：研究优先、轻量验证、可复用结果。

本地评估按 prompt 分组，用 OOF 的 balanced accuracy 比较方案，同时看类别混淆和短文本表现。隐藏测试类别不均衡、prompt 未见，不能默认每组都包含 18 个模型。交叉验证用于判断研究方向，不设置通过阈值。
