# Name That LLM

Kaggle 文本来源识别研究。给定回答正文，从 18 个候选模型中识别作者。

- [研究概况](research/overview.md) 汇总正式成绩、同底座证据和暂停的实验。
- [结题与停止原因](research/retired.md) 汇总已结束的实验和实测依据。
- `src/ntllm/` 存放数据、评价、训练与提交候选代码；`configs/` 存放本地训练配置。
- `data/` 存放官方数据、外部来源和准备后的数据；`runs/` 存放本地实验与提交证据。

通用 Q8 训练入口为 `uv run --offline python -m ntllm.train_lora --help`。新计划默认使用 `data/q8_protocol`，恢复运行读取冻结计划中的数据路径。
