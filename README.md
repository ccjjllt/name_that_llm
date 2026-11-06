# Name That LLM

当前研究见 [Q8 研究报告](research/q8.md)。

- `src/ntllm/`：算法与训练代码。
- [configs/q8.toml](configs/q8.toml)：Q8 参数；`auxiliary` 为辅助权重对照，`source` 为来源替换。
- `data/`：原始来源、当前训练数据与 tokenizer。
- `runs/`：当前研究的计划、模型、预测和结果。

## 准备实验

辅助权重对照：238,353 条扩展数据，完整两轮，每臂 26,484 步。

```bash
uv run --offline python -m ntllm.train_lora \
  --data-dir data/q8_protocol --phase external --variant expanded \
  --centered-balance class_mass --max-updates 26484 --plan-only \
  --backend-initial-state runs/scale_preparation_20261107/common_auxiliary_initialization
```

另一臂改为 `--centered-balance global_mass`，共用初始化路径。官方阶段按配置保持相同配方。

来源替换：119,616 条／臂，完整两轮，每臂 13,292 步。

```bash
uv run --offline python -m ntllm.train_lora --experiment source \
  --data-dir data/q8_source_full/historical_anchor_vs_protocol_unknown \
  --phase external --variant baseline --max-updates 13292 --plan-only \
  --initial-state runs/scale_preparation_20261107/common_source_initialization
```

另一臂使用 `--variant mix20`。[完整计划与规模](runs/scale_preparation_20261107/summary.json)已包含两种实验及各自官方五折；token 计量使用现有本地 tokenizer。`--plan-only` 不加载模型权重。

正式训练使用已有基座与 `--prepared-plan`；官方适配另用 `--init` 指向本臂完成的外部模型，恢复使用 `--resume`。冻结计划和恢复读取保存参数，不再传 `--experiment`。完整参数见 `uv run python -m ntllm.train_lora --help`。
