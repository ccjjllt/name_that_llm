# Name That LLM

当前研究与已有结果见 [Q8 研究报告](research/q8.md)。

- `configs/q8.toml`：共用参数与实验差异；`configs/backends/`：运行依赖。
- `src/ntllm/`：数据、训练、评价及当前研究方法。
- `data/`：当前原始来源、训练表、证据记录和 tokenizer。
- `runs/`：当前研究结果、冻结计划和训练产物，不提交 Git。
- [notebooks/q8_train.ipynb](notebooks/q8_train.ipynb)：Kaggle 分段训练、续跑与官方适配。

## Kaggle 训练

本地生成上传包，不加载模型或使用 GPU：

```bash
uv run --offline python -m ntllm.q8_kaggle_bundle
```

上传 `runs/kaggle_preparation/ntllm-kaggle.zip` 为私有 Kaggle Dataset，并与
`qwen-lm/qwen-3/Transformers/8b-base/1` 一起添加到 notebook 的输入。
模型使用已有 Kaggle 输入；代码不下载权重。环境设置和具体续跑步骤写在 notebook 中。

Notebook 每次只执行选定 job 的一段训练，默认 3 小时；每 100 步或 900 秒保存。
下一段附加上次成功保存的 notebook 输出，保留相同 job 和共同初始化。
官方适配必须使用对应外部阶段的完整终点。分段不改变总预算或学习率日程。

当前清单共 **40 个阶段**：centered 阶段参照 3 个外部＋25 个官方，来源替换 2 个外部＋10 个官方。
合计 113,236 次更新、约 2.97 亿有效 token；评价与重试另计。
原 class/global 对照已包含在阶段参照中，不重复训练。

## 配置

`q8.toml` 的 profiles：`auxiliary`、`source`、`off_off`、`off_on`、`on_off`、
`on_on`、`global_on`、`fusion_robustness`。其中融合稳健性是独立研究候选，未加入上述 40 个阶段。
`off/on` 分别表示外部和官方阶段是否启用 centered；共享外部阶段只训练一次。

冻结计划与恢复读取保存参数，不重新选择 profile。常规评价保留四个独立分支、
融合及稀疏布局；数据证据和来源替换范围随训练产物保存。详细参数见：

```bash
uv run --offline python -m ntllm.train_lora --help
```
