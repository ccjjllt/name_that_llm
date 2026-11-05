# 比赛事实与研究线索

以下资料截至 2026-11-04，综合官方任务说明、社区实验和本项目数据分析。社区分数按作者报告引用。

输入单段生成文本，预测来自六家厂商的 18 个模型标签。主指标为 balanced accuracy，即 18 类召回率的平均值。隐藏测试使用未见 prompt，各类别数量不等，public/private 类别比例也不同。评估应采用 prompt 分组留出，并避免用输出类别等量的假设强制重排预测。[官方评估](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)

完整推理在禁用网络的环境中进行，运行预算为 9 小时。模型规模、集成数量和来源条件似然评分应同时比较准确率与推理成本。[计算条件](https://www.kaggle.com/competitions/name-that-llm/overview/code-requirements)

## 数据实测

| 文件 | 形状 | 列 |
|---|---:|---|
| `data/train.csv` | 2,699 × 4 | `id`, `prompt_id`, `text`, `model` |
| `data/test.csv` | 3 × 2 | `id`, `text` |

- 共 150 个 prompt。149 个 prompt 含全部 18 类。`p002` 缺 `claude-fable-5`，该类 149 条，其余类别各 150 条。
- 所有字段无缺失或空白字符串，ID 无重复。训练文本原文及小写、空白归一化后均无重复。
- 按空白分词，训练文本平均约 114.66 词、中位数 101 词、范围 36–274 词。简介中的 40–250 词不是删样本的硬条件。
- 公开测试与训练没有完全相同的 ID 或文本。此结论不涉及隐藏测试。
- 官方说明隐藏重跑约 4,000 条，使用训练未出现过的 prompts。没有 `prompt_id` 输入。
- 公开测试只有 3 行，完整泛化能力主要通过分组交叉验证和隐藏测试成绩衡量。

详细数据统计见本地分析与官方数据说明。[本地统计](../data/data_audit.json) · [官方数据](https://www.kaggle.com/competitions/name-that-llm/data)

| 厂商 | 标签 |
|---|---|
| OpenAI | `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra` |
| Google | `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash` |
| Anthropic | `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5` |
| ZAI | `glm-5.3`, `glm-5.3-flash` |
| Qwen | `qwen3.8-max` |
| Moonshot AI | `kimi-k3` |

公开社区数据、公开模型与 LoRA 提供了外部训练来源。官方允许研究隐藏测试文本之间的关系，但同一 prompt 的回答数量没有固定保证，因此关系模型还需考察不完整回答组上的表现。[官方讨论](https://www.kaggle.com/competitions/name-that-llm/discussion/752084)

官方生成过程的精确模型版本、system prompt 与 sampling 参数仍有待澄清。外部生成数据可能带来版本、题材与文风偏移，这些差异需要通过分布分析和训练对照研究。[生成溯源讨论](https://www.kaggle.com/competitions/name-that-llm/discussion/750059)

## 值得复用的社区研究

下表中的 CV 和 LB 分别为作者报告的交叉验证与公开榜成绩。

| 线索 | 作者报告 | 对本项目的启发 |
|---|---|---|
| [分组 CV 与 CPU baseline](https://www.kaggle.com/competitions/name-that-llm/discussion/750390) | TF-IDF + LR：CV 0.343 / LB 0.353 | 按 prompt 分组。保留大小写和标点。随机行切分不能正确模拟新 prompt。 |
| [风格与长度分析](https://www.kaggle.com/competitions/name-that-llm/discussion/750664) | 风格融合后 balanced accuracy 0.372 | 细分同家模型、短文本是难点。标点规则适合做特征，不宜绝对排除标签。 |
| [冻结 Qwen 多层特征](https://www.kaggle.com/competitions/name-that-llm/discussion/750462) | 拼接多层 CV 0.414。融合后 LB 0.420 | 先探索无需微调的多层表征，浅层可能保留不同于末层的信号。 |
| [Open Pool v12](https://www.kaggle.com/competitions/name-that-llm/discussion/752766) | 约 211,369 条、59 个来源。新增数据尚未验证收益 | 数据更多不代表更贴合。`match_weight` 使用全量 train，直接用于 CV 可能偏乐观。 |
| [集成多样性](https://www.kaggle.com/competitions/name-that-llm/discussion/753177) | 单模型 OOF 0.569→0.605，替换集成后 LB 0.651→0.647 | 比较成员互补性，不只按单模型分数排序。 |
| [All Public + What Failed](https://www.kaggle.com/code/alexchilton/name-that-llm-0-667-all-public-what-failed) | 曾出现 nested CV 0.6865、LB 仅约 0.56 | 完整 18 回答的本地结构可能抬高关系建模收益，不能直接外推测试布局。 |

[Starter Notebook](https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition)提供词/字符 TF-IDF、逻辑回归和按 `prompt_id` 的 5 折 GroupKFold，是初始化基线的参考。部分公开 Notebook 使用强制类别均衡，与当前官方测试分布说明冲突，不作为默认策略。
