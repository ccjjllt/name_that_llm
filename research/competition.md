# 比赛事实与研究线索

核查日期：2026-11-04。来源为 Kaggle MCP 返回的官方页面、公开讨论、Notebook 源码，以及本地下载文件的统计。网页信息是当日快照；下文社区成绩均未由本项目复现。

## 任务与提交

- 输入单段生成文本，预测 18 个模型标签之一，涉及 6 家厂商。
- 主指标为 **balanced accuracy**，即 18 类召回率的平均值。隐藏测试各类别数量不相等，public/private 的类别比例也不同。
- Code Competition：Notebook 禁用网络，CPU/GPU 运行上限均为 9 小时，输出 `submission.parquet`，字段为 `id`、`model`。每个测试 ID 恰好出现一次，标签必须来自官方 18 类。
- 每天最多 5 次提交、最多选择 2 个最终提交、团队最多 5 人。总奖金 $50,000，前十名各 $5,000。
- 报名截止元数据对应北京时间 **2027-01-07 07:59**。
- **最终截止尚有冲突**：MCP 元数据对应北京时间 **2027-01-15 07:59**，Timeline 正文对应 **2027-01-14 07:59**。当前未找到官方澄清，安排实验时先按较早日期。

来源：[比赛主页](https://www.kaggle.com/competitions/name-that-llm)、[Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)、[Code Requirements](https://www.kaggle.com/competitions/name-that-llm/overview/code-requirements)、[Timeline](https://www.kaggle.com/competitions/name-that-llm/overview/timeline)、[Rules](https://www.kaggle.com/competitions/name-that-llm/rules)。

## 数据实测

| 文件 | 形状 | 列 |
|---|---:|---|
| `data/train.csv` | 2,699 × 4 | `id`, `prompt_id`, `text`, `model` |
| `data/test.csv` | 3 × 2 | `id`, `text` |
| `data/sample_submission.parquet` | 3 × 2 | `id`, `model` |

- 共 150 个 prompt。149 个 prompt 含全部 18 类；`p002` 缺 `claude-fable-5`，该类 149 条，其余类别各 150 条。
- 所有字段无缺失或空白字符串，ID 无重复；训练文本原文及小写、空白归一化后均无重复。
- 按空白分词，训练文本平均约 114.66 词、中位数 101 词、范围 36–274 词。简介中的 40–250 词不是删样本的硬条件。
- 公开测试与训练没有完全相同的 ID 或文本。此结论不涉及隐藏测试。
- 官方说明隐藏重跑约 4,000 条，使用训练未出现过的 prompts；没有 `prompt_id` 输入。
- 数据说明写公开 test 为 3 行，Evaluation 写 1 行；**实际下载文件为 3 行**。sample submission 的 `gpt-5.5` 只是格式占位。

完整统计、来源、SHA-256 见 [data_audit.json](../data/data_audit.json)。官方来源：[Data](https://www.kaggle.com/competitions/name-that-llm/data)。

| 厂商 | 标签 |
|---|---|
| OpenAI | `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra` |
| Google | `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash` |
| Anthropic | `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5` |
| ZAI | `glm-5.3`, `glm-5.3-flash` |
| Qwen | `qwen3.8-max` |
| Moonshot AI | `kimi-k3` |

## 官方讨论澄清

[规则集中答复](https://www.kaggle.com/competitions/name-that-llm/discussion/752084)中，STAFF/ADMIN 明确：

- 现有公共社区数据集、参赛者都可访问的公开模型权重和 LoRA 可以使用。
- 合理成本内自生成的数据无需强制公开，也没有公开截止日期。超过可能奖金 $5,000 的 token 花费被举为不合理情形，不能反推低于该金额必然合规。
- Benchmarks 额度用于 benchmarking，不应用来生成比赛训练数据；此前生成的公共数据可用，此前使用不追罚。
- 允许探索 hidden-test 文本之间的关系，但未承诺每个 prompt 拥有完整的 18 模型回答。

[生成溯源讨论](https://www.kaggle.com/competitions/name-that-llm/discussion/750059)询问了精确版本、system prompt、sampling 参数等，截至读取未见官方答复。外部生成数据的版本与分布一致性需要作为实验问题处理。

## 值得复用的社区研究

以下是线索，不是本项目成绩；CV 为作者本地交叉验证，LB 为作者报告的公开榜分数。

| 线索 | 作者报告 | 对本项目的启发 |
|---|---|---|
| [分组 CV 与 CPU baseline](https://www.kaggle.com/competitions/name-that-llm/discussion/750390) | TF-IDF + LR：CV 0.343 / LB 0.353 | 按 prompt 分组；保留大小写和标点。随机行切分不能正确模拟新 prompt。 |
| [风格与长度分析](https://www.kaggle.com/competitions/name-that-llm/discussion/750664) | 风格融合后 balanced accuracy 0.372 | 细分同家模型、短文本是难点；标点规则适合做特征，不宜绝对排除标签。 |
| [冻结 Qwen 多层特征](https://www.kaggle.com/competitions/name-that-llm/discussion/750462) | 拼接多层 CV 0.414；融合后 LB 0.420 | 先探索无需微调的多层表征，浅层可能保留不同于末层的信号。 |
| [Open Pool v12](https://www.kaggle.com/competitions/name-that-llm/discussion/752766) | 约 211,369 条、59 个来源；新增数据尚未验证收益 | 数据更多不代表更贴合；`match_weight` 使用全量 train，直接用于 CV 可能偏乐观。 |
| [集成多样性](https://www.kaggle.com/competitions/name-that-llm/discussion/753177) | 单模型 OOF 0.569→0.605，替换集成后 LB 0.651→0.647 | 比较成员互补性，不只按单模型分数排序。 |
| [All Public + What Failed](https://www.kaggle.com/code/alexchilton/name-that-llm-0-667-all-public-what-failed) | 曾出现 nested CV 0.6865、LB 仅约 0.56 | 完整 18 回答的本地结构可能抬高关系建模收益，不能直接外推测试布局。 |

[Starter Notebook](https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition)提供词/字符 TF-IDF、逻辑回归和按 `prompt_id` 的 5 折 GroupKFold，是初始化基线的参考。部分公开 Notebook 使用强制类别均衡，与当前官方测试分布说明冲突，不作为默认策略。
