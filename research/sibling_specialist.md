# Sibling Odds 错例定位与定向 LoRA

以 Sibling Odds v1 的最后判决逻辑重放 Dennis 的三组已记录分数，复原此前回放的 798 条预测。C2 的 100 个错例中，32 个为 GPT 型号互判，47 个的真实标签和首选标签均在 Claude／Gemini 内，其中 18 个为 Gemini 判成 Claude。100 个错例有 79 个的真类位于硬标点规则之前的前三名。开发集也有相同的集中方向，52 个错例中 GPT 互判 22 个、Claude／Gemini 块内 19 个。[逐条诊断](../runs/sibling_specialist_20261123/local/row_diagnostics.parquet) · [混淆计数](../runs/sibling_specialist_20261123/local/error_diagnosis.json)

三组共有 58 项标点改判，全部原判为 Gemini 且包含非空格包围的 `—`。只恢复这些判决，C1 改对 6、改错 0，C2 改对 9、改错 0，四类压力集改对 17、改错 0。C2 的 BA 从 0.68203 升至 0.70492。官方训练集 600 条 Gemini 回答中只有 1 条含 `—`，这些公开评估面板存在显著的标点分布迁移。隐藏集的标点分布未知。三组也缺少带标签的真实 GLM 和 Fable 例子，实验保留其保护决策。[标点逐条改判](../runs/sibling_specialist_20261123/local/dash_changes.csv) · [仅恢复 Gemini 判决](../runs/sibling_specialist_20261123/local/evaluation.json)

在 Qwen3-1.7B Base 前 14 层的末 6 层训练 rank 8 LoRA，共用 19,125 条外部训练正文，来自 23 个来源、6,995 道题，其中 8,423 条不足 100 词。两组以相同采样、初始化及训练量各训练外部和官方数据一轮。第一组使用类别加权的 18 类交叉熵，第二组另外学习供应商归属和混淆块内类别。两组都在 1,140 条训练来源留出上比较官方适应前后的迁移表现。[配置](../configs/sibling_specialist.json) · [采样清单](../runs/sibling_specialist_20261123/local/sample.json) · [训练实现](../src/ntllm/sibling_specialist.py)

对同一 798 条的底座分数比较原 Sibling Odds、仅恢复 Gemini 标点改判、在恢复基础上加入训练读出。模型只能在同一混淆块内将底座前三名之一提为首选，候选与原判的底座分差最多 1.5，且训练读出自己须支持该交换。读取 C1 的 BA，分别从 0、0.15、0.30、0.50 为两组选择融合权重，再核对 C2 与压力集的改对、改错和短文表现。与仅恢复标点的逐条差值衡量训练新增收益。C2 在本轮已用于诊断，因此这些差值仍是研究面板上的同底座对照。[完成训练的 Notebook](https://www.kaggle.com/code/zhuating/ntl-sibling-error-specialist/versions/2)

仅恢复 Gemini 的 C2 增量为 **+2.29 个百分点**，按题重采样的描述性 95% 区间为 **+0.89 至 +3.86 个百分点**。随后仍有 41 个错例的真类满足新模型的候选和底座分差要求。用上一轮冻结 Qwen3-4B 的两组预测替换新模型读出，C1 均选择 0.15。相对仅恢复 Gemini，两组在 C2 分别净改对 1 条和净改错 1 条，在压力集均净改错 1 条。这个对照估计了仅更换融合方式对旧分类器的影响。[纯隐状态复算](../runs/sibling_specialist_20261123/previous_reader_control/states_only/evaluation.json) · [预测分布复算](../runs/sibling_specialist_20261123/previous_reader_control/states_forecasts/evaluation.json)

## 定向训练实测

两组各完成 797 次外部更新与 113 次官方适应更新，下载的训练数据清单与本地逐条一致。798 条零权重预测与 Sibling Odds 末段回放逐条一致。C1 在两组都选择 **0 权重**，所以按预设方案，C2 的训练增量均为 **改对 0、改错 0、净增 0**。下表显示强制启用时相对仅恢复标点的差值。[复算指标](../runs/sibling_specialist_20261123/audited_results.json) · [逐条改判](../runs/sibling_specialist_20261123/audited_changes.parquet)

| 损失 | 权重 | C2 改对 | C2 改错 | C2 净改对 | C2 BA 增量（百分点） | C1 净改对 | 压力集净改对 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 定向 | 0.15 | 2 | 1 | +1 | +0.76 | −1 | −1 |
| 定向 | 0.30 | 4 | 3 | +1 | +0.69 | −3 | −2 |
| 定向 | 0.50 | 7 | 4 | +3 | +1.26 | −4 | −1 |
| 交叉熵 | 0.15 | 2 | 1 | +1 | +0.76 | −1 | −1 |
| 交叉熵 | 0.30 | 2 | 2 | 0 | +0.36 | −3 | −2 |
| 交叉熵 | 0.50 | 6 | 4 | +2 | +1.01 | −4 | +2 |

C2 上事后最好的定向权重 0.50，训练增量按题重采样的描述性 95% 区间为 **−1.07 至 +4.24 个百分点**。官方适应后，两组单独在两个外部留源的 BA 分别为 0.3441 / 0.2583（定向）和 0.3471 / 0.2417（交叉熵），方向互换。定向训练与简单交叉熵在现有面板上的差异不足以支持正式提交。可复核的正收益集中在 Gemini 标点规则修复，线上效应仍待验证。
