# ATLAS 似然复核与 Dennis 0.719 的同底座对照

## 研究问题

在 Dennis Guarded Fable v1 的实际末段候选上，源条件 token 似然能否减少相近型号间的错误？优先衡量真实标签处于候选中的错例、改对与改错、短文和类别召回，并用同一 Notebook 主流程的正式提交核对公开榜增量。基准是公开评分版本 `372174726`，其第 2 版另有族内策略改动。[Dennis v1](https://www.kaggle.com/code/dennis6886/glm-user-prefix-guarded-fable-return?scriptVersionId=372174726)

## 逐条基准

复用完整集成、两路 GLM、Fable 阈值保护、族内残差和字形决策。在公开有标签的输入上核对每一阶段的全类分数、前三候选和最终标签。C1 预设为开发集，C2 为评估集。每题抽至多 6 条，得到 180 条开发正文与 300 条评估正文，各含 12 类；另外抽取 318 条经返回版本核验且基于重建训练题的四类正文作为来源和题型压力对照。标签及题号仅用于诊断。按正文归一化键排除可见训练集与所挂载 Open Pool 的重复。[C1](https://www.kaggle.com/datasets/beroli/ntl-benchmark-c1-generated-panels/versions/1) · [C2](https://www.kaggle.com/datasets/beroli/ntl-benchmark-c2-generated-panels/versions/1) · [四类压力来源](https://www.kaggle.com/datasets/markbarney/ntl-matched-passages-four-models/versions/2) · [基准回放](../src/ntllm/atlas_baseline.py)

正式竞赛阶段只输出正常提交的预测。净改对数来自上述带标签对照，竞赛公开榜用 balanced accuracy 衡量线上差值。每题问题构成与隐藏测试可能不同，重建训练题的 cohort 单独解释。[评估说明](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)

| 诊断集 | 行数 | 前端集成 BA | 加入 GLM BA | 族内残差后 BA | 最终 BA |
|---|---:|---:|---:|---:|---:|
| C1 开发 | 180 | 0.76391 | 0.73446 | 0.75796 | 0.73164 |
| C2 评估 | 300 | 0.71037 | 0.69419 | 0.71547 | 0.69258 |
| 重建题压力 | 318 | 0.51835 | 0.51514 | 0.51210 | 0.45897 |

两路 GLM 共覆盖 239 条，Native 五折均覆盖全部 798 条。原版 Comp 的检查点搜索命中仅有配置的目录，前端随后按原版逻辑回退。C2 的 GLM 改对 3、改错 7；族内阶段改对 9、改错 3；字形阶段改对 0、改错 9。逐条表同时保留了原决策与其分数排序。[逐条候选与判决](../runs/atlas_0719_20261123/trace/baseline_decisions.parquet) · [完整分数](../runs/atlas_0719_20261123/trace/atlas_baseline/stage_scores.npz) · [阶段汇总](../runs/atlas_0719_20261123/trace/baseline_summary.json) · [GLM 覆盖](../runs/atlas_0719_20261123/trace/atlas_baseline/glm_report.json)

## 似然对照

用挂载的 Qwen3-1.7B Base 学习真类条件下的正文 NLL。注意力的四种投影使用共享下投影和按类上投影，另加每类一个可学习的连续前缀。该任务适配以 256 token 上限和 rank 8 为首轮规模。训练使用经过题目证据、标签和长度筛选的公共 Open Pool，排除与三个带标签样本组及官方训练集重合的正文。在 C1 上按真实类别可参与复核的错例数对候选模型对排序，累计选择最多三个互不重叠的模型对。若开发样本中没有可复核的错例，则使用历史分析中的 Gemini Flash 和 GPT-5.6 两组。[ATLAS 方法](https://arxiv.org/html/2610.19583v1#S3.SS2) · [本项目实现](../src/ntllm/atlas_likelihood.py) · [首轮配置](../configs/atlas_likelihood_pilot.json) · [Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool)

在同一个检查点比较平均 log likelihood 和按竞争条件逐 token 截断后的似然，温度为 1、2、4、8。只在底座主选与前三候选中的同组对手分差小、两者都通过原字形规则时计算两个条件。队列优先分差较小的正文，覆盖上限 12%。开发集用于标定似然分差，并在 0.10、0.25、0.50 三档中选权，相同开发 BA 时取较小权重。零权重为基准。评价集与重建题压力集分别报告同底座改判，正式比赛候选使用截断分支。[ATLAS 聚合定义](https://arxiv.org/html/2610.19583v1#A2.SS1) · [部署逻辑](../src/ntllm/atlas_notebook.py)

C1 选择的三个模型对为 **GPT-5.6 Luna / GPT-5.6 Sol**、**GPT-5.5 / GPT-5.6 Terra** 与 **Claude Opus 4.8 / Claude Opus 5**。固定路由在 C1 覆盖 13 条，其中 5 条的备选是真类，6 条原判正确。独立 C2 覆盖 19 条，其中 8 条备选是真类，7 条原判正确。318 条重建题样本中的 7 条路由候选都缺真类，该压力集主要覆盖 Gemini、Kimi、Qwen，可用于观察这些类被误改的情况。[分歧筛选](../runs/atlas_0719_20261123/trace/track_selection.json) · [实际训练配置](../runs/atlas_0719_20261123/trace/atlas_selected_config.json)

## 同底座实测

六类共训练 7,200 条，完成 900 步。预测与基准按 ID、真类、题号及数据来源一对一对齐，798 条原判逐条一致。平均似然与截断似然均采用 C1 给出的 0.10 权重，两种聚合在这 798 条上作出了完全相同的最终判决。[训练报告](../runs/atlas_0719_20261123/atlas/atlas/atlas_report.json) · [逐条预测](../runs/atlas_0719_20261123/atlas/atlas/paired_decisions.parquet)

| 数据 | 改对 | 改错 | 净改对 | 总改判 | 前后 BA |
|---|---:|---:|---:|---:|---:|
| C1 开发，180 条 | 0 | 0 | 0 | 0 | 0.73164 → 0.73164 |
| C2 独立评估，300 条 | 1 | 0 | +1 | 2 | 0.69258 → 0.69510 |
| 重建题压力，318 条 | 0 | 0 | 0 | 0 | 0.45897 → 0.45897 |

两次改判均发生在 C2。一条真值为 GPT-5.6 Luna，由 Sol 改成 Luna。另一条真值为 Sol，由 GPT-5.5 改成 Terra，仍为错误。C2 的候选对内有真类的 15 条中，两种聚合各排对 11 条。结果显示固定融合只兑现了 8 个可达错例中的 1 个，截断聚合在这批数据上未带来额外改判。[复算结果](../runs/atlas_0719_20261123/audited_atlas_comparison.json) · [两条改判详情](../runs/atlas_0719_20261123/clipped_changed_rows.csv)

11 月 24 日复核，本队原底座提交 `58754662` 与加 ATLAS 的提交 `58768879` 均为 **0.718**。按 Kaggle 显示精度，ATLAS 的公开榜差值为 **0.000**。同队 Sibling Odds v1 原样复现 `58768811` 为 **0.720**，比本队 Dennis 底座高 0.002。[ATLAS 正式对照](../runs/atlas_0719_20261123/official_comparison.json) · [Sibling Odds 原样复现](../runs/public_0720_sibling_odds_20261123/submission.json)
