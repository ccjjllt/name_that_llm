# Name That LLM 公开方案与 v6 提分研究 · 2026-11-05

截至北京时间 11 月 5 日上午，Kaggle MCP 将 Omer 和 Ezra 排在公开 Notebook 的前两位，两者的 `best_public_score` 都是 **0.668**，第三名 Matterhorn 为 **0.667**。MCP 也确认 `ccjjllt` 的提交 `57987571` 为 COMPLETE，公开分 **0.668**。Ezra 的九个代码单元只有一行与 Omer 不同，它按文件路径选择历史 Open Pool。我们的 v6 已实际加载同一份 190,588 行数据。Ezra 的冻结附件在可见元数据中有一个空引用。[MCP 排序与分数字段](../runs/public_scan_20261105/mcp-public-ranking.json) · [代码差异](../runs/public_scan_20261105/notebook-code-differences.json) · [Omer](https://www.kaggle.com/code/omerfarukmerey/0-668-verifier-matched-rows-pool-prompts) · [Ezra](https://www.kaggle.com/code/ezberch/nlm-v72)

队伍公榜第一是 Hod Carriers 的 **0.747**。讨论区有参赛者自报单模型约 0.670、0.67–0.68，这次检索未找到能复现这些分数的完整公开代码和权重。近期关于测试布局的讨论也没有提供主办方确认的每个 prompt 样本数。[队伍榜](https://www.kaggle.com/competitions/name-that-llm/leaderboard) · [单模型讨论](https://www.kaggle.com/competitions/name-that-llm/discussion/754860) · [布局讨论](https://www.kaggle.com/competitions/name-that-llm/discussion/755733)

## v6 实际计算什么

按正式提交使用的 Notebook 及其导出的脚本追踪。普通 commit 对三行示例走 `FAST_COMMIT`，这份日志能证实数据版本与附件，不能见证隐藏重跑的后处理路径。[v6 运行记录](../runs/public_0668_20261104/v6-completion.json) · [正式提交](https://www.kaggle.com/competitions/name-that-llm/submissions)

| 阶段 | v6 实际设置 | 影响 |
|---|---|---|
| 主模型 | 五组 LoRA 模型配置加一个 DeBERTa 分类器。Qwen3-14B、Qwen3-8B、Llama-3.1-8B 使用多层特征，并集成各自的多个 adapter | 同一模型配置可能包含多折和全量权重。模拟留出时，需要核对权重的真实训练范围 |
| 文本分类与同题特征 | 四种视图的 SVM 使用官方数据、匹配抽样和 Open Pool，决策分乘 12。主脚本用内容词 TF-IDF 聚类，阈值 0.15，再用组中心化特征和部分模型的 Gaussian 头 | 12 是分数乘数。聚类质量会改变哪些行能获得同题证据 |
| 概率融合 | 互补集成占 15%。当前两类概率差小于 0.32 时尝试 ATLAS，混合权重 0.20。随后做软边际调整，强度 0.30，目标比例向训练先验收缩 0.35 | 概率温度会同时改变 ATLAS 覆盖范围和后续重排。正式参数使用 `--post 0`，主脚本未启用最后的硬 Hungarian 分配 |
| 最后重排 | 设计顺序是标点证据、同厂商 V40A、V34 pairwise verifier。V40A 仅在家族内重新分配概率，最大强度 0.10。V34 在前两类概率差小于 0.35 时生效 | 应保存各阶段预测，分别统计救回和损失的样本 |

[实际生成的主脚本](../runs/public_scan_20261105/generated/infer.py) · [互补集成](../runs/public_scan_20261105/generated/comp_stack.py) · [家族重排](../runs/public_scan_20261105/generated/sibling_style.py)

## 首选实验：拆开最后三步

从 Kaggle 下载的 v6 普通运行输出 `infer.py` 在第 636 行引入 `pathlib as _V38Path`，第 688 行却直接使用 `pathlib.Path`。在普通新 Python 命名空间复放这个表达式会触发 NameError。脚本此前已写过基础提交文件，最终覆盖文件的代码在该行之后。另一个评估差异是 `--sim` 会跳过最后整个重排区段。这解释了为什么直接比较脚本自报的 sim 分数无法评价最后三步。我们确有 Kaggle COMPLETE 0.668 的记录，现有证据不足以判断隐藏重跑最终使用了哪一步的预测。[名称复放](../runs/public_scan_20261105/v6-code/path_probe.json) · [实际导出脚本](../runs/public_scan_20261105/generated/infer.py)

我们用一个公开 Qwen3-8B LoRA 模型的折外预测单独重放这些操作。其 2,699 行覆盖 150 个 prompt，五折各含 30 个完整 prompt。文件按作者注明的官方训练顺序关联 ID，并以标签顺序、分组完整性和作者公布的 BA 精确回验。原文件没有逐行 ID，基础模型的训练由作者完成。使用其公布的温度 1.5。每折只在该折训练数据估计标点统计、训练重排器。这些是代理模型的本地结果。[公开 OOF 来源](https://www.kaggle.com/datasets/utkarshkviim/ntl-lora-v3-qwen3-artifacts/versions/1) · [对齐记录](../runs/public_scan_20261105/fusion_probe/audit.json)

| 在同一组折外预测上重放 | OOF Balanced Accuracy |
|---|---:|
| LoRA 基础分 | 0.55937 |
| 加 v6 标点重排 | 0.55751 |
| 加标点、再加 V40A 家族重排 | 0.56047 |
| 只加 V40A 家族重排 | **0.56381** |

只加 V40A 时，32 条预测改变，16 条由错变对，4 条由对变错。相对基础分的五折变化全部为正。按 prompt 配对重采样 2,000 次，增量的描述性 95% 区间为 **+0.00184 到 +0.00741**。单加标点救回 4 条、损失 9 条。首选实验是在可信的 v6 主集成折外分数上，对照基础分、只加 V40A、加标点再加 V40A。固定主模型及 ATLAS，只改变这一步。[复放脚本](../runs/public_scan_20261105/style_probe.py) · [逐折结果与区间](../runs/public_scan_20261105/style-summary.json)

## Verifier 有错行，但修复本身没有显出提分

`v34_pairwise.py` 每 128 行分块。检索参考文本时使用当前块，但计算字符余弦、词余弦、长度比、标点相似度时，用全局数组索引 `q-start` / `z-start`。从第二块起，这四个特征会取第一块正文；A/B 特征和选中的参考文本仍属于当前行。三行示例不会触发这种差异。[实际生成的 verifier](../runs/public_scan_20261105/generated/v34_pairwise.py)

进行了五折配对实验。每折两组共用同一个拟合好的 verifier、参考集和随机种子，仅切换四个特征的索引。留出 prompt 的 A/B 注释全部剔除，保留原代码在无标签查询文本上拟合词表的做法。2,699 条 OOF 中，原索引独立 BA **0.04779**，修正后 **0.03817**。每折最初 128 行完全一致。剩下的 2,059 行里有 1,865 条改变首选标签，修正版救回 72 条、损失 98 条。五折中两折升、三折降。[五折原始结果](../runs/public_scan_20261105/verifier_ab/metrics_oof.json)

Verifier 在 v6 中只作小权重修正。我们又在上述公开 LoRA OOF 上复现 v6 的 0.35 概率差条件，1,751 行符合条件。

| 修正权重 | 原索引融合 BA | 修复索引融合 BA |
|---:|---:|---:|
| 0 | 0.55937 | 0.55937 |
| 0.05 | 0.56011 | 0.56011 |
| 0.10 | 0.56048 | 0.55937 |

因此，下一次实验应使用正确索引并保留 **权重 0** 对照。进一步改造时，值得让训练时的参考文本也由推理时相同的每类近邻检索器选出，避免训练看随机正负对，而推理只看每类最相似的几个样本。可先集中在易混型号上，比较相同算力下救回与损失的条数。公开的 Sambit 0.667 同时改过索引和 top-2 条件，外部行也不同，不能隔离其中一项的作用。[融合结果](../runs/public_scan_20261105/fusion_probe/metrics.csv) · [Sambit Notebook](https://www.kaggle.com/code/sambitpadhan/ntl-four-top2-gate)

## 第二条路线：同题数据与易混型号

**先改训练数据的分组键。** 互补集成使用 `prompt_group` 汇集同题外部文本。对冻结 v13 单独按现有长度、去重和每题至少 12 类条件计算，现有键给出 **2,560 组、36,618 行**。改用 `prompt_canon` 并只选 own_bank、own_text、other_text 三种题目来源证据，得到 **4,096 组、63,048 行**。这些是池内可用量，最终 stack 还会与其他来源合并。建议先固定共同样本和每类权重，仅更换分组键，再单独增加新组。尽量同时匹配来源构成，防止把新增来源的作用归因于分组键。[统计和选择口径](../runs/public_scan_20261105/v6-code/pool_panels.json)

**再调整查询文本的分组。** 用 v6 的 `capped_agglo` 原函数，在官方数据上模拟五个各 30 个新 prompt 的验证池。对于每题取 1–6 条、三个固定种子的布局，阈值从 0.15 降到 0.10，正确同题配对的召回从 **77.94% 提至 91.20%**，配对精度从 **99.35% 降至 98.52%**。这衡量聚类，对分类 BA 的作用还要通过中心化头/Gaussian 头验证。小验证池的词频分布与隐藏测试不同。可在不改模型的前提下比较两个阈值，同时报告小组、单例、短文本上的分类变化。[五折布局探针](../runs/public_scan_20261105/cluster_probe.py) · [全部阈值与布局结果](../runs/public_scan_20261105/cluster-summary.csv)

**聚焦型号混淆与短文本。** 在同一公开 LoRA 代理上，Gemini 3.7/3.8 Flash 相互误判 **84 条**，两款 GLM-5.3 相互误判 **68 条**。低于 60 词的 555 条 BA **0.42691**，至少 160 词的 654 条 **0.63546**。可固定训练规模和来源，给这些型号增加长度匹配的短文本，或训练只重排该型号对的轻量分类头，比较逐类召回及跨家族误伤。v6 原有的 24,443 条匹配抽样已让弯双引号出现率对官方文本的逐类平均绝对差，从全池的 3.47 个百分点降到 0.47 个百分点；短文本比例差只从 2.56 降到 2.14。先补长度与型号的联合分布，比直接加倍全部样本更值得对照。[外部池统计](../runs/public_scan_20261105/pool-summary.json) · [代理模型混淆矩阵明细](../runs/public_scan_20261105/fusion_probe/directed_confusions.csv)

## 后续实验的比较口径

1. 用训练范围可追溯的折权重或已有 OOF 文件重建主集成。三个 v5 adapter 家族均包含五个折权重和全量权重；在 `--sim` 里拿全量权重预测“留出组”会污染评估。使用各权重自己的留出映射，再按文本 ID、标签顺序汇总。每折重新拟合官方标签参与的选择和统计，并分别保存主集成、ATLAS、V40A、verifier 的预测。
2. 首先比较最终重排，再比较 canonical 分组键与数据增量，之后才比较 ATLAS 预算。ATLAS 当前在低 margin 候选中按长度从短到长评分，预算耗尽时保留已完成的行。先记录实际覆盖及每千 token 的净救回数，再评估把部分预算预留给易混型号的效果。普通 commit 的耗时不能估计隐藏测试覆盖率。
3. 同一组预测上报告 OOF Balanced Accuracy、各类召回、短文本结果以及按 prompt 的配对差值。上述 CPU 结果用于排列实验优先级。完整 v6 至今可确认的 Public LB 为 **0.668**。[本账号提交记录](submissions.md)
