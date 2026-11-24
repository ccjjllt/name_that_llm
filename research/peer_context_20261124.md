# 利用同一题的不同回答做模型归因

## 比赛差距与错例

11 月 24 日，Kaggle 公开代码按关联成绩排序，RanJeet 的 Sibling Odds 与 zayd 的镜像均为 **0.721**。两个固定版本共有 29 个可执行代码单元，逐字及顺序一致。我们对 Sibling Odds 的正式提交是 **0.720**。榜首 YK **0.809**，第二名 NguyenThanhNhan **0.802**，本次未找到这两队相应的公开实现。从 0.720 到 0.809，需要消除当前宏平均错误的约 31.8%。同源代码不同提交的 0.001 分差不能归因给一个新增组件。[公开代码排序](https://www.kaggle.com/competitions/name-that-llm/code?sortBy=scoreDescending) · [RanJeet 评分版](https://www.kaggle.com/code/ranjeet258/name-that-llm-sibling-odds?scriptVersionId=372780350) · [zayd 评分版](https://www.kaggle.com/code/zayd88903/0-721-name-that-llm?scriptVersionId=373020342) · [榜单](https://www.kaggle.com/competitions/name-that-llm/leaderboard) · [本队回执](../runs/public_0720_sibling_odds_20261123/submission.json)

在 Dennis 固定分数上回放 Sibling Odds 末段，C2 的 300 条错 100 条。恢复被破折号规则推翻的 Gemini 首选后，还错 91 条，其中 59 条不足 100 词。32 条为 GPT 内部互判、25 条为 Gemini 内部互判、16 条为 Claude 内部互判。真实型号在标点前前三的有 70 条、在前五的有 82 条。即使用真值给每个错例挑选过去四个分类头之一，也只有 30 条能从它们的首选中找到正确型号。过去 Q4 读出与定向 LoRA 的预设权重都被开发集选为零；另一次 ELECTRA 续训的单模型 BA 虽提高 6.18 个百分点，加入旧 Q8 底座仅净改对 2 条。优先研究能区分这些具体错误的新证据。[错例](../runs/sibling_specialist_20261123/local/row_diagnostics.parquet) · [Q4 对照](token_fingerprint.md) · [定向训练](sibling_specialist.md) · [续训实测](leaderboard_research_20261123.md)

## 同题近邻能提供多少帮助

仅用官方训练正文拟合词面向量，分别在 C1、C2、重建压力集的可见正文中检索。每篇最多保留 3 个余弦相似度至少为 0.20 的邻居。若邻居对某型号的支持强于本行，降低本行对该型号的分数，幅度随相似度和邻居置信度变化。多个邻居撞到一个型号只取最强证据，仍可输出重复标签。保留原有 GLM、Fable 以及已生效的字形保护。实际推理使用的字段是正文和底座预测。两种底座分别在 C1 选权重，打平取较小值，然后固定。[实现](../src/ntllm/peer_collision.py) · [近邻清单](../runs/peer_retrieval_20261124/lexical/edges.parquet)

| 固定底座与改动 | C1 改对 / 改错 | C2 改对 / 改错 | C2 BA | 重建压力集改对 / 改错 |
|---|---:|---:|---:|---:|
| Sibling Odds 加近邻 | 1 / 0 | 3 / 0 | 0.68203 → 0.69050 | 5 / 2 |
| 仅恢复 Gemini 标点，再加近邻 | 2 / 0 | 4 / 0 | 0.70492 → 0.72097 | 12 / 4 |

第一行 C2 的三次正确改判全为 GPT，Sol → Luna、Astra → Terra、Sol → Terra 各一次。C2 的 829 条有向近邻中，818 条同题，覆盖 287/300 篇；能找到的正确边占所有同题有向对的 54.5%。按题重采样，在原版上的 BA 增量为 +0.85 个百分点，描述性 95% 区间为 0 至 +1.89；在标点修复上再加近邻为 +1.61 个百分点，区间为 +0.25 至 +3.46。两项组合相对原版共改对 13、改错 0，其中标点独自贡献 9。C2 已多次用于研究，只含 12 个真实型号。[原版对照](../runs/peer_retrieval_20261124/lexical/sibling_odds/comparison.json) · [标点对照](../runs/peer_retrieval_20261124/lexical/dash_only/comparison.json) · [区间与压力结果](../runs/peer_retrieval_20261124/lexical/sensitivity.json)

切掉 C2 约一半的可见回答，冻结上游分数、重做近邻检索，原版的平均增量缩至 +0.42 个百分点。另一项压力试验人为用“同作者回答另一道题”替换 10% 的有向近邻，并保留原相似度，增量变为 −0.20 个百分点。这是有意制造的错误匹配，不能估计隐藏集的重复率。社区作者也报告其 15 次提交在稀疏与完整面板上的排序相关性差别很大，后续五次提交的排序相关性又接近零。稀疏条件必须单列，不宜用一次回放的千分位差挑冠军方案。[扰动记录](../runs/peer_retrieval_20261124/lexical/sensitivity.json) · [社区作者的稀疏验证实验](https://www.kaggle.com/competitions/name-that-llm/discussion/753177#3546487)

## 双 T4 训练检验的假设

模型在措辞和内容选择上都可能有稳定偏好，附近有另一篇同题回答时，可以测量目标文本里哪些词和表达变得可预测。Sun 等在同一系列四个 Qwen2.5 型号上报告 59.8% accuracy，作者归因在近亲模型上比跨家族困难。Oyama 等发现 prompt 改动会系统改变似然表示，用条件与无条件似然的差可以观察其中一部分变化。这两个结果支持测量“参照改变了什么”，并未在本赛题验证这套方法。[ICML 2025，Idiosyncrasies](https://proceedings.mlr.press/v267/sun25z.html) · [EMNLP 2026，语言模型的条件似然表示](https://arxiv.org/abs/2603.18593)

冻结 Qwen3-4B Base，对同一段正文分别给空参照、长度接近的低相似参照、以及检索到的高相似参照。使用正文最前 160 token、参照最多 96 token；逐 token 记录真实词的负对数概率和两层隐藏表示，比较加入参照前后的差异。在 21,525 条外部文本上训练三个相同结构的 18 类读出头，涉及 22 个来源、7,297 个规范题号。随后分别在 2,699 条官方题组五折上做折内适应；针对外部面板再用全官方数据适应。训练和预测均保留单篇情形。只有实际取得参照的行才用“上下文分类头相对单篇分类头的分数变化”参与 Sibling Odds 末段。每组在 C1 从 0、0.15、0.35、0.70 中独立选权重；另在相同权重比较高、低相似参照。该设计允许同一题的两篇回答出自同一型号。[训练配置](../configs/peer_context.json) · [实现](../src/ntllm/peer_context.py) · [数据清单](../runs/peer_retrieval_20261124/training/manifest.parquet) · [GPU 训练](https://www.kaggle.com/code/zhuating/ntl-peer-context-attribution/versions/1)

C2 之外，预先冻结的 GEN003 四个 Claude 型号覆盖 77 个新题。每题可见 1、2、3、4 答的题数分别为 20、19、19、19，合计 191 条，其中 106 条不足 100 词。两个 GPU 实验使用逐条相同的正文，底座只收到 id 和正文，仍可输出任何一个 18 类标签。新训练排除该发布者及已知评估题；第三方底座权重的全部训练史未知，另记录本次挂载的实际语料有无正文重合。原作者标注了 Opus 4.8 的风格偏移。独立面板主要回答 Claude 的问题，不能验证 GPT 和 Gemini 的收益。旧评分轨迹的三个公开面板共 798 条，其中 766 条取到参照且都为同题；两个训练外来源中是 611/1,144，迁移时检索可靠性明显下降。[面板组成](../runs/peer_retrieval_20261124/independent/panel.json) · [同底座独立回放](https://www.kaggle.com/code/zhuating/ntl-gen003-sparse-trace/versions/1) · [配对复算](../src/ntllm/peer_context_report.py)

## 下一笔算力投向哪里

1. **先解决 GPT 和 Gemini 的独立数据缺口。** 新核实了 120 组三型号 GPT-5.6 的同题同路由生成，Luna、Sol、Terra 各 120 条。对 Open Pool v15 无相同或近重复正文。对 v18 有 189 条正文逐条重合，若考虑同一道题的其他型号，117/120 组有暴露。适合检查只用 v15 训练的组件。下一批完整底座评估数据应来自未被其语料收录的新请求，并同时含同一型号的重复回答、同主题不同请求、单篇及短文，记录实际返回型号和词数设置。新标签带来的价值在于能分开查“型号挑错”和“题找错”。[外部源核验](../runs/peer_retrieval_20261124/holdout/audit.json) · [原始数据](https://www.kaggle.com/datasets/elliotthankinson/ntl-tells-v4) · [Open Pool v18](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/18)
2. **把易混主题的不同请求作为负例。** 在同一批 120 个 GPT 题上，固定原词面规则。每题三答时，761 条边有 634 条同题；只留两答时是 218/329；每题仅一答时仍连出 28 条边，全部来自同题材的不同请求。用官方 120 道题训练词面验证器，用另 30 题选阈值，再到 GPT 面板评估。三答精度从 83.31% 到 85.44%，双答从 66.26% 到 68.13%，两者真边全保留。但单答的 28 条假边一条也没被拦下。下一步值得训练的是区分同题材不同请求的语义判别器，训练负例中要有相同主题、不同限制条件的真实请求。是否扩大训练量取决于它能否拦住这批假边并保住真正的同题匹配。[固定规则的迁移实测](../runs/peer_retrieval_20261124/holdout/lexical_fixed_holdout.json) · [词面验证器对照](../runs/peer_retrieval_20261124/holdout/pair_verification.json)
3. **给单篇短文增加独立的词序证据。** 可借鉴 EMNLP 2025 的 PROFILER，在同一次前向中测“当前位置给附近多个已出现词的预测”，比较不同位移的损失和相关结构。我们之前的 Q4 测过实际下一词的 NLL、名次和熵，本实验则观察给定另一篇正文后的变化。多位置读出可在无近邻时工作。第一轮应在同一代理、同一数据、同等训练量下比较仅下一词与多位置两臂，单列短文、跨来源和前三候选中的净改判。论文报告的来源检测指标与赛题 18 类 BA 不同。[PROFILER 论文](https://aclanthology.org/2025.emnlp-main.1265/) · [作者代码](https://github.com/MarkGHX/Profiler) · [既有 Q4 对照](token_fingerprint.md)

## 双 T4 训练实测

本地用 GPU 输出的逐条分数重建了 798 条旧面板的全部设定，并核对独立面板的 191 个 ID、正文哈希和原判。C2 有 295 条、独立 Claude 有 163 条获得高相似参照，这两组取到的高相似参照均由同题回答提供。对应的低相似参照均为异题。独立面板的 20 条单答样本保持原判，191 篇正文与运行时外部语料池及官方训练正文都无完全重合。第三方检查点的全部训练史未知。按 C1 固定的权重，同题为 0.35、异题为 0.15。[配对复算](../runs/peer_retrieval_20261124/audit/report.json) · [逐条判决](../runs/peer_retrieval_20261124/audit/paired_predictions.parquet) · [语料比对](../runs/peer_retrieval_20261124/independent/gpu/atlas_baseline/body_overlap.json)

| 面板 | 同题参照相对原底座改对 / 改错 / 净数 | BA 前后 | BA 增量（百分点） | 按题重采样 95% 区间（百分点） |
|---|---:|---:|---:|---:|
| C2，300 条 | 6 / 10 / −4 | 0.68203 → 0.68559 | +0.36 | −2.14 至 +3.17 |
| C2 短文，153 条 | 3 / 7 / −4 | 0.62178 → 0.62465 | +0.29 | −3.99 至 +4.93 |
| 独立 Claude，191 条 | 1 / 2 / −1 | 0.84893 → 0.84337 | −0.56 | −2.47 至 +1.11 |
| 独立 Claude 短文，106 条 | 1 / 1 / 0 | 0.76843 → 0.76843 | 0.00 | −2.42 至 +2.42 |
| 重建压力集，318 条 | 8 / 5 / +3 | 0.45897 → 0.46859 | +0.96 | −1.24 至 +3.17 |

C2 的正确总数从 200 降至 196。BA 按类等权，只有 11 条的 Opus 5 改对了 2 条，足以抵消一部分样本量较大的型号的召回下滑。Gemini 3.6 Flash、3.8 Flash、3.1 Pro 的改错数合计 7 条。独立面板共改判 5 条，短文中两条影响正误的改判同属 Opus 4.8，恰好抵消。全组另有一条 106 词的 Opus 4.6 从正确改为 Fable。[变动清单](../runs/peer_retrieval_20261124/audit/changed_rows.parquet)

| 面板 | 权重（同题 / 异题） | 同题相对异题改对 / 改错 / 净数 | BA 增量（百分点） |
|---|---:|---:|---:|
| C2 | 0.35 / 0.15 | 8 / 7 / +1 | +1.90 |
| 独立 Claude | 0.35 / 0.15 | 1 / 3 / −2 | −1.08 |
| C2 | 0.35 / 0.35 | 7 / 8 / −1 | +0.39 |
| 独立 Claude | 0.35 / 0.35 | 2 / 3 / −1 | −0.60 |
| C2 短文 | 0.35 / 0.35 | 4 / 7 / −3 | −1.66 |
| 独立 Claude 短文 | 0.35 / 0.35 | 2 / 1 / +1 | +0.96 |

四个全样本同题与异题比较的区间均跨零。单篇 4B 分类头的官方折外 BA 为 0.30598，同题参照头为 0.30449。两项对照均缺少可迁移的同题优势。这版缺乏正式提交依据，保留逐条打分供错例分析。此前词面近邻的正收益来自不同改判方法，其固定权重在独立 Claude 面板也没有净改对。[完整指标](../runs/peer_retrieval_20261124/audit/report.json)
