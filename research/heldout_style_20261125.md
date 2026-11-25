# 十八类留源与风格编码对照

以公开数据的实际返回型号及来源台账建立新留源，测量已有诊断面板缺少真值的型号。Prvsiyan v13 的 895 篇来自 69 个多型号题组，覆盖全部 18 类。逐条排除了与官方训练、完整 Open Pool v18、原版堆叠纳入的 Prvsiyan 清洗正文和前轮评估正文的规范化精确重合。69 个原始题面与 Open Pool v18 的题面注册表无规范化精确重合。进一步排除前轮 Kimi 研究用过的整题后，保留 316 篇、23 题、18 类。清洗核验覆盖可见语料，第三方预训练权重的语料历史未知。较完整的同题组与已有稀疏面板分别报告。[Prvsiyan v13](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/versions/13) · [来源审计](https://www.kaggle.com/code/zhuating/ntl-heldout-source-audit/versions/2) · [计数报告](../runs/heldout_source_20261125/audit_v2_remote/source_audit/report.json)

同次审计在 Enaimaan 的公开 OpenAI 回答中保留 5,237 篇，覆盖 1,352 个题组，其中 1,189 组各有四款 GPT 回答。实际服务身份由发布者校验，公开表按行保存请求和题组信息。Claude GEN003 剔除已用评估正文后保留 5,639 篇。Colton 留源有大幅破折号频率偏移，适用于检验标点决策在偏移条件下的误伤。[Enaimaan v3](https://www.kaggle.com/datasets/enaimaan/ntl-openai-generated-responses/versions/3) · [Claude GEN003](https://www.kaggle.com/datasets/tadeuhayashida/ntl-gen003-claude-panel/versions/1) · [Colton v7](https://www.kaggle.com/datasets/coltondowning/ntl-panel-completion/versions/7)

Mark Barney v2 的 597 篇正文均记录了相同的请求与返回型号。与 Open Pool v18 和官方训练的规范化正文重合各为零，与已用历史面板重合 318 篇。剩余 **279 篇、72 个重建官方题组**，其中 **132 篇少于 100 词**。以同题 Gemini 3.7 与 3.8 的相对长度为例，69 个可对照的题里有 34 个在两来源中方向反转。两款模型的跨来源正文都满足长度差不超过 12 词且长度比不超过 1.15 的题只有 1 个。这份数据适用于同题跨来源复验，独立题组评估继续由 Prvsiyan 承担。[原始数据](https://www.kaggle.com/datasets/markbarney/ntl-matched-passages-four-models/versions/2) · [Kaggle 正文审计](https://www.kaggle.com/code/zhuating/ntl-matched-provider-audit/versions/1)

本轮比较 StyleDistance 和同架构原始 RoBERTa。StyleDistance 的作者用合成同内容异风格三元组和 Reddit 作者数据学习写作风格距离，任务仍与本赛题的近亲版本归因有差异。两臂对同一原文做冻结编码、均值池化及单位归一化，使用同一外部训练样本和同样的 18 类线性头。官方数据按题五折隔离，独立留源使用全部官方训练数据调整线性头。比较 BA、短文、相对原版底座的净改对，以及原版错误且真类不在前三时的新增候选。两臂的固定诊断融合权重均为 0.15。[StyleDistance 原论文](https://aclanthology.org/2025.naacl-long.436/) · [公开权重](https://huggingface.co/StyleDistance/styledistance) · [评估实现](../src/ntllm/style_distance.py)

来源均衡读出的两臂各抽到 44,522 条训练正文。用训练结束保存的两套权重补齐同一份 895 篇留源的逐条分数，归档正文仅用于检查重载分数和耗时。成员重排、来源均衡读出和风格读出的留源评估都核对输入 ID 顺序、逐条正文哈希与零权重原判。读出对照锁定 0.15 权重，报告 C2、895 篇全量、316 篇整题留出和各自短文的 BA、净改对与漏候选覆盖。[来源均衡训练](https://www.kaggle.com/code/zhuating/ntl-provenance-readers/versions/1) · [留源推理](../src/ntllm/provenance_reader_heldout_notebook.py) · [同底座复算](../src/ntllm/reader_transfer.py)

根据旧 GEN003 读出的结果，正式查看新留源预测前加测 Claude 族内融合。原判与新判必须同属 Claude，GLM、Fable 以及受排版改变的原判保持冻结。两臂均使用原 0.15 权重，与同一份零权重 Sibling Odds 作配对比较。

冻结风格表示的训练侧题目投影对照，使用与原 StyleDistance 相同的 9,000 篇外部样本和官方整题五折。在每折训练数据内先从表示减去类别与数据组均值，再从同题多型号组的质心学习 32 个题目方向。方向与拟合的型号质心差异正交。对照保留原表示，另把题号在模型、长度和数据组内打乱后执行相同秩的投影。所有臂的分类头结构和训练样本相同，推理只用单篇正文。比较真题方向和打乱方向的 BA、短文及底座前三之外新增候选。LEACE 提供线性消除的参考，Multi-Resolution 的同题中心化给出了同题信号的上限。这里使用受限正交投影并测其跨题收益。[LEACE](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html) · [Multi-Resolution](https://arxiv.org/abs/2610.28434)

## 同请求配对的训练供给

在相同训练前筛选下，v18 有 75,639 篇可用候选。Gemini 3.7／3.8 有 1,322 对同来源同题、两篇都少于 100 词且长度接近的正文。380 对同时保留两边规范化后的返回型号和相同路由、推理档位记录。该口径下 Luna／Sol 186 对、Luna／Terra 184 对、GPT‑5.5／Terra 103 对、Kimi／Flash 零对。元数据口径描述可用记录，忽略未标明的推理档位或路由会改变供给。[Kaggle 样本供给](https://www.kaggle.com/code/zhuating/ntl-pair-supply-audit/versions/1)

David Vertex v2 另有逐条实际发送用户消息、目标词数与 API 返回型号。两种 Gemini 的 2,450 对用户消息和目标词数逐字相同，均记录匹配的返回型号与 `STOP`。其中 1,120 对符合近等长短文口径，941 对在 v18 实际清洗后仍保留。普通抽样保留 265 对，来源均衡保留 170 对。较低的元数据计数未覆盖这批源日志证据。发布者统一记录无系统消息和供应商默认参数，逐条 API 请求 JSON 未发布。独立题组检验可用 Prvsiyan，Mark Barney 的 279 篇作为额外跨来源检查。[David 原始数据](https://www.kaggle.com/datasets/davidpiln/ntl-vertex-gemini-gen/versions/2) · [请求与样本 ID 审计](https://www.kaggle.com/code/zhuating/ntl-pair-supply-audit/versions/2)

固定来源均衡臂的全部 502 个配额分层后，配对数的最大值为 **257**。达到该数只需更换 87 篇，原来 44,522 篇中保留 44,435 篇。主要容量限制是 Gemini 3.8 在 David 来源的短文配额恰为 257 篇。[固定配额求解](https://www.kaggle.com/code/zhuating/ntl-pair-preserving-sample/versions/1)

将两款型号的 David 短文配额各扩大到 941 篇后，两臂仍各有每类 1,250 篇短文和全类 44,522 篇。五词长度区间、Vertex 批次、来源、路由、推理档位和其他正文保持完全相同。独立抽样已自然保留 **851 对**，完整配对为 **941 对**。两个样本共享 44,430 篇，差异集中在 92 篇替换。这次数据对照显示，仅以保留对数区分训练臂的空间较小。[扩大配额求解](https://www.kaggle.com/code/zhuating/ntl-pair-preserving-sample/versions/2)

GPU 对照锁定完整配对的 **44,522 篇**以及两轮的逐批输入，两臂都使用相同的带类别权重的交叉熵。实验臂在有同请求对的批次加权 **0.40**，根据两种标签分配的合计得分计算条件分配损失。每个配对在每轮出现一次，分数由单篇正文生成。官方正文用于外部迁移评估。C2 接回冻结的 Sibling Odds，再检查 Prvsiyan 的严格留题与 Mark Barney 的同题跨来源。既有同题中心化研究改变的是推理表示，这里测训练时使用已验证配对的效果。[Multi-Resolution](https://arxiv.org/html/2610.28434v1#A8) · [同批次训练实现](../src/ntllm/exact_pair_train.py)

Prvsiyan 的独立同底座复算对两臂使用相同的 0.15 融合权重。另测只允许 Gemini 3.7／3.8 之间改判的版本，遵循既有 GLM、Fable 和排版保护。全量、严格留题和各自短文同时报告两臂相对底座及彼此之间的净改对和 BA。通过批次哈希核对两臂编排，并单列混合精度跳步数与真实优化步数。Mark Barney 的 279 篇仍提供独立读出和同题辨认指标，此次同底座复算使用已有逐阶段轨迹的 C2 和 Prvsiyan。[训练对照](https://www.kaggle.com/code/zhuating/ntl-exact-pair-loss-readers/versions/1)

扩展到多型号的供给审计沿用 40–99 词、全组最大词差 12、最长与最短比例不超过 1.15。对已去重的 Enaimaan 5,237 篇，按规范题号有 379 组二路、41 组三路。改按日志中的有序消息和词数目标后，剩 36 组二路；要求记录的推理档相同后只剩 17 组二路，三路或四路均为零。GPT 标签由发布者核验，未公开逐条 API 原始回执。Astra 的已记录推理档为 `low`，另外三款为 `none`，发布者也明确实际设置一致性未获证实。[GPT 数据卡](https://www.kaggle.com/datasets/enaimaan/ntl-openai-generated-responses/versions/3) · [Kaggle 供给实测](https://www.kaggle.com/code/zhuating/ntl-matched-request-supply/versions/1)

Claude GEN003 的 5,639 篇均通过发布返回型号及结束状态的对齐。原题、系统文本和目标词数逐字段一致的近等长短文中，有 76 组二路、207 组三路、380 组四路。发布者注明 Opus 5 使用 2,000 token 上限，另外三款使用 700。基于同预算的三路供给另算，并分型号报告发布者风格筛查状态。四路计数描述文本请求和词数目标相同的供给。公开记录缺逐条 API 实际设置，不能推断四路生成条件完全相同。[Claude 数据卡](https://www.kaggle.com/api/v1/datasets/view/tadeuhayashida/ntl-gen003-claude-panel) · [Kaggle 供给实测](https://www.kaggle.com/code/zhuating/ntl-matched-request-supply/versions/1)

按发布者记录的相同 token 上限重分组，Opus 4.6、4.8、Sonnet 5 有 **402 个三路组、202 个二路组**，共 **1,610 篇**入选。Opus 4.8 的 1,416 篇已去重正文均未通过发布者的标点风格筛查，其他三款均通过。这个有完整分组的训练来源要保留风格筛查分层，且用独立发布者的 Claude 短文验证脚本是否记住了单一生成协议。同样的严格已记录设置口径下，GPT 只有 17 个二路组，当前不扩到 GPT 多路。[预算匹配复算](https://www.kaggle.com/code/zhuating/ntl-matched-request-supply/versions/2)

保持原来源均衡抽样的“型号 × 五词长度区间”计数后，保留 **308 个三路组、77 个二路组，共 1,078 篇**，其余 43,444 篇沿用原样。219 组因相应型号和长度区间容量不足整组退出；匹配正文对已冻结评估的正文和已知题面重合均为零。两臂共用同一份 44,522 篇和每轮逐批顺序。实验臂对匹配请求组枚举标签分配，以 `log(2)/log(k!)` 缩放分配损失并加权 0.40。对照只使用交叉熵。Prvsiyan 留源分别评估 185 个目标类真值、98 篇目标短文及其他类误伤，严格留题单独报告。[Kaggle 样本审计](https://www.kaggle.com/code/zhuating/ntl-claude-group-input/versions/1) · [组条件损失](../src/ntllm/claude_group_train.py)

FP32 StyleDistance 已完成同数据编码。895 篇独立留源上，独立线性头相对于 RoBERTa 改对 99、改错 78，BA 21.60% → 23.05%。415 篇短文改对 45、改错 39，BA 21.12% → 22.20%，前三真值覆盖率 44.10% → 51.81%。C2 以 0.15 回接原版，StyleDistance 改对 3、改错 3，BA 降 0.38 个百分点，短文改对 1、改错 2。风格向量在线性读出上出现跨来源互补，固定融合未改善 C2。双高斯、梯度选样和题目方向对照共享这份冻结表示。[FP32 Kaggle 评估](https://www.kaggle.com/code/zhuating/ntl-styledistance-fp32/versions/1)

双高斯在 C2 的独立 BA 比单高斯高 3.70 个百分点，比固定线性头低 3.78 个百分点。相对于线性头，895 篇留源独立复算改对 59、改错 113，BA 降 5.89 个百分点，短文降 6.11。以 0.15 接回 C2，同一底座改对 1、改错 3，BA 68.20% → 67.33%；短文 0／2，BA 62.18% → 61.42%。当前没有该混合头的留源同底座数据。[双高斯对照](https://www.kaggle.com/code/zhuating/ntl-style-mixture-audit/versions/1)

从训练题估计的投影在 895 篇独立留源上，相对原风格表示改对 21、改错 45，独立 BA 降 2.85 个百分点；短文降 2.66。严格留题 316 篇中，真实题投影整体比打乱题号的同秩投影低 0.08 个百分点。127 篇短文比打乱投影高 1.86 个百分点，仍比原表示低 0.53 个百分点。C2 固定融合改对 2、改错 3，BA 降 0.67 个百分点，短文 0／2。固定线性和这两种变化都未改善同底座 C2，暂停把风格头作为独立融合项推广。[题目投影对照](https://www.kaggle.com/code/zhuating/ntl-style-topic-projection/versions/1)

梯度相关性选样在 895 篇独立留源上，独立 BA 比三个同配额随机臂分别低 2.32、1.08、1.18 个百分点，比全量线性头低 3.49。以 0.15 接回 C2，改对 3、改错 3，BA 增 0.085 个百分点，结果与两个随机种子逐条一致。短文改对 1、改错 2，基于离散类别权重的 BA 增 0.77 个百分点，仍与同两个随机种子逐条一致。已将独立同底座复验改成重用三项诊断的冻结分数，直接量多个风格读出的候选互补和逐条改判，避免再次拟合。[梯度选样对照](https://www.kaggle.com/code/zhuating/ntl-gradient-selection-audit/versions/1) · [复验封装](../src/ntllm/style_frozen_transfer_notebook.py)

Gemini 同请求实测两臂均执行 5,566 次更新、零跳步，共看 89,044 篇次及 1,882 对次，批次顺序哈希相同。第二轮训练对正确分配由 637／941 到 745／941。独立读出在 C2 短文比交叉熵改对 8、改错 3，BA 升 2.80 个百分点，在 415 篇 Prvsiyan 短文改对 20、改错 28，BA 降 1.52。在同一 Sibling Odds 上以 0.15 融合，两臂在 C2 300 篇和严格留题 316 篇逐条相同。895 篇全量配对臂比普通臂改对 1、改错 0，BA 仅增 0.10 个百分点，短文净改判和 BA 持平。两臂相对原版均使 C2 全量和短文 BA 下降。Gemini 3.7／3.8 族内限定也没有提高 BA。这轮收益集中在训练和部分独立读出，未稳定传导到同底座预测。[同底座配对复算](https://www.kaggle.com/code/zhuating/ntl-pair-loss-transfer/versions/1)
