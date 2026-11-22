# Name That LLM 的 GLM 似然与分类读出

## 低推理力度与用户请求前缀

0.716 版本公开了两批真实 GLM 训练特征，分别覆盖 800 和 1,899 条官方回答。两批按 ID 合并，不改变已有 Native 的五个 prompt 折。把旧的 320-token 轨迹截取前 160 token 并重新计算同一组特征，可以在相同窗口上比较旧的 Max／空用户前缀与新的 Low／段落请求前缀。每种设置在四个训练折上重拟合相同的 54 维逻辑回归，在留出折上使用同一温度、权重和 GLM 后验更新规则。[0.716 代码](https://www.kaggle.com/code/estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix?scriptVersionId=371376285) · [公开的两批训练特征](https://www.kaggle.com/code/estopadilla/ntl-glm-prefix-ab)

| 前缀与正文窗口 | 同样 GLM 前 15% | 同等 token 预算 | 同预算覆盖数 |
|---|---:|---:|---:|
| Max、空用户、160 token | 0.6880 | 0.6879 | 669 |
| Max、空用户、320 token | 0.6872 | 0.6879 | 564 |
| Low、段落请求、160 token | **0.6928** | **0.6968** | 587 |

匹配窗口并固定同一批 404 条候选时，新前缀高 **0.48 个百分点**。换算到每 4,000 条 130,000 token 的同等工作量，新前缀比旧 320-token 高 **0.89 个百分点**，五折三升两平。后者将 GLM-5.3 召回从 0.667 提至 0.733，GLM-5.3-Flash 从 0.700 提至 0.773。同时改了推理力度和用户请求，表中衡量的是组合效果。此处的分数来自固定 Native 输入与真实 token 特征，0.716 榜单复现另记入比赛提交。[Kaggle 五折对照](https://www.kaggle.com/code/zhuating/ntl-glm-user-prefix-comparison) · [指标 JSON](../runs/glm_user_prefix_20261121/glm_user_prefix_metrics.json)

每个留出题目抽两条真实回答，重新按同一 token 预算分配 GLM 候选。20 次均匀抽样中，新前缀相对旧 320-token 版的平均 BA 从 **0.6841 到 0.6940**，有 16 次上升。按随机轮换的类别保留权重抽样时，平均从 **0.6837 到 0.6929**，有 15 次上升。不足 100 词的样本分别高 0.08 和 0.70 个百分点。重复抽样共享题目，结果衡量候选排序在不同批次组成下的稳定性。固定的 Native 分数保持原样，这也不涉及重新推断隐藏题组。[Kaggle 两条回答对照](https://www.kaggle.com/code/zhuating/ntl-glm-prefix-sparse-prompts) · [指标 JSON](../runs/glm_prefix_sparse_20261121/glm_prefix_sparse_metrics.json)

固定新前缀似然和融合权重后，比较纯 GLM 总概率排序与额外计入 Fable 概率的 0.711 排序。同等 token 预算下，两种方法都覆盖 587 条，其中 581 条相同。加 Fable 权重只改变两个最终判决，BA 从 **0.6968 到 0.6961**，Fable 召回持平。固定各折前 15% 时变化方向相反，BA 从 0.6928 到 0.6935。混合版采用 0.716 的纯 GLM 排序与分段裁尾，保留 0.711 集成的 Fable 回转和末段字形规则。[Kaggle 排序对照](https://www.kaggle.com/code/zhuating/ntl-glm-prefix-routing/versions/2) · [指标 JSON](../runs/glm_prefix_routing_20261121/glm_prefix_routing_metrics.json) · [混合版生成器](../src/ntllm/glm_prefix_hybrid.py)

在新前缀的固定融合权重下，把候选从 587 条扩到各折前 30%（809 条），用量从 87,409 增到 119,002 个 token 位置，balanced accuracy 都是 **0.6968**。两个 GLM 的召回各增 3.33 个百分点，同时 Claude Opus 4.8 降 2.67 个百分点，Opus 4.6 和 Opus 5 各降 1.33，Fable 和 Sonnet 5 各降约 0.67。继续增加覆盖的收益取决于能否减少这批候选中将 Claude 改判为 GLM 的错误。[五折指标 JSON](../runs/glm_user_prefix_20261121/glm_user_prefix_metrics.json)

在各折 GLM 总概率前 30% 的候选中，比较纯概率排序与加入 25% 预期改判收益排名的排序。用真实 160-token 似然轨迹计算融合后的改判，收益标签在各外折的训练侧按题目交叉拟合，排序只使用运行似然前的分数和文本长度。在相同预算下，两种排序的 balanced accuracy 均为 **0.69683**。新排序改对两条、改错两条，五折一升、三平、一降，两路正文与前缀合计用量从 173,698 降到 173,582 个 token。保留纯 GLM 总概率排序。[Kaggle 排序对照](https://www.kaggle.com/code/zhuating/ntl-glm-prefix-learned-routing/versions/1) · [指标 JSON](../runs/glm_prefix_learned_routing_20261122/v1/glm_prefix_learned_routing_metrics.json)

在固定的 587 条上，另拟合两个低维校准头，将族间间隔与 GLM 两型号的分差向原融合值收缩。用于拟合校准头的 GLM 分数在每个外折的训练侧交叉拟合，Native 沿用固定的折外参考。加入两个校准头，balanced accuracy 从 **0.69683 到 0.69720**，不足 100 词的回答从 0.60352 到 0.60421。五折两升、两平、一降，变化来自族间校准。把同一组头应用到 809 条候选，仍有上述 GLM 召回与 Claude 召回的取舍。混合版沿用原融合权重。[Kaggle CPU 对照](https://www.kaggle.com/code/zhuating/ntl-glm-prefix-residual-calibration) · [指标 JSON](../runs/glm_prefix_calibration_20261121/output/glm_prefix_calibration_metrics.json)

固定同样的 587 条候选、Fable 回转阈值和 0.15 的字形概率乘数，复现 mathleteutd 的选择性概率交接。对于字形步骤前仍判为 GLM 或触发 Fable 回转的行，改由似然后的分数驱动字形规则。在 2,699 条 Native＋GLM 折外回答上，改动产生 **0 次改判**，五折均持平，balanced accuracy 为 **0.69460**，不足 100 词的回答为 0.60280。这里加入了 Apex 残差、Fable 回转和字形步骤，输入仍是 Native 单成员的折外分数。[公开变体 v5](https://www.kaggle.com/code/mathleteutd/name-that-llm-320-token-glm-fable-typography/versions/5) · [Kaggle CPU 对照](https://www.kaggle.com/code/zhuating/ntl-glm-fable-typography-ab/versions/1) · [指标 JSON](../runs/glm_fable_typography_20261121/glm_fable_typography_metrics.json)

### 缩短正文窗口的代价

从低推理力度与用户前缀的真实轨迹重新汇总特征，按原折拟合每个窗口的分类头。四种窗口使用同一批 587 条候选、同一顺序和融合权重。160 token 重算精确复现总体与逐折指标。表中是两路合计的正文 token，各档还使用相同的 34,100 个前缀 token。[Kaggle CPU 对照](https://www.kaggle.com/code/zhuating/ntl-glm-prefix-window-study/versions/2) · [完整指标 JSON](../runs/glm_prefix_window_20261121/v2/glm_prefix_window_metrics.json)

| 正文窗口 | Balanced accuracy | 不足 100 词 | 两路正文 token | 正文节省 |
|---:|---:|---:|---:|---:|
| 160 | **0.69683** | 0.60352 | 139,598 | 0% |
| 128 | 0.69460 | **0.60569** | 123,804 | 11.3% |
| 96 | 0.68978 | 0.60365 | 102,738 | 26.4% |
| 64 | 0.67680 | 0.59183 | 74,282 | 46.8% |

128 token 的总体分数低 0.22 个百分点，四折下降、一折持平。两个 GLM 的召回各降 0.67 个百分点，Fable 召回降 2.68 个百分点。公开双 T4 的两批分段计时由 115,980 增至 290,068 个位置，Flash 与 Full 合计的每位置时间增量为约 0.033 秒。以相同候选比例外推到 4,000 条，128、96、64 token 预计分别腾出约 6、15、26 分钟，固定权重加载开销仍存在。当前保持 160 token，待 Kimi 有真实计时和准确率再比较是否值得交换这段预算。[800 条的公开计时](https://www.kaggle.com/code/estopadilla/ntl-glm-prefix-ab/versions/1/output) · [1,899 条的公开计时](https://www.kaggle.com/code/estopadilla/ntl-glm-prefix-ab/versions/2/output)

## GLM 后验保留与候选选择

公开 0.709 方案在生成式似然结束后，会用似然之前的集成概率计算字体规则，并覆盖部分改判为其他家族的预测。我们比较两种改动。第一种在同样的 15% 文本覆盖上限内同时考虑 GLM 概率、与非 GLM 候选的间隔和估计 token 开销。第二种把 GLM 总概率和 GLM 族内比例保留到最后一步，同时保留其他类别之间的相对概率。[sanpier v2](https://www.kaggle.com/code/sanpier/0-709-320-token-glm-likelihood-qwen-ensemble?scriptVersionId=371885729)

Kaggle CPU 对照使用 2,699 条官方训练文本、Native Qwen3-Embedding-8B 的公开逐折概率和两个 GLM 的 320-token 分数。似然分类头逐折拟合，按 `prompt_id` 划分。表中数值均为这组折外预测的指标，排序分别使用 GLM 家族概率总量和按预期改判收益与开销排序。[公开 GLM 分数](https://www.kaggle.com/code/estopadilla/ntl-glm-likelihood-train/versions/3) · [Kaggle 实验](https://www.kaggle.com/code/zhuating/ntl-glm-likelihood-dynamics/versions/2)

| 文本选择与改判 | Balanced accuracy | GLM-5.3 召回 | Flash 召回 | Fable 召回 |
|---|---:|---:|---:|---:|
| Native plain | 0.6516 | 0.253 | 0.427 | 0.416 |
| 家族概率排序，仅保留 GLM 获胜 | 0.6824 | 0.640 | 0.680 | 0.376 |
| 自适应排序，仅保留 GLM 获胜 | 0.6820 | 0.640 | 0.680 | 0.369 |
| 家族概率排序，保留 GLM 后验 | 0.6872 | 0.607 | 0.647 | 0.463 |
| 自适应排序，保留 GLM 后验 | 0.6872 | 0.607 | 0.653 | 0.456 |

保留后验相对仅保留 GLM 获胜提高约 **0.49 个百分点**。Fable 召回改善，代价是两类 GLM 召回略低。自适应排序在这组对照中没有进一步提升总体成绩。加入曲率、排名与熵的短程序列统计后，自适应排序组合为 0.6842。单独训练 GLM/Fable/其他家族的分层模型后为 0.6768，暂时继续使用原汇总统计。[指标 JSON](../runs/glm_dynamics_20261121/v2/glm_dynamics_metrics.json) · [逐 token 统计的研究方法](https://aclanthology.org/2024.findings-naacl.8/)

这组对照以公开 Native plain 概率近似原集成的概率输入，完整集成另有字体修正和中心化。相同 sanpier v2 的原样提交（`58650152`）获公开榜 **0.709**，自适应排序与保留后验组合提交（`58650603`）为 **0.701**，低 0.8 个百分点。这一榜单对照衡量两项改动的组合，后续混合版采用原候选排序。[比赛提交](https://www.kaggle.com/competitions/name-that-llm/submissions)

两种 GLM 的 **2,699 条**公开训练轨迹在 prefix 长度和真实目标 token ID 上逐条对齐。加入同 token 似然差、熵和排名的十维特征后，GLM 两型号之间的折外 AUC 从 **0.9540 到 0.9645**。直接用新模型更新 GLM 后验，集成 BA 从 **0.6872 到 0.6850**。固定原 GLM 总质量、只用配对特征分配两型号概率时为 **0.6879**，增加约 0.07 个百分点。这一增量仍小。[Kaggle 对照](https://www.kaggle.com/code/zhuating/ntl-glm-aligned-token-features) · [指标 JSON](../runs/glm_paired_20261121/v2/glm_dynamics_metrics.json) · [Kirill Tymchishin 的配对特征](https://www.kaggle.com/code/kirilltym/lora-trained27-evidence-for-close-calls/versions/24)

## 用通用模型替代真实作者似然

对同一批文本提取 Qwen3-4B-Base 与 Qwen3-4B 的逐 token 条件概率、排名、熵、前 10 项质量和标准化惊异度。两组特征及其差值分别用分位数汇总和卷积读出，再平均两个读出的概率。以作者标注训练的小模型每次只接触其他 prompt 的样本。两条通用参考的独立准确率为 **0.2415**。按 15% 权重叠加到 Native plain，准确率从 0.6516 到 0.6535，70 词以下从 0.5129 到 0.5123。30% 和 50% 权重下整体降至 0.6509 和 0.6442。本轮结果不支持把两种通用参考的似然加入整个集成。[Kaggle 实验](https://www.kaggle.com/code/zhuating/ntl-reference-likelihood) · [指标 JSON](../runs/likelihood_expert_20261121/likelihood_metrics.json) · [SeqXGPT](https://aclanthology.org/2023.emnlp-main.73/)

## 跨族轨迹与同题风格

跨族轨迹复用 0.709 集成已有的前向计算，以四个深度的早、中、晚段表示变化训练三个判别器。训练使用同题配对，部署仅在 Kimi↔GPT-5.5、Fable↔GLM 和 Fable↔Flash 的低间隔决策中应用。输入表示沿用已经用官方标签训练的全量适配器。榜单提交 `58651182` 与其同源原样提交 `58650152` 均显示 **0.709**，因此没有观察到榜单增益。后续 GPT-5.6 轨迹使用不同的冻结 Native 表征与族内任务，单独与 0.716 原版对比。[Kaggle Notebook](https://www.kaggle.com/code/zhuating/ntl-cross-family-trajectories?scriptVersionId=372057798) · [比赛提交](https://www.kaggle.com/competitions/name-that-llm/submissions)

风格读出用 Open Pool 中 1,200 个 prompt 的 18,086 条回答训练。对比目标把同题异作者作为难负例、异题同作者作为正例，并为每类学习多个风格原型。按公开 Native 的 prompt 五折评估，独立 balanced accuracy 是 **0.3679**，加入 Native 后全量只从 0.6813 到 0.6820。每题两条的两次抽样分别从 0.6451 到 0.6503、从 0.6605 到 0.6479，总体方向不一致。短文本在两次抽样中提高约 0.23 和 0.45 个百分点，幅度仍小。当前结果不足以把这一读出加入集成。[Kaggle Notebook](https://www.kaggle.com/code/zhuating/ntl-prompt-contrastive-styledistance) · [指标 JSON](../runs/style_readout_20261121/v3/style_results.json) · [同题控制的风格研究](https://aclanthology.org/2022.repl4nlp-1.26/)

## 对占集成 40% 的 Native 读出重新训练

8B 续训从公开 Native 的五个折权重分别开始，保留原八层末 token 表示，更新最后八层 LoRA 和分类层。目标包括 18 类交叉熵、同题减均值后的交叉熵、易混类别内部的交叉熵，并随机保留部分短片段。外部样本有 29,330 条、3,316 个题组，其中各 900 个题组用于比较同一路由的 Gemini 3.7/3.8 和 GPT-5.6 三个型号。样本选择去掉公共池标记的官方题目再生成及近似重生成文本。训练只使用本折允许的官方题目，验证同时看单条读出和每题两条的稀疏条件。[Native 8B 续训](https://www.kaggle.com/code/zhuating/ntl-native8-sibling-adaptation) · [Gemini 同路由数据](https://www.kaggle.com/datasets/davidpiln/ntl-vertex-gemini-gen) · [Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool)

4B 版本从 Qwen3-Embedding-4B 开始训练读出，兼顾同题差分和易混类别，之后做官方适配。外部训练用 5,688 条回答。4B 和 8B 都使用 512 token 推理分窗和按 token 数加权。[Native 4B](https://www.kaggle.com/code/zhuating/ntl-native4-style-readout) · [Qwen3-Embedding 模型](https://www.kaggle.com/models/qwen-lm/qwen-3-embedding)

8B 五折折外单条读出从 **0.6516 降到 0.6305**，五折中四折下降。不足 100 词的文本从 **0.5714 降到 0.5471**。每题抽两条时整体从 **0.6490 降到 0.6435**，该抽样中不足 100 词的文本从 0.5364 到 0.5732。两种采样下的总体成绩都下降，原版 Native 保留用于提交。[Native 8B 作业](https://www.kaggle.com/code/zhuating/ntl-native8-sibling-adaptation) · [指标 JSON](../runs/native8_style_20261121/native_adapt_metrics.json)

4B 已完成五折。单条 balanced accuracy 为 **0.3571**，100 词内为 **0.3224**。每题两条的两次抽样中，4B 与 8B 等权融合分别从 **0.6451 降到 0.6270**、从 **0.6605 降到 0.6510**。这组训练不支持以 4B 替换 Native 或等权叠加。[Native 4B 作业](https://www.kaggle.com/code/zhuating/ntl-native4-style-readout) · [指标 JSON](../runs/native4_style_20261121/v2/native4b.json)

## 候选作者的条件似然

在 Native 的 GLM 修正后折外概率中，首选命中率为 **0.6872**，前三个标签的类均衡覆盖率为 **0.9259**。按第二、第三候选的概率和除以预计 token 开销的平方根排序，只覆盖每折四分之一的文本，其中 **308 条**为首选错误而真值仍在前三。若把这 308 条全部改对、其余预测保持原样，balanced accuracy 上限可提高 **11.41 个百分点**。同覆盖率下，按熵排序找到 274 条，按第一与第二的概率差排序找到 298 条。可纠正样本中，Gemini 3.7/3.8 的双向混淆有 50 条，GPT-5.6 三个型号内部也反复出现。[Kaggle 选样对照](https://www.kaggle.com/code/zhuating/ntl-source-likelihood-routing) · [指标 JSON](../runs/source_routing_20261121/source_routing.json)

用 Qwen3-1.7B 训练作者条件似然，并在每折优先比较约 25% 文本的前三个候选。折外 balanced accuracy 从 **0.6872** 随融合权重 0.25、0.50、1.00 依次降到 **0.6857、0.6831、0.6791**。不足 70 词的文本在三个融合权重下均下降。在最低权重时，GPT-5.5 召回从 0.680 到 0.700，GPT-5.6-sol 从 0.600 降到 0.560。候选范围有改判空间，这次似然模型给出的相对分数不足以利用它。[Kaggle 条件似然训练](https://www.kaggle.com/code/zhuating/ntl-source-conditioned-likelihood) · [指标 JSON](../runs/source_likelihood_20261121/source_likelihood_metrics.json)

对选中的 674 条回答，统计原首选与两个候选之间包含真类的 878 个成对比较。原概率差 AUC **0.6197**，条件似然差 **0.4676**，Gemini 和 GPT 的族内分别为 **0.4633**、**0.4985**。用同一折中一半 prompt 的双错候选估计作者偏置、再交换评估另一半，AUC 为 **0.4640**，融合 BA 为 **0.6839**。这种小模型条件似然在目标混淆上没有稳定区分能力，作者偏置校正也没有恢复它，停止当前做法。[Kaggle 诊断](https://www.kaggle.com/code/zhuating/ntl-source-likelihood-readout) · [指标 JSON](../runs/source_likelihood_20261121/readout/source_likelihood_readout.json)

同路由风格配对各用 900 个 Gemini Flash 和 GPT-5.6 题组。用标点、句首尾功能词和归一化词形拟合同题相对偏好，在首选属于该组且前三中至少有两个同组候选时重排。外部样本训练的组合特征在 0.35 权重下从 **0.6872 到 0.6894**，修正 27 条、改错 21 条，五折有两折下降。Gemini 3.7/3.8 的平均召回增加 3.33 个百分点，GPT-5.6 内降低 0.89 个百分点。规则提供了一些 Gemini 方向的信号，总体增益有限。[Kaggle 对照](https://www.kaggle.com/code/zhuating/ntl-sibling-style-pairs) · [指标 JSON](../runs/paired_sibling_style_20261121/paired_sibling_style_metrics.json)

后续比较冻结 Native 8B 的两类表征。第一类沿用原八层的末 token 读出，第二类取第 6、18、30 层正文三段表示的变化。两者都用外部同路由回答的同题差向量训练成对分类头，保留官方题目五折切分与同一 Native＋GLM 折外分数。预测时只重排前三中同组的候选，固定权重为 0.15。[Kaggle 对照](https://www.kaggle.com/code/zhuating/ntl-frozen-native-sibling-heads)

分段轨迹把 balanced accuracy 从 **0.6872 提至 0.6913**，改对 25 条、改错 14 条，五折四升一降。GPT-5.6 三个型号的平均召回从 **0.6200 到 0.6400**，改对 11 条、改错 2 条。Gemini Flash 两个型号从 0.5600 到 0.5667，改对 14 条、改错 12 条。100 词以下的回答提高 0.55 个百分点，较长回答提高 0.30 个百分点。末 token 方案为 0.6857。分段轨迹的收益主要来自 GPT-5.6，可在 Native 已有前向中提取这三层的表示。这里沿用原 Native＋GLM 的折外分数，混合集成的隐藏榜单效果仍待评分。[五折指标 JSON](../runs/native_frozen_siblings_20261121/output/native_frozen_siblings.json)

用于榜单单变量对照的候选以 estopadilla v5 为上游，沿用它的 GLM 选择和字体判决。五折 Native 均完成时，平均五个 GPT-5.6 判别头的相对分数。仅当前首选属于 GPT-5.6 且前三至少有两个同组候选时，才在这些候选间以 0.15 权重重新分配组内总概率。Gemini 轨迹留作单独的实验变量。与 v5 原样提交的分差可用来评估这一项改动。[公开 v5](https://www.kaggle.com/code/estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix?scriptVersionId=371376285) · [候选生成器](../src/ntllm/native_trajectory_deploy.py)

## 类别比例变化与 Sinkhorn

0.711 版本将 Native 和集成按折融合，再做 Sinkhorn，最后平均五折结果。GLM 候选排序和温度都使用平衡后的分数。这里固定上游，仅在 Native 单成员的折外原分数和 GLM 更新后分数上重算同一规则。每折从真实留出题目按类别分别抽样，轮换哪些类别较少，类别标签只用来抽样和计算均衡准确率。把类别比例偏移与条件文本分布偏移区分开后，如果训练后验已校准，在纯类别比例偏移下，目标后验的先验修正与均衡准确率的目标先验除权正好抵消。[0.711 v21](https://www.kaggle.com/code/areebbb/0-711-fingerprints-sibling-specialis?scriptVersionId=371951077) · [类别比例偏移研究](https://proceedings.mlr.press/v80/lipton18a.html) · [后验校准与类别比例估计](https://proceedings.neurips.cc/paper/2020/hash/219e052492f4008818b8adb6366c7ed6-Abstract.html)

| 样本类别比例 | 公开 Sinkhorn，GLM 前 | 公开 Sinkhorn，GLM 后 | 限幅 Sinkhorn，GLM 后 |
|---|---:|---:|---:|
| 全量 | 0.00 | -0.04 | -0.04 |
| 中度偏斜 | +0.04 | +0.06 | +0.08 |
| 重度偏斜 | +0.07 | +0.09 | +0.05 |
| 缺三类 | -0.29 | -0.34 | -0.14 |

表中为相对原判决的 balanced accuracy 变化，单位为百分点。训练先验接近均匀，直接修正未造成改判。公开规则在所有类别仍存在时平均变化小于 0.1 个百分点。缺类时，GLM 后分数误分到缺失类的回答平均从 69.6 增至 78.6，每批约 1,230 条。当前结果不足以在完整集成中改动 Sinkhorn。[Kaggle 对照](https://www.kaggle.com/code/zhuating/ntl-class-prior-transfer) · [指标 JSON](../runs/class_prior_transfer_20261121/class_prior_transfer.json)

随后用同样的切分和类别比例比较 Riad Mohamed 公开方案的末段校正。按 `−0.40 × log(预测类别边缘)` 调整 logits，保留已判为 GLM 的回答。应用到 GLM 更新后的全量折外预测，BA 从 **0.6872 到 0.6876**，改对 6 条、改错 5 条，五折有两折上升、两折下降、一折持平。中度和重度偏斜的平均变化分别为 **+0.02** 和 **+0.15 个百分点**，缺三类时平均下降 **0.73 个百分点**。这项调整在比例变化下没有稳定收益。[公开方案](https://www.kaggle.com/code/riadmohamed42/name-that-llm-apex-320-glm-supreme-sovereign-gold) · [Kaggle 对照 v2](https://www.kaggle.com/code/zhuating/ntl-class-prior-transfer/versions/2) · [指标 JSON](../runs/class_prior_transfer_20261121/v2/class_prior_transfer.json)

## Kimi-K3 的量化似然

Moonshot 发布的 Kimi-K3 对应比赛中的 `kimi-k3`，共有 2.8 万亿参数、每 token 激活 1,040 亿。Kaggle 的 23 个 UD-Q1_0 GGUF 分段覆盖全部 93 层，GGUF 文件原始大小约 466.5 GB。分段计算要保留每个 token 的 Attention Residuals，后续分段的状态宽度会增长。[Moonshot 模型卡](https://huggingface.co/moonshotai/Kimi-K3) · [分段 A](https://www.kaggle.com/datasets/xerneas/kimik3-q1-ntl-segs-a) · [分段 B](https://www.kaggle.com/datasets/xerneas/kimik3-q1-ntl-segs-b) · [分段 C](https://www.kaggle.com/datasets/xerneas/kimik3-q1-ntl-segs-c)

Native＋GLM 的折外分数识别出 150 条 Kimi 中的 86 条。若修正剩余 64 条且其他类别不受影响，总 BA 上限可增 2.37 个百分点。按全局不确定性挑选文本会漏掉大部分 Kimi 错误，因此对照只比较 Kimi 进入前三、与最强非 Kimi 候选分差较小的官方留出回答。每条最多读 128 token，比较相同段落的 token 对数概率、熵和排名。校准只用同一折一半 prompt，预测另一半，再交换。保留每段的权重读取和计算时间，以判断能否与已验证的 GLM 似然同时使用。[Kaggle 对照](https://www.kaggle.com/code/zhuating/ntl-kimi-real-likelihood) · [较早的 Kimi 混淆分析](kimi_k3_diagnostics.md)

Kimi K3 的公开编码器把助手推理和可见正文放在 `think` 与 `response` 两个通道。可在同一批正文 token 上比较单空格前缀与 `Write one paragraph.`、空 `think`、打开 `response` 的组合。这个最短对照包含 13 个结构标记与 15 个普通文本片段，显式加 `low` 的全局消息后是 18 个结构标记与 26 个普通片段。片段数与 tokenizer 的 token 数不同，准确开销需在同一 GGUF 词表上测量。官方 API 的 `low` 仍生成推理，空 `think` 是实验条件。先测单空格前缀的似然和运行时间，再判断是否有预算比较聊天格式。[官方编码器](https://huggingface.co/moonshotai/Kimi-K3/blob/main/encoding_k3.py) · [K3 技术报告](https://arxiv.org/abs/2607.24653) · [官方 API 配置](https://huggingface.co/moonshotai/Kimi-K3#6-model-usage)

在 2,699 条留出回答中，有 602 条把 Kimi 排进前三。按 Kimi 与最强其他候选的绝对分差选样，比较仅在这两个候选间改判的理论上限。

| 选中回答 | 估计 token 数 | 可改判为 Kimi | 可改判为其他模型 | BA 理论增幅 |
|---:|---:|---:|---:|---:|
| 75 | 约 8,012 | 10 | 6 | +0.59 个百分点 |
| 125 | 约 12,998 | 16 | 8 | +0.89 个百分点 |
| 200 | 约 20,993 | 27 | 14 | +1.52 个百分点 |
| 360 | 约 38,713 | 33 | 17 | +1.85 个百分点 |

200 条已覆盖 360 条方案可纠正的 50 条错误中的 41 条，估计 token 数约为其 54%。125 条方案中，Kimi 与 GPT-5.5 的双向混淆占 24 条可纠正错误中的 11 条。词数换算得到的 token 数只用于比较开销，增幅是所有可改判行都改对时的上限，实际收益仍取决于量化 Kimi 似然。[Kaggle 选样对照](https://www.kaggle.com/code/zhuating/ntl-kimi-route-budget) · [指标 JSON](../runs/kimi_route_budget_20261121/kimi_route_budget.json)

单空格前缀的双 T4 实验评分了 **360 条回答、37,183 个正文 token**。23 段总用时 **55 分 29 秒**，其中权重加载约 17 分钟、计算约 38 分钟，另有一次编译。入选行的 Native 间隔 AUC 为 0.7952，独立 Kimi 似然特征的折外 AUC 为 0.4792。加入 0.25 或 0.50 权重，总体 balanced accuracy 分别从 **0.68721** 到 0.68499 和 0.68536。[Kimi GPU 对照](https://www.kaggle.com/code/zhuating/ntl-kimi-real-likelihood/versions/3) · [指标 JSON](../runs/kimi_real_likelihood_20261121/v3/kimi_real_likelihood_metrics.json)

用同一批实际轨迹，按相同候选排序比较 125、200、360 条回答以及 64、128 token 窗口。200 条、64 token、0.25 权重的最高结果是 **0.68758**，只净改对一条，五折两升、两平、一降，不足 100 词的回答提高 0.20 个百分点。其余窗口和候选数组合均低于原判决。直接使用单空格似然暂未显示稳定收益。配对实验沿用同一批回答和正文 token，比对官方格式的空 `think` 加 `response`。挂载的 GGUF 词表已确认四个结构标记，最短正文前缀为 31 token，单空格为 1 token。[Kaggle CPU 窗口对照](https://www.kaggle.com/code/zhuating/ntl-kimi-window-tradeoff/versions/1) · [指标 JSON](../runs/kimi_window_tradeoff_20261121/v1/kimi_window_tradeoff.json) · [Kimi 前缀配对实验](https://www.kaggle.com/code/zhuating/ntl-kimi-response-likelihood/versions/1)

31-token 前缀在同一批 37,183 个正文 token 上用时 **64 分 25 秒**，比单空格多 8 分 56 秒。两种绝对似然特征的折外 AUC 分别为 0.4792（单空格）和 0.4415（正文通道）。在逐 token 的对数似然差上取均值、标准差与首末 32 token 的变化，AUC 提高到 **0.6024**。该差分以同一篇正文作参照，检验改变上下文对正文概率的影响。0.50 权重下，合并 Native 的 AUC 从 0.7952 到 0.8051，balanced accuracy 从 **0.68721 到 0.68795**，改对 5 条、改错 3 条，五折三升、一降、一平。两次独立评分合计约两小时，现有榜单候选仍采用 GLM 与冻结 Native 的组合。[Kaggle 配对结果](https://www.kaggle.com/code/zhuating/ntl-kimi-prefix-pair/versions/1) · [指标 JSON](../runs/kimi_prefix_pair_20261121/v1/kimi_prefix_pair_metrics.json) · [生成式似然比方法](https://proceedings.neurips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html)

将校准训练扩大到其余四个 prompt 折，每次用 288 条拟合、72 条评估。沿用同一批样本、三个特征组合和融合权重，单空格、正文通道、差分的独立 AUC 分别为 0.4942、0.5325、**0.6137**。差分以 0.50 权重合并时，balanced accuracy 从 **0.68721 到 0.68832**，Kimi 召回从 57.33% 到 60.67%，改对 6 条、改错 3 条，五折两升、两平、一降。完整用时与较小的整体增益使其暂不适合占用榜单推理预算。[Kaggle 校准复核](https://www.kaggle.com/code/zhuating/ntl-kimi-prefix-pair/versions/2) · [指标 JSON](../runs/kimi_prefix_pair_20261121/v2/kimi_prefix_pair_metrics.json)

同一新前缀折外对照中，Qwen3.8-Max 召回为 **121/150**，Kimi 为 **86/150**。若只纠正各自漏判，总 balanced accuracy 的理论增幅分别为 **1.07** 和 **2.37** 个百分点。Qwen 同源公开量化权重在 Kaggle 的五组镜像共 693.14 GB，普通 GGUF 分片只覆盖 16 片中的前 15 片，缺末片约 37.5 GB。按层执行还需重新打包并处理 Gated DeltaNet 和注意力状态衔接。结合可修正量和准备成本，先测 Kimi。[五折指标 JSON](../runs/glm_user_prefix_20261121/glm_user_prefix_metrics.json) · [Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) · [Kaggle 镜像说明](https://www.kaggle.com/datasets/ggayoayogg/qwen38-24t-ud-iq2xs-5) · [Unsloth 文件表](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/main/UD-IQ2_XS)

### 评分时间的来源

公开 0.701 同系集成在 597 条真实回答的运行中，Qwen3-14B 的固定训练前向与高斯拟合约 453 秒，Qwen3-8B 对应部分约 130 秒，GSQ 测试阶段用时约 541 秒。这是公开运行的耗时，0.711 私测规模仍需单独测量。两个高斯头的拟合只使用官方训练集，可以移到评分前。隐藏测试的表征、聚类和组内中心化仍在评分中计算。[597 条运行日志](https://www.kaggle.com/code/tianyialexlyu/ntl-0701-rehearsal/log) · [575 条运行日志](https://www.kaggle.com/code/yujiehe0925/ntl-0701-rehearsal-v1cover-s11/log)

公开资产已有按早期 v27 代码拟合的同名高斯头，部分前向配置与 0.711 有差别。0.711 v21 与 0.716 v5 的 q14v5、q8v5 前向和高斯拟合代码一致。两者都按同一顺序合并及撤销 FP16 适配器、以同一官方训练集和题目组拟合，并用相同的 WPCA 维度和协方差配置。按 v21 精确计算的参数可以用于两版，前提是对应成员的权重与设备布局一致。测试中出现 BF16 重算或 q8v5 退回单副本时，按原路径重算该成员的训练前向和拟合。[已有高斯参数](https://www.kaggle.com/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets) · [预计算实现](../src/ntllm/prefit_gauss.py)

在 Kaggle 双 T4 上用 2,699 条官方回答实际预拟合。Q14 训练前向 **484.9 秒**、WPCA 和高斯拟合 **9.8 秒**，没有 BF16 重算。Q8 的双副本加载因显存不足退回跨卡单副本，训练前向 256.7 秒、拟合 10.6 秒。0.716 预计算候选只加载 Q14 的参数。**8 分 15 秒**是本次移出的训练计算实测值，评分中的实际节省还取决于权重路径和测试时是否触发重算。[Kaggle 预拟合日志](https://www.kaggle.com/code/zhuating/ntl-l24-gaussian-heads/versions/1/log)

从两次公开运行分别扣除 GSQ 的 110 秒和 69 秒权重准备，将其余耗时按 4,000 条线性外推，含权重准备约为 50–53 分钟。正文长度与挂载速度会改变实际结果。GSQ 在 0.711 占 3% 权重，去掉它的准确率代价需要通过同配置对照测量。Kimi 分段的读取与计算耗时确定后，再比较是否值得用这段时间替换 GSQ。[597 条运行日志](https://www.kaggle.com/code/tianyialexlyu/ntl-0701-rehearsal/log) · [575 条运行日志](https://www.kaggle.com/code/yujiehe0925/ntl-0701-rehearsal-v1cover-s11/log)

两份公开方案在当前分段的 GPU 前向期间预读下一段到系统页缓存。Flash 与 Full 最大相邻两段分别合计 **41.62 GiB** 和 **28.80 GiB**。GLM 作者在约 29 GB 内存的双 T4 上观察到单段加载加快，同时进程间等待约 60 秒，端到端时间没有下降。0.716 的单批动态裁尾支持该预取，但现有公开运行未量化这一组合的完整耗时，混合版沿用 0.716 的默认设置。[预取实现](https://www.kaggle.com/code/lbjyang/ntl-700-fixed160-prefetch) · [320-token 公开变体](https://www.kaggle.com/code/mathleteutd/name-that-llm-320-token-glm-fable-typography) · [GLM 作者计时](https://www.kaggle.com/code/estopadilla/0-689-real-glm-5-3-likelihood-on-2xt4)

两次公开运行中，Comp 先完成 GLM4-9B 表征与分类头，随后按模型类型和 hidden size 选择 Qwen3-8B。排序首个匹配是只包含配置的 `lora24/final/base_config`，加载失败后整个 15% 的 Comp 项被跳过，已耗时 **1,711** 和 **1,729 秒**。0.716 作者记录的 1,200 条公开运行也有该问题。另一个版本按实际权重文件筛选路径，504 条运行完成 Comp，用时 3,614 秒。前两次结果估计了在同样缺权重条件下可回收的时间，成功运行只验证了该版本的推理可用性。[597 条失败日志](https://www.kaggle.com/code/tianyialexlyu/ntl-0701-rehearsal/log) · [575 条失败日志](https://www.kaggle.com/code/yujiehe0925/ntl-0701-rehearsal-v1cover-s11/log) · [504 条成功日志](https://www.kaggle.com/code/sigmatango/ntl-pub696-timing/log) · [0.716 作者记录](https://www.kaggle.com/code/estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix)

独立的 Comp 修复候选沿用 504 条公开对照的模型选择条件，针对按类型匹配的路径要求实际权重并排除 embedding 与 GGUF。保持 0.716 v5 的 15% Comp 权重与下游 GLM 时限。候选的榜单对照检验新增 Comp 流水线的准确率收益，以及它占用的时间对 GLM 覆盖的净影响。[候选生成器](../src/ntllm/comp_checkpoint_notebook.py)
