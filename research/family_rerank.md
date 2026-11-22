# 族内判别器与条件重排（方向 A，2026-11-16）

研究问题：现有 13-term 栈的错误里，**族内错误占 21%**（家族准确率 0.8748、精确 0.6917、族内条件 0.7908）。如果用一个只用外部池训练的文本判别器，只在「top-1 与 top-2 同族」的行上重排，能不能把 balanced accuracy 拉高 2pp 以上？

## 1. 实验设计

两个互补实验，代码在 `src/ntllm/family_rerank.py`，入口 `runs/kaggle_a_20261116/run_a.py`，Kaggle CPU 内核 27 分钟跑完。

- **A1（可迁移性，主判据）**：判别器**只用外部池**（238,353 行）训练，应用到官方 OOF 的 2,699 行。两组特征独立评测：`char_wb` 3–5 gram TF-IDF（max_features 300k）与 word 1–2 gram TF-IDF + SVD(256)。条件重排只在 top-1/top-2 同族时触发，混合式 `log P' = log P + w·(logQ − logP|族内)` 并逐行复原族内总质量，因此族间决策按构造不变。`w=0` 必须逐行等于基线。
- **A2（对照）**：用官方 OOF 的逐行标签按 prompt 分组 5 折交叉拟合同一判别器，再把族内决策**完全替换**为判别器 argmax。这不是"上界"，而是"直接替换会怎样"的对照。

数据前提先做了检查：外部池与官方 OOF 文本 **精确重合 0 行、200 字符前缀近重复 0 行**，所以不存在文本级泄漏。

## 2. A1 结果

基线 `plain_centered_gaussian` argmax：**BA 0.691660**（与历史值逐位一致，自检通过）。

| 特征 | 模式 | w | BA | ΔBA | 改变行数 |
|---|---|---:|---:|---:|---:|
| char_wb | family | **0.5** | **0.69499** | **+0.00333** | 131 |
| char_wb | family | 0.25 | 0.69277 | +0.00111 | 48 |
| char_wb | hard | 0.25 | 0.69129 | −0.00037 | 23 |
| char_wb | hard | 0.5 | 0.69129 | −0.00037 | 70 |
| word_svd | family | 0.25 | 0.68981 | −0.00185 | 34 |
| word_svd | family | 0.5 | 0.68758 | −0.00408 | 89 |
| word_svd | hard | 0.5 | 0.69018 | −0.00148 | 40 |
| char_wb | family | 1.0（全替换） | 0.57795 | −0.11371 | 762 |
| word_svd | family | 1.0 | 0.47498 | −0.21668 | 1040 |

「hard」= 只对三个难对组（glm 二分类、gemini 3.7/3.8、gpt-5.6 三分类）训练判别器。

**逐折看最好的那一行**（char_wb family w=0.5）：

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 基线 | 0.69444 | 0.66481 | 0.70926 | 0.69815 | 0.69170 |
| Δ | **−0.00741** | +0.00926 | +0.00556 | +0.00741 | +0.00185 |

4 正 1 负，但最大的一折是负的，pooled 只有 +0.33pp。**达不到 2pp 门槛，也不能算稳健。**

逐类看，增益是散点式的：`gpt-5.6-terra` +3.3pp、`gemini-3.1-pro` +3.3pp、`gpt-5.6-sol` +2.0pp，但 `gemini-3.7-flash` −6.7pp、`claude-opus-4-8` −2.7pp。也就是说判别器在修好一些行的同时改坏了另一些行——这正是"弱于栈"的表现。

## 3. A2 对照说明了根因

| 特征 | 基线 | 完全替换为判别器 | 差值 |
|---|---:|---:|---:|
| char_wb | 0.69166 | 0.46386 | **−0.22780** |
| word_svd | 0.69166 | 0.43015 | **−0.26151** |

即使**用官方 OOF 的标签交叉拟合**（信息条件远好于只用外部池），把族内决策交给词面判别器也会崩掉 23–26pp。这说明：

> **13-term 栈的族内决策远强于一个从零训练的 TF-IDF 判别器。** 栈里已经包含了 30 个 LoRA 头、native embedding 专家、GSQ、comp stack 与 Sinkhorn，词面判别器的信息量补不上它的缺口。

所以 A1 的 +0.33pp 不是"判别器带来了新信号"，而是"在极弱混合权重下给栈加了一点点噪声，恰好有 131 行的 argmax 被推向正确方向"。这和随机微调的量级一致。

## 4. 触发结构（对后续设计有用）

- top-2 同族：1,840 / 2,699 = **68.2%** 的行会触发；
- 832 行错误里 544 行（65%）的 top-2 同族——族内重排在结构上**够得着**这 544 行；
- 但判别器在族内只有约 50–60% 的准确率（A2 可见），而栈的族内条件准确率是 79%，**判别器比被替换者更差**。

这就是 A 失败的全部原因，与特征工程细节无关：`word_svd` 比 `char_wb` 更差、hard 组比 family 更差、`w=1` 全面崩塌，方向一致。

## 5. A2 组级重排：跨族错误也救不回来

族内重排碰不到跨族错误，所以另做了一组实验（`src/ntllm/group_rerank.py`，内核 `qingyu1232/ntl-a2-group-rerank`，14 分钟）：

- **组级重排**：把组定义为族的真子集或跨族并集，允许在组内重新分配概率；两种质量模式（`restore` 保组内总质量 / `free` 不保）；三种触发（top-1 在组内 / top-2 在组内 / 组质量占优）。
- **`lift` 变体**：用「组 vs 其余」的二分类分数做加性 logit 抬升，这是唯一能让组内标签突破组质量上限、真正改写跨族决策的机制。
- **单标签族门控**：`kimi-k3` 用 kimi-vs-rest 分数按 `w·logit(score) − w·logit(prior)` 加偏置（`qwen3.8-max` 作对照）。

组内错误块（kernel 复算，与混淆矩阵一致）：

| 组 | 组内错误行数 | 完美判别器 BA 上限 |
|---|---:|---:|
| `glm_claude_triple` | 191 | +7.09pp |
| `gpt_quint` | 169 | +6.26pp |
| `gemini_quad` | 153 | +5.67pp |
| `glm_pair` | 74 | +2.74pp |

实测最好结果：

| 配置 | BA | ΔBA | 触发 | 实际改写 |
|---|---:|---:|---:|---:|
| `char_wb gpt_quint｜top2_in_group｜free｜lift=0.5｜w=0.5` | 0.69610 | **+0.00444** | 953 行 | 52 行 |
| `char_wb gpt_quint｜top2｜free｜lift=0.25｜w=0.5` | 0.69499 | +0.00333 | — | — |
| 组级最优 + kimi 门控（G3 叠加） | 0.69499 | +0.00333 | — | — |
| `kimi-k3 门控` 单独（prior=balanced, w=0.5） | 0.69314 | +0.00148 | — | — |
| 全部组一起（all_groups） | 0.69240 | +0.00074 | — | — |

逐折（最优那行）：+0.19 / +1.11 / +0.93 / +0.37 / **−0.37** pp，4 正 1 负，pooled +0.44pp。

**即便如此，仍然只有 0.44pp。** 191 行的理论空间是 +7.09pp，实测只兑现了 0.44pp（6%），因为判别器在组内也只是勉强优于随机——它改写了 953 行里的 52 行，其中正负几乎抵消。

预判与实测的偏差也值得记一笔：事前认为 `glm_claude_triple` 最有机会（错误块最大），实测最好的却是 `gpt_quint`；`glm_claude_triple` 没有进入前列。这说明**错误块的大小不等于可修正性**——glm↔fable 的混淆恰恰是词面特征最分不开的。

## 6. 结论与对后续的含义

1. **方向 A 用词面特征不可行**：族内重排最好 +0.33pp，组级重排 + lift 最好 +0.44pp，两者都逐折不稳、逐类互有胜负。不要再在 TF-IDF / SVD / LR 这条线上加特征。
2. **判别器的瓶颈是表征而不是数据**：方向 C 已证明外部池→官方迁移几乎无损耗（AUC 0.767 vs 0.776），所以问题不在"数据不够多/不够像"，而在"字符与词 n-gram 看不到栈能看到的东西"。
3. **如果还要做族内判别，必须换表征**：用 Qwen3-Embedding / 栈自身的 24 层特征，而不是 TF-IDF。但这与方向 D（native 槽重训）本质是同一件事，应合并投入。
4. **跨族错误（glm ↔ claude-fable 117 行）族内重排碰不到**，已另开 A2 组级实验（`runs/kaggle_a2_20261116/`）专门验证。
5. 这条结论对提分路径的选择很关键：**便宜的条件重排（CPU、几十分钟）不是提分点**；公开侧唯一被反复验证有效、且与我们正交的增量仍然是**生成式似然**（GLM-5.3 阶段 +0.012）。

## 7. 证据文件

- `runs/kaggle_a_20261116/output/results_a.json`（完整结果，190KB）
- `runs/kaggle_a_20261116/output/ntl-a-family-rerank.log`（运行日志，1,628 秒）
- `runs/kaggle_a2_20261116/output/results_a2.json`（组级实验完整结果）
- `src/ntllm/family_rerank.py`、`src/ntllm/group_rerank.py`
- `runs/kaggle_a_20261116/run_a.py`、`runs/kaggle_a2_20261116/run_a2.py`

## 8. Claude 短片段的外部同题迁移（2026-11-22）

用独立 Claude API 数据中的 1,000 道新题、4,998 条回答，按题拟合五个 Claude 型号之间的十组成对判别头，每对有 998–1,000 个题组。特征为相邻功能词的身份及其间其他词数的分桶，从外部回答中间最多 80 个词提取。判别方向仅用外部样本训练，官方五折的训练侧短回答只用于估计分数中心。固定 Native 折外分数、Low 用户前缀的 160-token GLM 后验及其原融合参数。仅在不足 100 词、首选为 Claude 且前三至少有两个 Claude 候选时，以 0.15 权重在同族候选之间重分配概率。[Claude API 数据](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages) · [Kaggle CPU 对照](https://www.kaggle.com/code/zhuating/ntl-claude-anchor-pairs/versions/1)

2,699 条官方回答的 balanced accuracy 从 **0.69683 到 0.69720**。355 条短 Claude 回答从 **0.61269 到 0.61621**，改对两条、改错一条，五折一升、一降、三平。Fable 短回答召回仍为 0.4189。138 条短 Claude 错误中，53 条首选已落到 Claude 族外，只有 64 条满足重排条件且真类在前三。

外部信号在官方短回答上可区分部分型号。Opus 4.6 对 Opus 5 的 AUC 为 0.726，Fable 对 Opus 5 和 Sonnet 5 分别为 0.653 和 0.652，Fable 对 Opus 4.8 为 0.562。现有结果不足以把这项重排加入集成。[指标 JSON](../runs/claude_anchor_pairs_20261122/output/claude_anchor_pairs_metrics.json)

## 9. Gemini Flash 的句位与尾段转折（2026-11-22）

两项对照都使用 Native＋GLM 的 `family_mass_glm_posterior` 折外概率，沿用官方 2,699 条回答、150 个题目和五折。首选属于 Gemini 3.7/3.8 Flash，且前三候选同时含有两者时才重排。固定证据权重 0.15，共命中 294 行。其中 93 行原首选错误且真值是同组另一候选，全部修正的全量均衡准确率上限为 +3.44 个百分点。[折外概率来源](https://www.kaggle.com/code/zhuating/ntl-glm-likelihood-dynamics)

句位对照使用 2,237 个 Gemini 3.7/3.8 Flash 的真实 Vertex 同题组（A 组 290、C 组 1,947）。两模型回答相同的用户请求且响应回传的模型名与标签一致。比较原风格汇总、句级特征均值（DC）、以及加入按相对句位加权的第一余弦分量（AC1）。配对头仅使用外部样本拟合。句级特征包括句长、标点、词形以及句首尾的功能词。[公开配对数据](https://www.kaggle.com/datasets/davidpiln/ntl-vertex-gemini-gen) · [句位对照](https://www.kaggle.com/code/zhuating/ntl-gemini-sentence-order/versions/1) · [指标 JSON](../runs/gemini_sentence_order_20261122/output/gemini_sentence_order.json)

尾段对照复用冻结 Qwen3-8B-Base 在官方训练文本上的第 4、6、8、22 层特征。缓存只保存末 32 和末 64 个有效 token 的均值。使用两个均值和有效 token 数恢复末 32 个 token 与其前相邻片段的差，再比较单独的末 64 均值与拼接该差分的特征。每折只使用另外四折的 120 个同题 Gemini 对拟合。[公开缓存](https://www.kaggle.com/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets) · [尾段对照](https://www.kaggle.com/code/zhuating/ntl-gemini-tail-contrast/versions/1) · [指标 JSON](../runs/gemini_tail_contrast_20261122/output/gemini_tail_contrast.json)

| 证据 | 全量 BA | Gemini Flash 双类 BA | Gemini 双类 AUC | 改对 / 改错 |
|---|---:|---:|---:|---:|
| 原判决 | 0.68721 | 0.5600 | 0.6651 | 0 / 0 |
| 原风格汇总，Vertex A/C | 0.68832 | 0.5700 | 0.6728 | 5 / 2 |
| 句均 DC | 0.68795 | 0.5667 | 0.6720 | 4 / 2 |
| 句均 DC ＋ AC1 | 0.68684 | 0.5567 | 0.6714 | 1 / 2 |
| Qwen 尾段 DC | 0.68350 | 0.5267 | 0.6629 | 9 / 19 |
| Qwen 尾段 DC ＋差分 | 0.68388 | 0.5300 | 0.6669 | 6 / 15 |

尾段差分把官方同题排序准确率从 73.3% 提高到 75.3%，但实际改判四折下降、一折持平，少于 100 词和更长回答的双类 BA 分别下降 1.25 和 5.49 个百分点。原风格汇总的小幅增益全来自 3.8 Flash，两折升、一折降、两折平。先前冻结 Native 末 token 与三段差分的独立双类 AUC 分别为 0.6425、0.5755，也低于原概率差的 0.6651。当前实测支持保留原判决。[冻结 Native 对照](../runs/native_frozen_siblings_20261121/output/native_frozen_siblings.json)

## 10. 冻结 Qwen3.5 的全类来源特征（2026-11-22）

将 READER 的单条回答方法用到比赛数据。冻结 Qwen3.5-9B，从第 18 层提取回答前 384 个 token 的末 token、均值（DC）与首个零均值余弦分量（AC）。在官方四折训练 18 类逻辑回归，留出一折打分，保留相同的五个 prompt 分组。独立读出的 BA 为末 token 0.36227、DC 0.42413、DC＋AC 0.40230。对固定的 Native＋GLM 折外分数加入 0.20 的 DC＋AC 后，BA 从 **0.68721 到 0.69351**，2,699 条回答改对 67 条、改错 50 条。提取与拟合用约 378 秒，两张 T4 处理不同回答。[READER 论文](https://arxiv.org/abs/2606.10794v3) · [Kaggle 五折](https://www.kaggle.com/code/zhuating/ntl-reader-qwen3-5-attribution/versions/1) · [指标 JSON](../runs/reader_qwen35_20261122/v1/reader_qwen35_metrics.json)

进一步固定 Low 用户前缀的 160-token GLM，只在原 Apex 族内重排之后、硬字形过滤之前融合 20% 的读出分数。GLM 保留原 587 条候选及原获胜保护。Mirza v7 在字形过滤后恢复 32 条 GLM 似然阶段改判为 Fable 的回答，单独使用时改对 12 条、改错 3 条，使 Fable 召回从 35.57% 到 43.62%。[Mirza v7](https://www.kaggle.com/code/mzishran/name-that-llm-glm-and-claude-fable/versions/7) · [Fable 五折对照](https://www.kaggle.com/code/zhuating/ntl-glm-fable-v7-restore/versions/1)

| Qwen3.5 特征 | Padilla 规则 | 加 Fable 恢复 | 加恢复后不足 100 词 |
|---|---:|---:|---:|
| 无 | 0.69124 | 0.69461 | 0.60280 |
| DC | 0.69865 | 0.70165 | **0.61419** |
| DC＋AC | **0.70013** | **0.70313** | 0.61216 |

在 Fable 恢复规则下加入 DC＋AC，改对 68 条、改错 45 条，五折四升一平。相对两项改动加入前，组合高 **1.19 个百分点**。这些结果使用固定 Native 折外分数模拟集成。候选使用 DC＋AC 与 Fable 恢复，隐藏集仍需正式提交评估。[Kaggle CPU 对照](https://www.kaggle.com/code/zhuating/ntl-glm-reader-apex-ab/versions/1) · [指标 JSON](../runs/glm_reader_apex_ab_20261122/v1/glm_reader_apex_ab_metrics.json) · [组合方案](https://www.kaggle.com/code/zhuating/ntl-0716-reader-qwen35/versions/1)

运行 Qwen3.5 前，用原 Apex 的最高两类分差选择最难判的回答，排除 V1 判为 GLM 的回答和触发 Fable 恢复的行。在覆盖全部回答的 30%、50%、70% 三档中，整体 BA、五折分数和各长度段分数均与全量 Qwen3.5 的 **0.70313** 一致。50% 档的改判也是同样的 150 条，按正文字符数估计的 token 位置为全量的 **46.0%**。30% 档用量为 26.9%，少改判一条，净正确数仍相同。选择 50% 档给隐藏集的候选保留余量，实际推理耗时仍需在比赛运行中测量。[Kaggle 分差对照](https://www.kaggle.com/code/zhuating/ntl-glm-reader-margin-budget/versions/1) · [指标 JSON](../runs/glm_reader_margin_budget_20261122/v1/glm_reader_margin_budget_metrics.json) · [50% 组合方案](https://www.kaggle.com/code/zhuating/ntl-0716-reader-qwen35-margin50/versions/1)

另用公开的冻结 Qwen3-8B 末 64 个 token 均值检验主题正则化。每个训练折用同题回答均值估计共享主题方向，并在 18 类逻辑回归中增加沿这些方向变化的惩罚。留出折逐条打分。独立读出的 BA 从 0.42562 到 0.42636。以相同 0.20 权重加入 Native＋GLM 后，两版均为 **0.68869**，正则化后的短文 BA 从 0.59939 到 0.59783。该对照保留原特征。[Kaggle CPU 对照](https://www.kaggle.com/code/zhuating/ntl-prompt-invariant-readout/versions/1) · [指标 JSON](../runs/prompt_invariant_readout_20261122/v1/prompt_invariant_readout.json)

在相同的 2,699 条回答、五个 prompt 折、Low 前缀 GLM 和 Fable 恢复下，用同一 Qwen3.5 DC＋AC 特征，在每折的训练侧以同题回答之差拟合 GPT-5.6 和 Gemini 3.7/3.8 Flash 的四个成对方向。固定 0.15 权重，仅在加入 Qwen3.5 证据前的前三名中重排同族候选。成对分数单用时，BA 从 0.69461 到 0.69720（改对 23 条、改错 16 条）。叠加到已有的 0.20 全类读出后，BA 从 **0.70313 降至 0.70128**（改对 13 条、改错 18 条，五折一升一平三降）。四对分数与全类读出的相应分差相关性为 0.673–0.842，成对 AUC 均低于全类读出。保留全类方案。[Kaggle CPU 对照](https://www.kaggle.com/code/zhuating/ntl-glm-reader-pair-ab/versions/1) · [指标 JSON](../runs/glm_reader_pair_ab_20261122/v1/glm_reader_pair_ab_metrics.json)

在同一回放中固定 DC＋AC 权重为 0.20，在原 GLM 保护、字形和 Fable 恢复之前逐族补偿 logsumexp，使六个家族恢复到加入读出前的概率质量。整体 BA 从全局融合的 **0.70313 降至 0.70053**，短回答从 0.61216 降至 0.61079。按改判前后预测的家族划分，全局融合的同族改判改对 56 条、改错 40 条，跨族改对 12 条、改错 5 条。保持质量后对应为 55/40 与 3/2。全局融合对真实 Claude 和 GPT 的族内净修正分别为 11 和 9 条，跨族再净修正 3 和 1 条。真实 Gemini 的族内净减少 5 条，跨族净修正 1 条，保持质量后的 Gemini BA 仍为 0.67833。这项限制损失了大部分跨族净增益，保留全局融合。[Kaggle CPU 对照](https://www.kaggle.com/code/zhuating/ntl-glm-reader-family-mass/versions/1) · [指标 JSON](../runs/glm_reader_family_mass_20261122/v1/glm_reader_family_mass_metrics.json)

用同一次 Qwen3.5 前向的末 token 与 DC＋AC 读出，在同一低前缀和 Apex 回放中固定 0.20 总权重。DC＋AC 单用为 **0.70313**，末 token 单用为 **0.69980**，各占 0.10 时为 **0.70461**，相对纯 DC＋AC 改对 33 条、改错 29 条。等权组合五折两升、两平、一降，短文下降 0.184 个百分点，长文上升 0.553 个百分点。三个变体的全量与低分差 50% 预测逐行一致。比赛候选继续使用纯 DC＋AC。[Kaggle CPU 对照](https://www.kaggle.com/code/zhuating/ntl-glm-reader-terminal-ab/versions/1) · [指标 JSON](../runs/glm_reader_tail_ab_20261122/v1/glm_reader_tail_ab_metrics.json)

在同一次冻结 Qwen3.5-9B 前 18 层前向中提取第 8、13、18 层的 DC＋AC，每层用同一组官方 prompt 五折和原 `C=16` 拟合。原 Low160、Apex 和 Fable 恢复保持一致，READER 总权重固定为 0.20。第 18 层复算与既有存档的 2,699 条最终预测逐行相同，BA 为 **0.70313**。第 8、13 层分别为 **0.70090、0.69905**，三层 logits 等权平均为 **0.70387**，相对第 18 层改对 24 条、改错 22 条，五折两升、一平、两降，短文 BA 从 0.61216 降至 0.61123。各变体在全量与原低分差 50% 覆盖下预测逐行一致。保留第 18 层读出和原低分差 50% 覆盖。[Kaggle 双 T4 对照](https://www.kaggle.com/code/zhuating/ntl-reader-qwen35-depth-ab/versions/2) · [指标 JSON](../runs/reader_qwen35_depth_ab_20261122/v2/reader_qwen35_depth_ab_metrics.json)
