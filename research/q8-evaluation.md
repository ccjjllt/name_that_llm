# 双头、关系分支与集成评价 · 2026-11-05

本次修复将评价对象从 plain 扩展到 centered、固定分支融合、合法训练侧拟合的 Gaussian，以及候选 q8 块对固定集成的贡献。核心区别是单成员分类改善、关系分支改善与集成互补性属于不同问题，不能用单个 plain BA 替代。

当前已有真实 CPU 模型的五折结果和集成主融合前缀结果。尚未训练新 Q8，也没有完整 0.668 的可信分支缓存，因此没有新的 Qwen 成绩或完整正式集成增益结论。

## 分支与推断条件

同一个 fold checkpoint 提取训练、验证表示。plain 与 centered 使用同一次验证表示；对于线性 centered 头 `u_i = W_c z_i + b_c`，任意恢复簇的分数可精确写成 `u_i − mean_group(u) + b_c`。因此保存原始头分数与偏置后，可以在不同回答集合上重算关系分支，不必重新运行底座。独立的真实特征重算中，这一恒等式与缓存 LP 的最大差为 3.11×10⁻¹⁴。

分组明确区分两种协议。默认 `train_fold` 按项目约定仅在当前训练折正文拟合一次 TF-IDF，所有验证 cohort 复用词表和 IDF；`query_transductive` 保留原公开配方，每个 query cohort 在自身无标签正文上拟合。两者均为内容词 TF-IDF，min_df=2、max_df=0.05，平均连接、阈值 0.15、组上限 18。它们的差异是无标签表示的拟合范围，不是标签泄漏。真实 prompt 只用于外层分折、预先规定的抽样、误聚类诊断与配对重采样，不进入推断分组。训练折拟合失败不会改用 query 拟合；fit 失败或查询无特征的行会保持 singleton。

centered 与 Gaussian 在 singleton 行的加法贡献严格为零；不能把零向量经过 centered 头所得的偏置分布当作关系证据。评价分别报告 centered 的活跃行指标，以及所有行上 singleton 回退 plain 的结果。没有有效 centered 更新的头标为不可用，不能用随机头补齐完整融合。有效训练证据同时检查当前阶段与外部起点。

Gaussian 使用同一 checkpoint 的训练表示拟合标准化、128 维 within-prompt PCA、类别均值与协方差，温度固定为 3。任何验证表示或标签都不参与拟合。公开方法的小组 injective 后验与大组 repel 都是假设，不是官方数据保证；另提供同时关闭两项的敏感性视图。表示本身若来自见过验证文本的 full 权重，即使 Gaussian 只拟合训练行，也不能称为可信 OOF。

q8 块诊断采用固定尺度 `softmax(P + 0.35C + 0.22G)`，其中 P/C/G 为 log 分数、非活跃关系行贡献零。这个尺度保持 singleton 与 plain 相同；正式集成则使用 `2P + 0.70C + 0.44G` 加入其他成员后，再除以全系统分母 M。两个尺度明确区分。

每折独立评价 full、每题一条、每题 1–3 条、每题 1–6 条和预先规定的类别不均衡集合；后四种各有 seed 42、43，共九种 cohort。每个集合都重新聚类、中心化和计算 Gaussian，不能裁切 full 的关系预测；只有传导协议重新拟合各 query 词表。不同 seed 是敏感性分析，不作为新增独立样本扩大置信度。对缺类别的折或切片，BA 明确为观察类别的平均召回，缺失类别 recall 记为空；最终同场景、同协议先合并五折逐行 OOF，再计算整体指标。旧产物按其明确的词表拟合来源识别为历史协议，不能因为默认值改变而改记为训练折协议。

## 真实 CPU 双头实验

以下历史表格使用原公开配方的 `query_transductive` 分组口径。实验沿用项目的官方 GroupKFold 五折。分类器表示的词/字符 TF-IDF（各最多 12,000 维，保留大小写）、128 维 SVD 与标准化始终仅在训练侧拟合，随后拟合 plain 及同题三行 centered 逻辑回归头；这里的分类器 TF-IDF 与用于恢复 query 簇的另一个 TF-IDF 必须区分。centered 权重来自实际窗口后的类别预算，再按该线性模型的正则口径归一化。它不能替代 Qwen/LoRA 实验，也不能与不同表示、切分或分组协议下的分数直接归因比较。

| 完整回答，2,699 条 OOF | C=2 参照 | C=0.5 候选 |
|---|---:|---:|
| plain | 0.255215 | 0.255956 |
| plain+centered | 0.294484 | 0.291148 |
| plain+centered+Gaussian | 0.298558 | 0.295961 |

C=2 加 centered 时救回 243 条、损失 137 条；三分支相对 plain 救回 260 条、损失 143 条。C=0.5 只改变同一合法表示上的两头正则，三分支相对 C=2 的 ΔBA 为 −0.002598，救回 44、损失 51，按 prompt 配对重采样的描述性 95% 区间约为 [−0.0100, +0.00481]。这不支持仅凭该差值宣称候选更差或更优。

| C=2 稀疏诊断 | 行数 | plain | plain+centered | 再加 Gaussian |
|---|---:|---:|---:|---:|
| 每题一条，seed 42 | 150 | 0.235741 | 0.235741 | 0.235741 |
| 每题 1–3 条，seed 42 | 310 | 0.283407 | 0.298538 | 0.304221 |
| 每题 1–3 条，seed 43 | 320 | 0.246021 | 0.289987 | 0.285038 |
| 每题 1–6 条，seed 42 | 560 | 0.245628 | 0.274027 | 0.279891 |
| 每题 1–6 条，seed 43 | 515 | 0.240105 | 0.315212 | 0.315750 |
| 类别不均衡，seed 42 | 1,623 | 0.246316 | 0.295888 | 0.299780 |

每题一条的各折只有 30 行，历史传导式 TF-IDF 参数触发 singleton 回退，不能把没有关系收益解读为对所有 singleton 恢复方法的否定。Gaussian 在 sparse1–3 seed43 降分，表明完整回答组上的平均增益不足以支撑部署结论。40 个稀疏 fold/cohort 的关系分数都与简单裁切 full 不同；独立核验确认了训练/验证隔离、训练侧 Gaussian 统计、直接 centered 计算和 singleton 回退。[基线结果](../runs/q8_evaluation_20261105/cpu/summary.json) · [候选结果](../runs/q8_evaluation_20261105/cpu/candidate/summary.json) · [独立核验](../runs/q8_evaluation_20261105/evaluation_verification.json)

## 两种分组协议的受控对照

本轮复用完全相同的五折、cohort ID、CPU 双头分数、表示与 Gaussian 训练侧，只改变用于分组的 TF-IDF 拟合范围。所有 plain 概率保持不变，分数差来自恢复簇及其下游关系推断。相同 sparse1–3 seed42 的 310 条文本得到：

| 指标 | query_transductive | train_fold |
|---|---:|---:|
| 关系活跃行 | 268 | 199 |
| 同题配对 precision | 96.60% | 100% |
| 同题配对 recall | 92.56% | 67.91% |
| 混题行 | 13 | 0 |
| plain BA | 0.283407 | 0.283407 |
| plain+centered BA | 0.298538 | 0.286611 |
| 三分支 BA | 0.304221 | 0.292295 |

两侧该集合的零特征行都为零，活跃行下降不能简单归因为整行未知词。词表与 IDF 的拟合范围改变了相似度和连边，形成了覆盖与误配的取舍。full 三分支 BA 在训练折协议下反而从 0.298558 升至 0.303746；sparse1–3 seed43 则从 0.285038 降至 0.262632。每题一条 seed43 的训练折协议还产生了两行误合并，传导协议因小集合词表失败而保持全 singleton。默认采用训练折协议是为了与项目约定一致，不是因为本轮证明它在所有布局上更准确。

两种协议的九个场景、五个 fold 已分别重评，分组与独立计算逐行一致，三个主视图概率最大差为 2.22×10⁻¹⁶。每个训练折的词表、IDF 与拟合 ID 在九个 cohort 间保持一致；传导协议的拟合来源则是各 query cohort 自身。样本集合身份与包含协议、拟合来源的分组身份分开记录，即使两协议碰巧形成相同簇，也不会共用同一个集成缓存身份。两协议不能混为一份 OOF 汇总。[独立对照](../runs/q8_protocol_20261105/diagnosis/comparison.json) · [正式实现核对](../runs/q8_protocol_20261105/implementation_verification.json) · [训练折协议](../runs/q8_protocol_20261105/train_fold/summary.json) · [传导式对照](../runs/q8_protocol_20261105/query_transductive/summary.json)

同时修复了独立重评后的证据读取不一致：四格比较的行身份校验与计分现在使用同一份已读预测。已有历史公开预测和真实 CPU 配对结果均保持不变；本轮没有伪造尚不存在的完整四格模型结果。[读取核对](../runs/q8_protocol_20261105/reader/reader_verification.json)

## 固定集成的替换评价

对每个同样的 cohort，分别插值旧、新 q8 的 P/C/G：`L_beta=(1−beta)L_old+beta L_new`，保留权重 2、0.70、0.44。M 由该 cohort 旧系统中实际非零的 LP 分支数量决定，并在 beta=0、0.25、1 间固定。不同 cohort 可以有不同 M；不新增分支来隐式改变温度。

完整顺序为主融合 → 15% complementary stack → 当前 margin<0.32 的 ATLAS 候选及 20% 混合 → 软边际调整 → typography → V40A → pairwise。样本少于 50 时原配方 ATLAS 选择全部行；边际调整的目标为 0.65×预测边际+0.35×平滑训练先验，迭代 100 次，再以 0.30 强度混合 log 分数。所有 gate 都从每个 beta 的新上游概率重新计算。缺少新触发候选的 ATLAS 分数时，不能静默跳过后宣称完整重放。

原代码的末尾路径变量错误和 sim 跳过尾部的问题，使 0.668 不能证明尾部各步骤实际生效。因此当前评价明确区分 pre_tail 与修正路径后的 corrected_tail；缓存齐备仅意味着指定计算路径可重放，不意味着与历史隐藏执行逐值等价，也不证明所有成员的训练历史独立。

实际 CPU 参照/候选另与同训练折拟合的 char-only LR 组成主融合，固定该其他成员与 M=4：

| beta | 主融合 BA | 相对 beta=0 救回 / 损失 |
|---|---:|---:|
| 0 | 0.308563 | 0 / 0 |
| 0.25 | 0.308929 | 10 / 9 |
| 1 | 0.316709 | 61 / 39 |

候选单块三分支 BA 略低，但该主融合前缀中更高，说明成员排序不能替代互补性评价。这里没有 comp 或 ATLAS 替身，所有结果都明确停在 main 前缀，`full_system_supported=false`。真实 fold0 的尾部独立诊断与原源码 typography、V40A、pairwise gate 逐值一致，也不能将这种隔离诊断当成完整系统收益。[主融合前缀](../runs/q8_evaluation_20261105/replay/main_prefix_summary.json) · [尾部算法核对](../runs/q8_evaluation_20261105/replay/tail_isolated_diagnostic.json)

要报告正式完整集成增益，仍需同一合法评价集合的其他原始 weighted LP、comp、ATLAS 完成分数及预算信息、训练先验、各尾部基础证据，以及所有成员的训练来源。现有公开 plain logits 和末端 save_lp 不能反推出这些输入。q8 单成员合法 OOF 也不能消除其他 full 成员见过官方文本的历史；需要所有相关成员未见过的确认 prompt，或全部成员真实的折外产物。当前缺失项与可用证据已逐项记录，没有以 CPU 结果替代正式系统成绩。[配方与缓存审计](../runs/q8_evaluation_20261105/audit/recipe_and_cache_audit.json) · [缓存约定](../runs/q8_evaluation_20261105/audit/cache_contract.json)
