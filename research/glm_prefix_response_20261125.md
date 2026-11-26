# 同一 GLM 在任务前缀与空前缀下的似然差

公开 Kiro v66 的 PB115 对齐同一前缀下两个模型的逐 token 分数。已有 Max 空用户前缀和 Low 任务前缀也不能用来单独估计提示效应，因为推理档位和运行代码一并变化。另一个公开逐 token 缓存只有 Low 任务前缀。SPADE 在人类与模型文本二分类中发现对两种前缀的似然作差比各自单独打分更可靠，近亲模型归因需要独立验证。[Kiro v66](https://www.kaggle.com/code/lumibee/ntl-kiro-v66-pb115-hybrid-apex?scriptVersionId=373265187) · [Max 空前缀](https://www.kaggle.com/code/estopadilla/ntl-glm-likelihood-train/versions/3) · [Low 任务前缀](https://www.kaggle.com/code/estopadilla/ntl-glm-prefix-ab/versions/2) · [SPADE](https://arxiv.org/html/2611.16929v1#S5.SS3)

固定 Sibling Odds 已用的 GLM Full、Flash 权重、二进制、Low 推理档和原文。任务条件沿用带词数的单段写作请求，空条件仅清空用户消息。任务条件保留首 160 个正文 token，空条件保留首 32 个。逐条核对公共正文 token 的 ID 完全一致。两组序列一起进入同一分片流程，分别保留逐 token 分数、真实前缀开销及分片耗时。[双前缀实验构建](../src/ntllm/glm_prefix_response_notebook.py)

公开原生 GLM 复现记录指出，混批位置会影响个别文段的量化似然。因此两款模型各附加 16 个同 token 的重复序列，比较原序列与重复序列的请求批槽、逐 token 对数概率、熵和词表秩。记录实际批尺寸是否发生降档。探针从两条件的读出数据剔除，其计算位置单列。它测同一分片流程的批组成敏感度。[公开批位置消融](https://github.com/rexziywbang/name-llm-challenge/blob/main/EXPERIMENTS.md#when-likelihood-replication-and-top-1-agreement-diverge)

校准数据从官方训练集固定 32 个类别齐全的题组，共 576 篇。整题留源使用已封存的 Prvsiyan 316 篇、23 题。先比较两类同模型内的前缀响应差，以及 Flash 与 Full 响应差的差。控制头使用任务前缀 160 个正文位置，内容预算对照使用任务 128 加空前缀 32，另报告第二段前缀也计入总位置时可投入的正文长度和实际覆盖。联合采集同时保留 160 与 32 个正文位置，用于离线同样本消融。部署耗时仍需重新验证。校准按整题分折，留源不参与拟合或选择权重。[冻结数据](https://www.kaggle.com/code/zhuating/ntl-glm-prefix-response-panel/versions/1)

GLM Full、Flash、Fable 每类在 316 篇留源中有 21 篇，该子集的不足 100 词正文缺少真 Flash。报告先按类列出支持量、原判与净改对数，再看全 18 类 BA。三个目标类的条件召回仅说明在标签族已知的情况下能否区分。回放时固定底座候选、温度、Fable 规则和成功打分的行集合，按题估计统计区间。

Kaggle CPU 使用存档 798 篇重放，在相对分数中恢复出原版实际走完 GLM 双模型的 239 篇，与原报告一致。Fable 保护和末段决策逐条一致。存档三个面板中三种目标真类均为零，留源承担正向收益验证。等成本重放先扣除原 54 维头从长任务窗口缩到短窗口的分数变化，再加入双前缀头相对同窗口单前缀头的对数概率差。这样原评分保持为对照，未继续把长窗口尾部的证据留给双前缀。另报告现采 task160 对存档分数的逐条复现差异。[Kaggle 原路由复验](https://www.kaggle.com/code/zhuating/ntl-glm-prefix-route-audit/versions/1) · [回放实现](../src/ntllm/glm_prefix_response_replay.py)

## 扩大原生 GLM 覆盖的对照

单独用已公开的同前缀特征复现 Legacy54 和 Kiro PB115 在原 587 条路由上的逐条结果。保留原五个整题折、每折融合温度和 GLM 概率排序，把各折原估计位置预算分别扩大到 1.5、2 和 3 倍，再测全量覆盖。报告新增覆盖的改对、改错、短文和各真类的 BA。使用真值仅保留正确改判的一臂用于估计现有似然证据可挽回的上限。位置预算来自公开 token 数与原前缀估计，不换算为实际时长。若扩大覆盖确有收益，再评估能否用较小模型蒸馏似然证据，或把现有打分移到更高吞吐的实现。[Kaggle 覆盖曲线](https://www.kaggle.com/code/zhuating/ntl-glm-coverage-oof/versions/1) · [实现](../src/ntllm/glm_coverage_oof.py)

两种头在原路由都逐条复现 2,699 条预测。PB115 的原路由 BA 为 70.24%，覆盖 587 条，消耗 87,409 个估计位置。以下各行与该路由配对比较。[指标与逐条结果](../runs/glm_coverage_20261125/audit_remote/glm_coverage/)

| PB115 预算 | 覆盖 | 改对 | 改错 | BA 变化 | 短文 BA 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.5 倍 | 891 | 18 | 15 | +0.11 个百分点 | +0.18 个百分点 |
| 2 倍 | 1,187 | 25 | 28 | −0.11 个百分点 | −0.31 个百分点 |
| 3 倍 | 1,768 | 34 | 47 | −0.48 个百分点 | −0.45 个百分点 |
| 全量 | 2,699 | 52 | 71 | −0.70 个百分点 | −0.65 个百分点 |

1.5 倍预算已经救回扩大覆盖后所有可救回的真 GLM 样本，Full 3 条、Flash 4 条。同期其他真类合计净改错 4 条。全量 PB115 由真值挑选改判的上限为 +1.93 个百分点，实际同权重下降 0.70 个百分点。Legacy54 从 1.5 倍开始净改错，全量下降 0.70 个百分点。暂停为扩大覆盖做蒸馏或额外 GPU 提速，单独检验双前缀能否提高已选候选的正确率。

同预算分配固定原五折合计 87,409 个估计位置和 PB115 折外预测。比较原 GLM 概率优先、按每估计位置的 GLM 概率、以及从原分类的置信度、GLM 概率与长度预测每估计位置的净收益。用 Ridge 拟合改对、改错与不改变正确性的有符号目标，并按真类频数对齐 BA；用于拟合的似然结果同时排除校准行所属折与最终评估折。沿用归档 Native 首选和分数，避免给新路由改变底座。真值优先分配仅给出可回收的诊断量。有限预算的专家推迟决策是方法依据，实际运行时须另测。[预算专家推迟决策](https://arxiv.org/abs/2510.26706) · [Kaggle 同预算分配](https://www.kaggle.com/code/zhuating/ntl-glm-route-utility/versions/1)

原路由的 **2,699 篇**预测逐条复现。按 GLM 概率除以花费排序，覆盖 600 篇，相对原路由改对 **4**、改错 **2**，BA **+0.074** 个百分点，按题 95% 区间 −0.111 至 +0.260。1,322 篇短文为 2／2，BA −0.014。学习净收益后覆盖 583 篇，全量 **1／2，−0.037**，短文 **0／1，−0.069**。真值已知时可选择 209 个相对 Native 原判变正确的病例并取消其余打分，相对于原路由 **72／0，BA +2.668**，短文 **40／0，+3.023**。两条可部署路由与原路由大幅重合，没有达到真值对照揭示的收益。上游 Native 分数来自既有外折，未重新做嵌套训练。当前不替换原调度。[完整回放](../runs/glm_route_utility_20261125/audit_remote/glm_route_utility/report.json)

### 先用 Flash 打分再选 Full

用同一批 2,699 篇的原五折训练 Flash 专用似然头。每个折的估计成本上限等于原路由的配对位置数乘以两款模型各自的位置费率之和。比较把 55%、65%、75% 的成本用于扩大 Flash，只在其内从剩余预算派发 Full。Full 优先级分别用原生 GLM 边际、Flash 后三类边际熵和 Flash 首选改变三类标签后的熵。取得 Full 的样本沿用已冻结的完整 PB115 改判。位置成本没有计入模型启动和硬件吞吐变化，也不视作已节省比赛时长。[Kaggle 级联对照](https://www.kaggle.com/code/zhuating/ntl-glm-flash-cascade/versions/1)

55% 成本加 Flash 后熵的结果最好，覆盖 **870 篇 Flash、421 篇 Full**，相对原双模型配对 **19／26，BA −0.26** 个百分点，按题 95% 区间 −0.70 至 +0.22。短文 **12／14，−0.14**，GLM Full、Flash 与 Fable 三类 **4／10，−1.33**。全部预算投给 Flash 覆盖 1,573 篇，全量 BA **−3.59**。扩大 Flash 覆盖未弥补完整配对证据的损失。[逐条和报告](../runs/glm_flash_cascade_20261125/audit_remote/glm_flash_cascade/)

### 归档真实路由的尾部覆盖

在两个真实 Sibling Odds 批次里按 GLM、Flash 的原概率和排序，只保留原生已打分的前 2/3 或 3/4。798 篇历史批次实际已有 239 篇获评，895 篇独立批次已有 268 篇获评。回滚后，各臂分别恢复整个尾部、只恢复最终判为 GLM 或 Flash 的项、再允许最终判为 Fable 的项。每篇的原始概率、融合权重和末段保护逐条保持同一底座。这里模拟的是已经打分范围内的覆盖变化。[Kaggle 归档尾部回放](https://www.kaggle.com/code/zhuating/ntl-glm-late-transfer/versions/2)

两个切分中，恢复尾部的 GLM、Flash 或 Fable 判决在 C2、独立留源和严格留题均为 **0／0**，短文也一致。独立 895 篇完全恢复尾部为 **1／1**，短文恢复 1 篇 Sonnet 5；两类限制相对归档原判使短文 **0／1，BA −0.20** 个百分点。C2 的较窄切分完全恢复尾部会误伤 1 篇 Terra，较宽切分无新增正确性变化。官方折外尾部的 GLM、Flash 增益暂未在真实留源复现。由于回放中没有原始未获评的证据，无法推断覆盖由 30% 扩到 40% 的真实效果。[逐条与报告](../runs/glm_late_transfer_20261125/glm_late_transfer_remote/glm_late_transfer/)

独立留源两个尾部的最终 Full 候选仅 2 个和 1 个，Flash 候选均为零。真实 Flash 在全量有 33 篇，在严格留题有 21 篇，两组短文均未覆盖真 Flash。这次回放对 Flash 短文收益缺少观察，原有尾部的触发证据也较少。[按类支持](../runs/glm_late_transfer_20261125/glm_late_transfer_remote/glm_late_transfer/report.json)
