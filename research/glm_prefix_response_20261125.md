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
