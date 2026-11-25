# Name That LLM 研究概况

任务是根据回答正文识别 18 个候选模型。官方训练集有 2,699 条回答、150 个 prompt。实验按 `prompt_id` 做五折，主指标为 balanced accuracy（各类召回均值），同时看类级召回和短文本。原文保留大小写、标点和排版。

## 榜单与评估

| 本队提交 | 公开榜 BA | 记录 |
|---|---:|---|
| zayd Sibling Odds v1 镜像原样复现 | **0.721** | [58816493](../runs/current_frontier_20261124/official_comparison.json) |
| RanJeet Sibling Odds v1 原样复现 | 0.720 | [58768811](../runs/public_0720_sibling_odds_20261123/submission.json) |
| RanJeet Sibling Odds v1 + Gemini 破折号修复 | 0.720 | [58813845](../runs/current_frontier_20261124/official_comparison.json) |
| Dennis Guarded Fable v1 原样复现 | 0.718 | [58754662](../runs/atlas_0719_20261123/official_comparison.json) |
| Dennis Guarded Fable v1 + ATLAS | 0.718 | [58768879](../runs/atlas_0719_20261123/official_comparison.json) |
| Mirza v5 原样复现 | 0.717 | [58692158](../runs/public_0717_mirza_20261122/submission.json) |
| Padilla v5 原样复现 | 0.716 | [58659450](../runs/public_0716_20261121/submission.json) |
| Padilla v5 加 Qwen3.5 READER 与 Fable 恢复 | 0.707 / 0.707 | [全量](../runs/l0716_reader_qwen35_20261122/competition_submission.json) · [低分差 50%](../runs/l0716_reader_qwen35_margin50_20261122/competition_submission.json) |

此前的复现为 0.668 → 0.680 → 0.684 → 0.696 → 0.702 → 0.709 → 0.711。Mirza v5 的代码和挂载集合与 Padilla v5 一致，离线 T4 镜像不同。历史折外回放中的 Native 或 Native＋GLM 分数只用于同输入对照。[Mirza v5](https://www.kaggle.com/code/mzishran/name-that-llm-glm-and-claude-fable?scriptVersionId=372138390) · [Padilla v5](https://www.kaggle.com/code/estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix?scriptVersionId=371376285)

## 保留的研究

- **同题回答的比较**。11 月 24 日，公开代码最高关联分为 0.721，榜首为 0.809。在 Sibling Odds 的 300 条 C2 回放中，用词面近邻处理型号冲突改对 3 条、改错 0 条。先恢复 Gemini 标点再应用近邻，两项相对原版合计改对 13 条、改错 0 条。词面近邻在独立 Claude 面板净增为零。Qwen3-4B 同题参照相对原版在 C2 改对 6 条、改错 10 条（BA +0.36 个百分点），在 191 条独立 Claude 数据改对 1 条、改错 2 条（BA −0.56 个百分点），相同权重也未稳定优于异题参照。此版缺乏提交依据。[实验与研究依据](peer_context_20261124.md)
- **PROFILER 多位置打分**。双 T4 固定同一 Qwen3-4B 和训练数据。C1 选出的下一词权重是 0，多位置权重是 0.35。多位置融合相对原版 Sibling Odds 在 C2 短文改对 2 条、改错 8 条，独立 Claude 短文改对 1 条、改错 2 条。独立短文在全部非零权重下均回落。[实验与复算](profiler_20261124.md)

- **GLM 真实似然**。同一 Native 折外输入、同一批候选和 160-token 窗口，低推理力度加段落请求的组合比旧空用户前缀提高 0.48 个百分点。同等 token 预算下，低前缀 160 token 为 0.6968，旧 320 token 为 0.6879。概率总量排序、两路都得分才融合及按剩余时间裁尾仍是基线组成。两个版本同时改变了请求与推理力度。[输入对照](../runs/glm_user_prefix_20261121/glm_user_prefix_metrics.json) · [窗口对照](../runs/glm_prefix_window_20261121/v2/glm_prefix_window_metrics.json)
- **冻结 Native 的 GPT-5.6 族内轨迹**。外部同题差向量训练的第 6/18/30 层分段读出，在 Native＋GLM 上同时开启 GPT-5.6 与 Gemini 后，BA 从 0.6872 提至 0.6913，五折四升一降。GPT-5.6 三型号平均召回从 0.620 到 0.640，贡献净改对 9 条。保留仅开启 GPT-5.6 的提交候选，完整集成收益待测。[五折指标](../runs/native_frozen_siblings_20261121/output/native_frozen_siblings.json) · [生成器](../src/ntllm/native_trajectory_deploy.py)
- **推理开销**。Q14 高斯预拟合实测移出 484.9 秒训练前向和 9.8 秒拟合，适用性取决于评分时的权重与设备路径。Comp 的检查点选择修复保留独立候选，净效应仍需结合它占用的时间和 GLM 覆盖判断。[高斯预拟合](../src/ntllm/prefit_gauss.py) · [Comp 候选](../src/ntllm/comp_checkpoint_notebook.py)
- **可复用的参照**。保留自训 Q8 五折原始预测、完整协议数据和通用训练入口，也保留冻结 Qwen3.5 的单条文本特征提取、分类头和预算排序，供其他读出方法对照。[Q8 五折](../runs/q8_official_20261113/five_fold_oof.json) · [Qwen3.5 读出](../src/ntllm/reader_qwen35.py)

已停止的做法及实测依据见 [结题与停止原因](retired.md)。线上差异、误差分布和复算结果见 [2026-11-23 研究复核](status_20261123.md)。公开前沿与同题训练见 [冲榜研究](leaderboard_research_20261123.md)。完整底座逐条回放与似然复核见 [ATLAS 同底座对照](atlas_0719.md)。
