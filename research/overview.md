# Name That LLM 研究概况

任务是从回答正文识别 18 个候选模型。官方训练集有 2,699 篇、150 题。按题分折，以 balanced accuracy（BA）为主，同时记录每类召回、短文和逐条改判。短文指不足 100 词的正文。

## 公开榜与本队成绩

11 月 26 日公开代码的最高关联分为 **0.725**，Rohan v12 与 Dennis 并列。Rohan 在 PB115、40% GLM 行覆盖上限和联合排版的底座恢复 Sibling Odds，族内调整权重 0.55、分差门槛 0.8。11 月 25 日最后留存的榜首为 **0.813**。 [公开代码排序](https://www.kaggle.com/competitions/name-that-llm/code?sortBy=scoreDescending) · [榜单](https://www.kaggle.com/competitions/name-that-llm/leaderboard) · [Rohan v12](https://www.kaggle.com/code/mathleteutd/name-that-llm-pb115-coverage-sibling-odds?scriptVersionId=373953199)

| 本队提交 | 本地存档的官方结果 |
| --- | --- |
| zayd Sibling Odds v1，58816493 | **0.721**，本队已评分最高。 [回执](../runs/current_frontier_20261124/official_comparison.json) |
| RanJeet v1（58768811）与 Gemini 破折号修复（58813845） | 均为 **0.720**。 [回执](../runs/current_frontier_20261124/official_comparison.json) |
| Dennis Guarded Fable v1（58754662）与 ATLAS（58768879） | 均为 **0.718**。 [回执](../runs/atlas_0719_20261123/official_comparison.json) |
| LumiBee Kiro v66，58865511 | 11 月 25 日 17:55 最后核对为待评分。 [回执](../runs/kiro_0722_20261125/official_comparison.json) |
| Rohan v12，58919923 | 11 月 26 日最新本地回执为待评分。直接提交公开固定版本。 [回执](../runs/rohan_0725_20261126/submission.json) |

## 同底座证据

C2 有 300 篇，其中短文 153 篇，缺两种 GLM 和 Fable 真值。独立评估使用 GEN003 的 191 篇 Claude、三源 Kimi／Flash 的 191 篇短文，以及 Prvsiyan 895 篇全类留源和其中 316 篇严格留题。已排查可见训练正文重合，公开底座各检查点的完整训练经历未知。

| 改动 | C2 与独立对照 |
| --- | --- |
| 成对候选排序 | 在已评分的 Gemini 补丁之上，Prvsiyan 895 篇改对 5、改错 0，BA +0.63 个百分点。C2 短文改对 0、改错 2，严格留题短文零改判。 [逐条对照](https://www.kaggle.com/code/zhuating/ntl-official-choice-transfer/versions/1) |
| GLM 扩覆盖 | 同样的 87,409 个估计位置只能打分 587 篇，行上限从 30% 放宽至 40% 无变化。预算增约一半后的折外边际为 +0.15 个百分点，区间跨零。 [覆盖消融](https://www.kaggle.com/code/zhuating/ntl-lumibee-pb115-coverage/versions/1) |

三阶段候选并集对 C2 的 101 个原判错误覆盖 82 个，加入共同成员后覆盖 93 个，计入全部成员和新增读出后覆盖 95 个。Prvsiyan 对 168 个原判错误覆盖 164 个。候选可见性还需经过实际改判与留源检验。两份成员分数归档同目录存有阶段分数、覆盖和原判。 [候选审计](https://www.kaggle.com/code/zhuating/ntl-member-candidate-frontier/versions/1) · [历史 NPZ](../runs/claude_group_20261125/colab_input/ntl-sibling-odds-member-trace/atlas_baseline/member_scores.npz) · [留源 NPZ](../runs/claude_group_20261125/colab_input/ntl-prvsiyan-member-trace/atlas_baseline/member_scores.npz)

同一候选和 160-token 窗口的历史折外对照中，GLM 的 Low 档加段落请求比旧空用户前缀高 0.48 个百分点，推理档位和前缀一并改变。冻结 Native 的 GPT‑5.6 轨迹在三型号上净改对 9 篇，其独立线上增量待验证。 [GLM 对照](../runs/glm_user_prefix_20261121/glm_user_prefix_metrics.json) · [轨迹对照](../runs/native_frozen_siblings_20261121/output/native_frozen_siblings.json)

## 保留参照与暂停的 GPU 方向

- 保留官方原始数据、现训练输入 `data/q8_protocol`、自训 Q8 五折 OOF 和一份全量适配器。Native 的 GPT‑5.6 冻结轨迹及通用 Qwen3.5 读出可作对照。Comp 权重路径已核实，修复后的推理耗时和净收益仍待实测。 [Q8 五折](../runs/q8_official_20261113/five_fold_oof.json) · [训练入口](../src/ntllm/train_lora.py)
- **Claude 同请求组训练**。已固定 44,522 篇共同输入，其中 308 个三路组、77 个二路组。比较组分配损失与普通分类，训练及同底座评估待完成。 [输入审计](https://www.kaggle.com/code/zhuating/ntl-claude-group-input/versions/1) · [冻结输入](../runs/claude_group_20261125/colab_input/)
- **GLM 双前缀似然**。固定权重、低推理档和正文，对比任务前缀与空用户消息。校准 576 篇、严格留源 316 篇，配对评分待完成。 [冻结面板](https://www.kaggle.com/code/zhuating/ntl-glm-prefix-response-panel/versions/1)
- **DRJ 独立留源复验**。已筛出 163 篇、99 题，涵盖 Fable、GLM Full、Flash 和 Kimi。型号由发布者标注。两种冻结选择策略的真实底座逐条轨迹待评分。 [来源审计](https://www.kaggle.com/code/zhuating/ntl-focal-source-audit/versions/2)

已结题方案和停止依据见 [结题与停止原因](retired.md)。
