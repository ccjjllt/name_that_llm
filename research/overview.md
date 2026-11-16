# 研究概况与清理记录（2026-11-16）

本文件记录本项目做过哪些研究线、各自的结论与去留、以及 2026-11-16 这次磁盘清理删掉了什么、为什么删、怎么重建。清理后仓库从 **13 GB 降到 4.0 GB**，研究文档的引用没有破损。

## 1. 项目当前状态

比赛 `name-that-llm`：给定一段 prompt + 回复文本，判断 18 个候选模型中哪一个写的，指标是 balanced accuracy（各类召回的均值）。

主队公开榜最好成绩 **0.696**（submission ref 58380844，另一条独立重跑 58390481 同为 0.696），此前是 0.684（ref 58240088）。0.696 来自公开方案 `areebbb/0-696`（feiyu26 v1 的逐行镜像）的逐字复现；相对 0.684 的增量只有一个 GLM-5.3 / GLM-5.3-Flash 真实似然重打分阶段。

Q8 主线（本项目自训的 LoRA 读出）：外部 26,484 更新 + 官方五折各 192 更新 + full 部署件 300 更新已全部跑完，构成六件套。五折 pooled OOF：plain 0.64201、centered 0.68831、固定 `P+.35C+.22G` 0.69166。端到端把六件套替换进 0.684 栈的 `q8v5` 槽位，实测 public LB **0.681**（ref 58396783），相对 0.684 为 **−0.003**。

结论：成员级 OOF 的 +2.8pp 没有传导到集成；`q8v5` 只是 13 个 term 里的 3 个，稀释后有效份额约 4.3–5.7%。**该槽位不是提分点**。若要继续，方向是换槽位形状（`native` 的 40% 份额）或换信号维度（生成式似然），而不是在同一槽位加码。

## 2. 各条研究线的结论与去留

| 研究线 | 一句话结论 | 去留 |
|---|---|---|
| 公开方案复现 0.668 → 0.680 → 0.684 → 0.696 | 逐级爬升，0.696 已两次复现 | 保留提交记录，代码不再维护 |
| Q8 长日程（2,000 → 26,484 外部更新） | 六个视图全部提升 1.6–6.9pp，plain +6.89pp | **主线，保留** |
| Q8 官方五折 + full 部署件 | 六件套可用，槽位替换实测 −0.003 | **主线，保留** |
| centered 四类文本替换（关系分支） | centered 相对 plain +4.63pp，是最稳的单项；固定三分支只 +0.335pp 且区间含零 | 已定论，不再训练 |
| scale 对照（baseline / mix20 / expanded） | 同预算下 expanded 优于 baseline；该结论已被 26,484 更新版覆盖 | 已定论，删除逐行表 |
| 类别均衡与 class/global 对照 | 单臂结果已记录，缺多臂对照 | 暂停，缺预算 |
| Gaussian 排斥 vs 无排斥 | 主族应为无排斥；含排斥只作敏感性 | 已定论 |
| bagging refinement（成员增降、平滑 vs 硬选） | 降选择敏感但未证保 BA，默认不变 | 已定论，删除逐行表 |
| fusion robustness（分组、偏斜、多方向） | 方向敏感、未建立因果；默认配方不变 | 已定论，删除逐行表 |
| 可观测性（observability） | 结论未进入任何决策，后端对比被 `q8_backend_compare` 取代 | **放弃**，整目录删除 |
| live Gaussian 校验 | 结论已固化为 `q8_evaluation` 的 `gaussian_noninjective` 视图 | **放弃独立线**，整目录删除 |
| provenance / protocol repair / exposure contract | 治理与协议修复，随协议冻结结束 | 已定论，删除大表 |
| 26k 冻结计划的准备副本（scale_preparation） | 计划已固化并在 Kaggle 跑完，副本可由 `--plan-only` 重建 | **放弃**，整目录删除 |
| 上游 0.680 的 native embedding 资产副本 | 被 0.696 线取代 | **放弃**，删除 |
| 旧 LoRA 权重（lora_artifacts v3） | 不再被任何代码引用 | **放弃**，删除 |
| 协议控制数据（q8_protocol_controls） | 控制臂数据，可由 `q8_protocol_controls` 重建 | **放弃**，删除 |
| scale 线数据（data/q8_scale） | 2000 更新线的数据副本 | **放弃**，删除 |

## 3. 本次删除清单

| 对象 | 大小 | 处理 | 理由 |
|---|---|---|---|
| `runs/scale_preparation_20261107` | 1.6 G | 整目录删除，归档 119 个小 JSON/脚本 | 26k 计划准备副本，计划已跑完且可重建 |
| `runs/observability_20261106` | 379 M | 整目录删除，归档 192 个小文件 | 研究线放弃，结论未被采用 |
| `runs/gaussian_live_20261107` | 35 M | 整目录删除，归档 89 个小文件 | 结论已固化进评估代码 |
| `runs/public_0680_20261109/native_q8_0679_meta` | 905 M | 删除 | 上游资产副本，被 0.696 取代 |
| `runs/kaggle_preparation/ntllm-kaggle.zip` | 739 M | 删除 | 上传包，可由 `q8_kaggle_bundle` 重建 |
| `runs/q8_official_20261113/job` | 579 M | 删除 | 原始五折 adapter，被重跑六件套取代；OOF 预测保留 |
| `runs/q8_full_20261115/job` | 116 M | 删除 | 训练快照，部署布局 `export/` 保留 |
| 九条已完成 CPU 研究线的 parquet / npz | 2.57 G | 删除逐行表，保留全部 JSON 结论 | 结论已写入 `q8.md`，逐行表不再需要 |
| `data/q8_protocol_controls` | 701 M | 删除 | 控制臂数据副本，可重建 |
| `data/q8_scale` | 302 M | 删除 | scale 线数据副本，可重建 |
| `data/lora_artifacts` | 116 M | 删除 | 旧 LoRA 权重，无引用 |
| `.cache/uv` | 1.2 G | 删除 | 包缓存，可再生 |

净效果：**13 GB → 4.0 GB**。

因删除 `data/q8_scale/manifest.json` 而使 `q8.md` 的一处引用失效，已改用 `runs/archive_20261116/q8_scale_manifest_recomputed.json`：那是用保留的 baseline / mix20 / expanded parquet 重算的计数（119,616 / 119,616 / 238,353，新增 118,737），其中 mix20 的 23,372 无法从保留文件复现，仅作为原报告值保留。

## 4. 明确保留的内容

- **数据**：`data/q8_protocol`（训练/打包协议数据）、`data/external`（外部原始池）、`data/q8_source_full`（来源对照）、`data/q8`（旧协议表，仍被引用）、`data/tokenizers`。
- **主线产物**：`runs/q8_submission_20261115/q8v5_bundle`（部署六件套）、`runs/q8_full_20261115/export`（full 件的提交布局）、`runs/q8_official_20261113/oof` 与 `five_fold_oof.json`（五折逐行预测与汇总）、`runs/q8_full_20261115/plans`（300 更新冻结计划）。
- **提交记录**：`runs/public_0668`、`public_0680`、`public_0684`、`public_0696`、`public_scan`。
- **结论证据**：各已完成研究线保留的 `summary.json`、审计 JSON、报告 JSON。
- **归档**：`runs/archive_20261116`，401 个文件、58 M，存放被删目录里的小体积证据与被删数据集的计数重算。
- **代码**：`src/` 全部保留（1.8 M）。删除代码省不出磁盘，却会破坏文档里记录的方法与可复现性，因此一行未删。

## 5. 重建入口（模块级）

被删的计划、数据与打包产物都有对应模块可以重建，因此不视为不可恢复：计划由 `ntllm.train_lora` 的 plan-only 路径生成；Kaggle 上传包由 `ntllm.q8_kaggle_bundle` 重建；协议控制数据由 `ntllm.q8_protocol_controls` 生成；scale 数据由 `ntllm.q8_expansion` 生成；提交布局由 `ntllm.export_q8v5_layout` 导出。

## 6. 清理后校验

- 研究文档中的 `../runs/...` 与 `../data/...` 链接：**0 处破损**（唯二的例外已改指 `runs/archive_20261116/` 下的重算记录）。
- 虚拟环境仍可用（`pandas 3.1.0`、`torch 2.15.0`）。
- 仓库总计 4.0 G：`runs/` 1.2 G、`data/` 1.8 G、`.venv` 1.1 G。
