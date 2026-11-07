# Name That LLM

当前研究见 [Q8 研究报告](research/q8.md)。

常规评价与学习曲线均输出 `gaussian_with_plain_fallback`（Gaussian-only）和
`gaussian_noninjective_with_plain_fallback`（无类别排斥 Gaussian-only）。有关系的行仅使用对应 Gaussian
分数，无关系的行回退 plain；不乘融合权重，也不使用最终校准温度。两者随其他视图输出 BA、NLL、
类别与长度切片，以及 `evaluation/<cohort>/<view>_predictions.parquet`，可用于五折汇总和 `compare_lora --view`。
缺少 Gaussian 特征时不输出这两个视图；关闭无排斥评价时仅省略无排斥视图。普通评价和学习曲线都遵循
`eval_gaussian`、`eval_gaussian_noninjective`（默认开启）；四分支校准要求两者均开启。

- `src/ntllm/`：算法与训练代码。
- [configs/q8.toml](configs/q8.toml)：Q8 参数；`auxiliary` 为辅助权重对照，`source` 为固定关系的四类局部文本替换。
- `data/`：原始来源、当前训练数据与 tokenizer。
- `runs/`：当前研究的计划、模型、预测和结果。

## 融合选择与压力评价

[融合稳健性研究](research/q8.md#融合选择的样本敏感性)保留固定 plain、固定无排斥 Gaussian 与原选中融合的同模型比较。现有 CPU 的原配方重选率在每次重拟合温度后约为 26%–49%；不能用固定温度的重采样替代整个选择过程，也不能把配方变化率当作性能损失。

独立配置 [configs/fusion_robustness.toml](configs/fusion_robustness.toml) 使用 `crossfit_calibrated_inner_v3`：24 个开发 prompt 分四组，每行用于选择权重的温度由其他组拟合，随后全开发集重拟合最终温度。仍只有原来的 11 个候选。当前 CPU 五折选中结果与 v2 完全相同，重采样也没有显示一致稳定改善，因此它只是显式研究候选；原配置和冻结计划保持原协议。

新配置还启用 `eval_imbalance_suite = "rotating_v1"`，增加 20 个方向×2 强度×2 seed 的 80 个布局。旧布局不变，每个新子集重新聚类并计算关系分数。`membership` 记录实际行、类别支持与缺类；`relation_conflicts` 记录纯组、误合和真实同类碰撞，只用于评价，不能据此用真值修正推断。

`q8_crossfit_fusion.select_crossfit_calibrated_recipe` 接收已绑定的 inner evaluation、split 和训练证据，输出锁定配方及 `crossfit_selection_branches.npz`。新配方可以应用到新增压力布局，但模型与聚类／Gaussian 核心设置必须保持一致；历史配方继续按原完整评价设置核对。`q8_fusion_stability.bootstrap_selection` 分别复算固定温度、重拟合温度和交叉拟合的选择敏感性；外层输入只有分支分数，不接收外层标签。

`q8_fusion_bagging.fit_bagged_members` 与 `apply_bagged_members` 提供另一个独立研究候选：完整重拟合每个 prompt-bootstrap 成员，再对概率取固定算术均值。[400 成员实测](runs/fusion_robustness_20261107/bagging/summary.json)保留每折成员参数和全部压力布局预测；它尚未接入默认训练配方，不根据这些已观察的外层结果选择成员数或平均方式。

[同成员数的进一步研究](research/q8.md#概率平均的成员噪声与平滑配方)使用 `q8_bagging_refinement`：`make_coupled_bootstrap_bank`／`coupled_prompt_counts` 生成精确耦合抽样，`fit_member_from_counts` 用 multiplicity 拟合与展开重复行等价的温度和候选 BA，`apply_member` 同时输出硬选择、固定尺度平滑、候选等权、仅 G 温度平均。查询应用不接收标签；三种配方规则共用同一温度拟合与抽样。

本轮保存了四个独立抽样组、50–1,600 成员前缀、全部 400 对 400 删除条件和配对概率方差。平滑减小了敏感性，但同条件 BA 点估计下降，暂不替换默认融合。`runs/bagging_refinement_20261107/experiment.json` 固定研究参数，`study.py` 按折执行并保留已有产物；`summarize_stability.py` 只读取无外层标签的影响数组，质量比较单独核对相同评价行。复用已完成产物时，源内容或设定改变会被拒绝，重新定义实验应使用新目录。

已完成的 [80 布局真实研究](runs/fusion_robustness_20261107/imbalance/summary.json)与[生产评价运行](runs/fusion_robustness_20261107/integration/summary.json)均使用已有 CPU 模型。新配置经过一折真实数据的 Q8 `--plan-only` 准备，但尚未运行 Q8。正式使用时应创建新计划，并显式提供对应已完成的外部父模型，不修改旧计划的融合政策。

## 数据治理

现有数据的治理文件位于各数据目录的 `governance/`，独立于原始训练表与冻结 manifest。逐行记录保留原顺序、ID、标签和正文哈希，只关联完全相同正文的证据；身份别名、版本样式标识、来源说明与实际调用记录分别标注。未知证据不触发删除或改标签，治理统计也不作为分类特征、采样分数或损失权重。

`train_lora` 自动读取同目录治理包，官方适配继承外部阶段的绑定；训练配置、快照、常规评价、五折汇总及融合报告均保留证据范围。旧目录没有治理包时明确记为 `unavailable`；包与数据哈希不符时拒绝引用，避免把别的数据证据挂到当前实验。记录的是完整来源文件的证据覆盖，不是已完成更新的实际曝光比例。

治理包由 `ntllm.q8_governance --data-dir DATA_DIR --source-registry REGISTRY_JSON` 生成，已有包不会覆盖。来源登记保留证据文件与哈希，区分原始生成条件和聚合容器的筛选历史。更新证据时用 `--out NEW_DIRECTORY` 生成新版本，新实验通过 `train_lora --governance-dir NEW_DIRECTORY` 选择；恢复和官方适配沿用已记录版本，旧运行引用的文件应保留。当前研究结论仍限定为固定公共来源条件下的开发比较，prompt 分折不消除上游官方数据依赖。研究依据见 [治理口径](research/q8.md#现有数据的治理口径)。

## 准备实验

辅助权重对照：238,353 条扩展数据，完整两轮，每臂 26,484 步。

```bash
uv run --offline python -m ntllm.train_lora \
  --data-dir data/q8_protocol --phase external --variant expanded \
  --centered-balance class_mass --max-updates 26484 --plan-only \
  --backend-initial-state runs/scale_preparation_20261107/common_auxiliary_initialization
```

另一臂改为 `--centered-balance global_mass`，共用初始化路径。官方阶段按配置保持相同配方。

关闭 centered 的参照使用 [configs/centered_reference.toml](configs/centered_reference.toml)，明确声明 `off_off`、`off_on`、`on_off`、`on_on` 两阶段开关，以及保留的 `global_on` 权重条件。在原 `auxiliary` 上只给外部命令传 `--centered-weight 0`，官方阶段仍为开启，因此得到的是 off_on。

```bash
uv run --offline --group lora python -m ntllm.prepare_centered_reference \
  --output-dir runs/centered_reference_20261107/preparation
```

该脚本复用原来的 12 个 class/global 计划，新增 1 个关闭辅助的外部计划和 15 个官方计划，共 3 个外部、25 个官方阶段。按实际曝光文件核对文本、伙伴、顺序、物理批次和学习率日程；共同 Q8 初始张量仍需正式训练时从已有基座物化。已有汇总不会覆盖；重新准备请指定新目录。[阶段参照计划](runs/centered_reference_20261107/preparation/summary.json)只代表数据准备，不能作为训练成绩。

完成训练后，`ntllm.compare_centered_stages` 的 `--off-off`、`--off-on`、`--on-off`、`--on-on` 各接对应的 5 个官方 run，可选 `--global-on`。使用 `--output-dir` 指定新的结果目录。它比较 plain OOF、条件化阶段效应和交互项，核对真实完成状态、公共初始参数记录、父阶段、同折物理计划及内层切分；旧 `compare_centered` 仍用于历史数据×辅助的另一种四臂设计。

`relation_coverage` 中的分母、可用资格、异类伙伴及覆盖率描述物理关系机会；λ=0 时仍保留同一批次和随机数消费，但有效 centered 系数为零。融合独立分支报告会区分 C 的训练可用性与该 cohort 的活跃行；未训练 C 的纯 plain 回退值不作为 C 成绩。

固定关系的四类局部文本替换：119,616 条／臂，只有 2,955 条正文变化（2.47%），完整两轮、每臂 13,292 步。旧文本中 2,948 条（99.76%）本来就属于协议组合证据不足，另 7 条有协议记录；这项实验检验指定正文、来源和证据结构的联合替换效果，不是可靠协议对未知协议的比较。

直接覆盖 Fable、GLM 5.3、GLM Flash、Kimi，类内替换率分别为 21.37%、7.90%、7.29%、3.04%。内部 `source`、`baseline/mix20` 与原目录名保持兼容；这里的 `mix20` 不代表替换了 20%。`intervention_scope/` 保存按实际正文差异得到的范围与文件绑定；训练会记录累计实际替换曝光，短段不会套用完整两轮的 5,910 次。

报告保留 18 类总体、直接替换覆盖的 4 类、未直接替换的 14 类及 prompt 配对区间。分组只筛评价行，始终使用完整 18 维预测；后 14 类可能发生共享参数与推断关系带来的联动变化，不能当作未处理对照。分组是诊断，不改变原内层融合选择目标。

```bash
uv run --offline python -m ntllm.train_lora --experiment source \
  --data-dir data/q8_source_full/historical_anchor_vs_protocol_unknown \
  --phase external --variant baseline --max-updates 13292 --plan-only \
  --initial-state runs/scale_preparation_20261107/common_source_initialization
```

另一臂使用 `--variant mix20`。[完整计划与规模](runs/scale_preparation_20261107/summary.json)已包含两种实验及各自官方五折；token 计量使用现有本地 tokenizer。`--plan-only` 不加载模型权重。范围描述可由 `ntllm.q8_source_scope --data-dir DATA_DIR` 在新目录生成；新实验用 `--source-scope-dir` 选择其他不可覆盖版本，恢复与官方适配继承已记录版本。

正式训练使用已有基座与 `--prepared-plan`；官方适配另用 `--init` 指向本臂完成的外部模型，恢复使用 `--resume`。冻结计划和恢复读取保存参数，不再传 `--experiment`。完整参数见 `uv run python -m ntllm.train_lora --help`。

Token 计量与训练共用一次编码结果：`--plan-only` 按批记录统计，不保留全池 token 列表；正式训练仅在加载模型后，使用实际模型 tokenizer 对每个输入流编码一次、计量一次。每批仍分别获取未截断长度和实际截断 IDs，以保持原特殊 token 与截断语义。prepared 计划继续逐行核对冻结统计，恢复训练继续核对输入与 tokenizer 实现身份。根目录和 `execution_tokens/` 的计量文件来自同一次结果，不重复计算。

## 长训练保存与分段

默认每 100 次成功更新，或距上次保存满 900 秒，在先到条件后的安全更新边界保存恢复点；段结束时也会保存。`checkpoints/` 保留最近 2 个恢复点，包含模型、优化器、学习率调度器、随机状态和数据游标。预声明观察点的 `snapshots/` 单独保留，不受这两个恢复点的轮换影响。

每段默认 `segment_seconds = 10800`（3 小时），从 `run()` 入口开始计时，准备、训练、梯度诊断和中间观察都计入时限，开始最终评价前也检查同一时限。这是在安全边界停止的软限制：已经开始的单次更新、诊断、评价以及保存，都可能越过时限。如果平台有硬性会话时限，应将本段时限设得更短，为这些工作预留余量。

达到时限后先保存恢复状态，不再开始新的中间或最终评价。预声明快照的开发集评价可能标为 `pending`，恢复时会在同一更新步补做，再继续训练；最终评价若尚未开始，会记录 `final_evaluation_pending` 并暂停。已达到总更新目标时，下一段恢复只补做评价，不增加训练更新。脚本默认不会自动启动下一段，需要再次执行恢复命令。

恢复点恰好是预声明快照点时，会重做该点观察，以覆盖保存后评价中断的情况。若加载加单次观察已接近本段时限，应增大 `--segment-seconds`，避免反复停在同一点。

保存频率与分段时限位于 [configs/q8.toml](configs/q8.toml) 的 `[execution]`，独立于冻结的科学实验 `settings`，不会修改总训练预算、学习率日程或样本顺序。执行策略会保存并在恢复时继承；没有这部分记录的旧 prepared 计划和旧 checkpoint 也会获得默认保护。CLI 可以覆盖已保存策略：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--checkpoint-every` | 100 | 按成功更新数保存；`0` 关闭此频率 |
| `--checkpoint-seconds` | 900 | 按距上次保存的秒数保存，在更新边界检查；`0` 关闭此频率 |
| `--segment-seconds` | 10800 | 本段软时限；`0` 关闭时间限制 |
| `--stop-after-updates` | 无 | 本段额外执行的成功更新数上限；`0` 清除继承的步数上限 |

外部阶段的总预算始终保持来源实验的 13,292 步或辅助权重实验的 26,484 步。使用 prepared 计划启动和恢复时会读取这个原始预算，无需重传 `--max-updates`，更不能将它改成“剩余步数”。需要缩短当前段时，使用 `--stop-after-updates`；例如已完成 5,000 步，再设 `--stop-after-updates 1000`，本段最多推进至第 6,000 步，总日程不变。官方阶段同样保留各自冻结的总预算。

以下命令使用占位路径，仅供手动执行；应先替换为已有基座和已准备计划的位置，不会自动下载模型。按默认保存与分段策略启动一臂：

```bash
Q8_BASE="/path/to/existing/Qwen3-8B-Base"
Q8_PREPARED="/path/to/prepared/run"
uv run --offline python -m ntllm.train_lora \
  --base "$Q8_BASE" --prepared-plan "$Q8_PREPARED"
```

本段暂停后，从该段 `metrics.json` 的 `last_checkpoint` 字段（即 `metrics.last_checkpoint`）取得实际恢复目录。后续只用 `--resume` 恢复，不能用 `--init` 代替；`--init` 仅用于从已完成外部模型开始官方适配。

```bash
Q8_CHECKPOINT="/path/from/metrics.last_checkpoint"
uv run --offline python -m ntllm.train_lora --resume "$Q8_CHECKPOINT"
```

如需覆盖本段策略，例如最多再运行 1,000 步或 2 小时：

```bash
uv run --offline python -m ntllm.train_lora \
  --resume "$Q8_CHECKPOINT" \
  --stop-after-updates 1000 --segment-seconds 7200
```

下一次恢复会继承这些限制；可用 `--stop-after-updates 0` 清除步数上限，或显式设置新的时间限制。所有段的 run 目录、冻结计划、共同初始化及所需数据和基座都应放在持久化存储中，并保持引用路径可用，不能只保存最后一个 checkpoint 文件夹。
