# Name That LLM

围绕现有 0.668 集成中的 q8v5 研究数据更新与两阶段 LoRA 训练。当前实现使用本项目明确的分折与优化配置，原作者未公开的训练细节记录在[研究方案](research/q8.md)。

代码位于 `src/ntllm/`，正式结构配置为 `configs/q8.toml`。原始数据位于 `data/`，每次分析或实验的配置、指标与预测写入 `runs/`。

本地数据分析使用已有数据，执行以下命令会生成独立的官方五折、原池参照和受控新增文本组。

```bash
uv run --group lora python -m ntllm.q8_data \
  --train data/train.csv \
  --pool data/external/nlm-matched-clean-v3/nlm_pool_v5.parquet \
  --new-pool data/external/open-pool-v14/ntl_open_pool.parquet \
  --out data/q8
```

数据准备先联合读取原池与新版的题目风险证据，再对两池一致隔离，并按清理后的 baseline 重算 mix20 配额。`data/q8/isolation.json` 记录规则与数量，`isolated_rows.csv` 保存逐行原因和原始行号。当前两组各 119,861 条，已排除原池的 38 条疑似官方题目文本；这属于保守风险控制，不代表已经确认语义泄漏。修复前的数据快照和前后比较保存在 `runs/q8_isolation_20261105/`。

默认辅助监督使用 baseline/mix20 共同文本组成的固定三行组，并单独平衡各目标类别的 centered 损失预算。所有文本每轮仍参与一次 plain CE。`--plan-only` 分析真实覆盖、同题上下文、逐类损失系数与更新预算。

```bash
uv run --group lora python -m ntllm.train_lora \
  --phase external --variant baseline --centered-weight 0 --plan-only
uv run --group lora python -m ntllm.train_lora \
  --phase external --variant mix20 --centered-weight 0 --plan-only
uv run --group lora python -m ntllm.train_lora \
  --phase external --variant baseline --centered-weight 1 --plan-only
uv run --group lora python -m ntllm.train_lora \
  --phase external --variant mix20 --centered-weight 1 --plan-only
uv run --group lora python -m ntllm.train_lora \
  --phase external --variant matched --fold 0 --plan-only
uv run --group lora python -m ntllm.train_lora \
  --phase official --fold 0 --plan-only
```

四格中的每个数据×辅助监督条件各自建立外部起点，供该条件的官方五折独立适配。`matched` 的目标统计只来自指定折的训练 prompt，需要为每个官方折分别准备外部阶段。官方阶段只在该折训练侧构造三行组。

`--centered-balance global_mass` 保持辅助总强度与默认 `class_mass` 一致，但不修正类别份额，可单独比较重平衡作用。`--centered-plan native --centered-balance none` 保留原来的二/三行与 plain 共用损失权重参照。各计划保存 `centered_loss_weights.npy`、逐类监督统计和关系上下文指纹；本次真实分析位于 `runs/q8_centered_20261105/`。

训练入口要求显式提供已有的本地模型目录 `--base`。官方适配另需 `--init` 指向对应外部阶段产物，并继承其 centered 计划、平衡方式和强度，显式冲突会报错。仅在 `--plan-only` 下，`--init` 也可引用外部阶段的已有计划，便于检查两阶段设置；实际训练仍要求保存的 checkpoint。两阶段分别建立优化器。`--max-updates` 定义部分训练，学习率日程和辅助权重都按实际预算重算；若某个活跃 epoch 的前缀没有全部 18 类的辅助支持，启用 `class_mass` 时需增加预算。未来实际模型运行的设备、时间和分数以产物为准。

完整四格各自取得五折预测后，可分解数据更新、辅助监督与交互项：

```bash
uv run --group lora python -m ntllm.compare_centered \
  --baseline-off /path/to/baseline_off/fold{0,1,2,3,4} \
  --mix20-off /path/to/mix20_off/fold{0,1,2,3,4} \
  --baseline-on /path/to/baseline_on/fold{0,1,2,3,4} \
  --mix20-on /path/to/mix20_on/fold{0,1,2,3,4}
```

该比较要求完整两阶段训练与五折 OOF，并核对实际预算、代码、精度、分折和共享关系上下文；计划或部分训练不能作为完整实验结果输入。校验和计分复用同一次读取的预测表，并记录实际路径及内容 SHA，独立重评后不会校验根文件却读取另一份结果。该四格仍以 plain 为共同指标，避免把未训练 centered 头用于 off 条件。当前没有这四格的新 Q8 分数。

官方折训练结束后会自动评价 plain、centered、两头固定融合及训练侧拟合的 Gaussian，并在 `evaluation/` 保存原始分支、恢复簇、逐行概率和指标。默认包括 full、每题一条、每题 1–3/1–6 条、类别不均衡等九个 cohort。`eval_grouping_protocol="train_fold"` 默认只在当前训练折拟合一次分组 TF-IDF，所有 cohort 复用词表和 IDF，再各自聚类、中心化。显式设为 `query_transductive` 可保留原公开配方的无标签传导式对照：每个 query cohort 自行拟合词表。两种协议分开存储和汇总，不把后者称作标签泄漏。

`configs/q8.toml` 的 `eval_gaussian` 控制 Gaussian，`eval_cluster_threshold` 和 `eval_cohort_seeds` 控制关系评价。未获有效训练的 centered 头或未拟合的 Gaussian 会标为不可用。manifest 记录协议、拟合文本/ID、词表与 IDF 摘要；样本集合签名与包含协议的分组签名分开，集成缓存不能跨协议混用。

同条件的五折结果可直接汇总，稀疏场景先合并逐行预测再计算 BA：

```bash
uv run --group lora python -m ntllm.q8_evaluation \
  --runs /path/to/run_fold{0,1,2,3,4} --output-dir runs
```

`evaluation/head_logits.npz` 保存两个头的原始 logits 与 centered bias；Gaussian 参数和查询投影也有单独缓存。每个 cohort 的 `branches.npz` 保存未加权 P/C/G 及活跃掩码，`manifest.json` 保存 ID、标签顺序、训练来源与推断条件。旧 `valid_predictions.parquet` 继续对应 full/plain。

概率比较支持按相同 ID、标签和分折对齐两个方案，并计算固定的 log-probability 混合。

```bash
uv run --group lora python -m ntllm.compare_lora \
  --baseline /path/to/baseline/run \
  --candidate /path/to/candidate/run \
  --view plain_centered_gaussian --cohort full
```

完整集成替换使用逐分支插值及缓存重放，不从末端概率猜回上游分数：

```bash
uv run --group lora python -m ntllm.q8_ensemble \
  --reference /path/to/reference/evaluation/full \
  --candidate /path/to/candidate/evaluation/full \
  --cache /path/to/ensemble_cache.json --beta 0 0.25 1 \
  --tail corrected_tail --require-complete
```

缓存需要原始其他成员 weighted LP、comp、ATLAS 及其可用掩码、训练先验和末尾重排的基础证据；具体字段见 `runs/q8_evaluation_20261105/audit/cache_contract.json`。缺失时默认仅输出合法连续前缀及缺失清单；`--require-complete` 会拒绝未覆盖请求阶段的比较。`pre_tail` 与修正路径错误后的 `corrected_tail` 明确分开，均不冒称已经复现 0.668 的隐藏执行路径。

可用 `ntllm.evaluate_cpu --grouping-protocol train_fold` 或 `--grouping-protocol query_transductive` 显式选择 CPU 评价口径。已有 `runs/q8_evaluation_20261105/cpu/` 是历史 query-transductive 结果；本轮复用其头分数与表示的对照保存在 `runs/q8_protocol_20261105/train_fold/` 和 `query_transductive/`，重评入口为同目录 `run_evaluation_contrast.py`。两者都只是 CPU 研究结果，不是 Qwen 或正式集成成绩。重评请使用独立输出目录保留对照；有当前索引却缺预测文件时，reader 会报错，只有无索引的历史 plain 根文件采用兼容读取。

本地 TF-IDF 参照仍可通过 `ntllm.baseline --config configs/tfidf.toml` 复现。

- [q8 主方案与本地数据实验](research/q8.md)
- [辅助监督均衡与受控比较](research/centered-supervision.md)
- [双头、关系分支与集成评价](research/q8-evaluation.md)
- [比赛数据与评价指标](research/competition.md)
- [已有模型实验结果](research/experiments.md)
- [公开集成消融](research/public-solutions.md)
- [比赛成绩记录](research/submissions.md)
