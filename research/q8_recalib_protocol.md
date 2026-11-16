# Q8 6件套+重标定：训练前准备（本地CPU，不耗GPU配额）

## 目标
新训 Q8 LoRA 替换集成 `q8v5` 槽位：5折 OOF + full 部署件共6个适配器，换后重标定，不自动升级默认融合。

## 冻结边界
- 槽位：`keep=24, layers=[2,4,6,8,12,16,20,24], max_len=384, r16/alpha32`，7个投影（本地 `configs/q8.toml:5-10` 与提交 `q8v5_f0..f4+q8v5` 一致，已用 `load_q8_config` + `TARGET_MODULES` 验证）。
- 不动：`w=2.0/twin=0.35/gauss=0.44`、互补0.15、ATLAS 10% selective、GSQ 3%、native 40%（见提交 `CONFIG`）。
- 只重拟：四分支温度 `[0.05,20]` balanced NLL + 11候选 `P+cC+gG`（`src/ntllm/q8_fusion.py:33-46`），几何份额只诊断不改默认。

## 6件套缺口与本次准备
- 现冻结40 jobs全是 `fold 0..4/shared`，无 `final`（`runs/kaggle_preparation/kaggle_jobs.json` 实测 40，`shared:5, 0..4:各7`）。
- 新增 `[experiments.on_on_full]`（`configs/q8.toml`）：`extends=auxiliary`，仅 `research_inner_validation=false` + `fusion_protocol=inner_prompt_fusion_v1`，其余监督/LR/seed与 `on_on` 同源。
- full干跑验证（CPU `--plan-only`，输出 `/tmp`，未污染冻结目录）：
  - `training_rows=2699, validation_rows=0`，每轮900 batch（899×3+1×2），150 updates/轮，两轮300 updates
  - `actual_rows=5398, forward rows=5398, input_tokens=792430, padded=1141306`
  - `centered_rows=2697/轮`，readback断言通过（见本次会话命令输出）

## 数据治理结论
- 5折划分一致：官方2699行/150 prompt；source用120全训（240更新），aux用96 fit+24 dev（192更新），同一120的内外层切分。
- 2955替换全在 plain-only 槽，与冻结计划无冲突；`protocol_unknown 86.44%` 中 `generation_status未知 52.90%`，不可简写。
- full三件套（inner_split/exposure/token三件套/plan.npz）本次仅探针验证，正式冻结计划待生成到版本化 `runs/` 目录。

## 重标定CPU流程（GPU产出分支特征后执行）
1. 绑定验证：`split.json`（30 outer/96 fit/24 dev）+ `training_evidence` sha + `inner_dev/full/branches.npz`，禁读 `outer_test/`。
2. 温度：`fit_temperature_records` 同 `calibrated_cpu_v2` 参数，关系分支只用active行。
3. 选权：冻结温度后11候选按开发BA，平局按序；主族无排斥G，含排斥只敏感性。
4. 几何份额 `s∈{0,0.2,0.3,0.4,0.5,0.6}` 只报 `ΔBA vs 0.4` + prompt区间，不写默认。
5. 验收：T在界内且NLL下降、argmax不变、缺类披露、重选率说明（历史31–52%）、产物 `calibration.json/recipe.json/candidate_scores.json/geometric_share.json` + `summary.json`。

## 下一步（需GPU时才做）
1. 生成正式full冻结计划到 `runs/`（参照本次 `/tmp` 探针命令，改输出目录）。
2. 扩展 `q8_kaggle_bundle.py` 纳入第41 job并重跑 `--catalog-only`。
3. GPU执行外部 `centered_class` 26484 + 官方6件套（5×192 + full 300），物化 `common` 初始化后启动。
