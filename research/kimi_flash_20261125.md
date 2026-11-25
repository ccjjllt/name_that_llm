# 只复核 Sibling Odds 判成 GLM Flash 的短文

上一轮三类专家相对原版，在缺少真实 GLM 的重建压力集救回 3 条 Kimi，同时在包含真实 GLM 的 Arxitekton 留源净损 7 条。这一轮直接估计把原判为 GLM‑5.3‑Flash 的短文改成 Kimi‑K3 时，两类各自的得失。[上一轮结果](effort_pair_20261124.md)

从 Open Pool v18 选 696 对同来源、同题、长度比不超过 1.5 的两类短文，共 1,392 篇。过滤已知模型代理标记、重建题和来源交叉，按可识别的完整题组隔开训练和留源。正文维持原样，模型单篇读入。两份 Qwen3‑4B Base 前 20 层模型各适配末 8 层，用 rank‑8 LoRA 和相同训练集做二元交叉熵，按两个预定种子训练。专家证据为两个种子的 `Kimi logit − Flash logit` 平均值。[数据清单](../runs/kimi_flash_20261125/data/manifest.json) · [训练配置](../configs/kimi_flash.json)

主留源使用 Prvsiyan 的 59 篇短文，真实 Kimi 27 篇、Flash 32 篇，逐条具有实际返回型号记录，两类均使用各自厂商的默认推理设置。Elliot 提供 61 篇不同设置的短文，作者记录逐条校验返回型号，公开数据用题号标识题目。drjcfmd 提供 71 篇作者记录 API 生成型号的短文。191 篇均为 40–99 词，17 对同来源同题正文满足长度比限制。留源正文与完整 Open Pool v18、官方训练及原版纳入的 Prvsiyan 清洗输入均无规范化完全重合，复跑时再次对照原版读取的池和堆叠额外训练语料。第三方冻结权重的完整训练历史未知。[Prvsiyan v13](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/versions/13) · [Elliot v7](https://www.kaggle.com/datasets/elliotthankinson/ntl-exact-gens-v3/versions/7) · [drjcfmd v4](https://www.kaggle.com/datasets/drjcfmd/ntl-generated-v2/versions/4)

原版 Sibling Odds 首选为 GLM Flash 且正文不足 100 词时才复核。以产生原判的 GLM 或族内残差阶段分数为依据，取 Flash 对 Kimi 的底座非负分差。只有专家给 Kimi 的优势截断到 ±3 后乘固定权重大于底座分差，才改为 Kimi。主权重锁定为 0.35，另看 0、0.15、0.70；0 与原版逐条对照。按题重采样比较 BA 变化，同时报告真实 Kimi 救回数、真实 Flash 误伤数、改对与改错净数，并检查历史 C2 短文和已有 Kimi 压力集。[评估实现](../src/ntllm/kimi_flash_report.py)

## 留源结果

两份模型各执行 348 步，训练集、查询集和评估覆盖的哈希一致。Kaggle 复核对齐底座与专家的逐条 ID，并用原版代码复现留源预测。留源 191 篇与运行时加载的 Open Pool 231,036 篇、实际加入堆叠的 45,398 篇额外语料及官方训练正文均无规范化完全重合。[Kaggle 审计](https://www.kaggle.com/code/zhuating/ntl-kimi-flash-audit/versions/1) · [指标与逐条对照](../runs/kimi_flash_20261125/audit_remote/audit/)

| 短文来源 | 样本数 | 原判 Flash | 救回 Kimi | 误伤 Flash | 净改对 | BA 原版 → 权重 0.35 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prvsiyan 主留源 | 59 | 13 | 0 | 0 | 0 | 33.56% → 33.56% |
| Elliot | 61 | 9 | 0 | 0 | 0 | 46.30% → 46.30% |
| drjcfmd | 71 | 13 | 1 | 0 | +1 | 16.57% → 18.04% |
| 三源合计 | 191 | 35 | 1 | 0 | +1 | 31.61% → 32.09% |
| 历史 C2 | 153 | 3 | 0 | 0 | 0 | 62.18% → 62.18% |
| 重建压力集 | 160 | 5 | 0 | 0 | 0 | 40.95% → 40.95% |

三源合计 BA 提升 0.49 个百分点，按题重采样的 95% 区间为 [0, 1.56] 个百分点。权重提高到 0.70 后，三源仍只救回 1 篇 Kimi，另误伤 1 篇 Flash，BA 下降 0.08 个百分点。主留源在四个融合权重下都零改判，两个独立种子按主权重各救回 1 篇和 0 篇。34 篇长度匹配的同题留源也零改判。

底座错判的 73 篇真实 Kimi 中只有 5 篇原判 Flash。即使专家完美纠正这 5 篇且保住 30 篇已判对的 Flash，三源合计 BA 也最多提高 2.43 个百分点。主留源 19 篇 Kimi 错例中仅 1 篇在改判范围内。候选覆盖面是这一规则的主要限制，当前结果不支持正式提交。
