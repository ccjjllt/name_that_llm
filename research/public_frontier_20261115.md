# 公开前沿技术地图（2026-11-15）

研究问题：公开 notebook 的 0.684 是怎么构成的；0.684 之后公开分数靠什么继续前进；其中哪一部分与我们自己的 Q8 读出正交、可整槽替换。

证据口径：**[官方]** 竞赛页面/API 读取；**[公开实测]** notebook 作者公布、且多账号或多版本重复过的榜分；**[社区自述]** 只出现在标题、评论或个人页面的分数；**[推断]** 我读代码得出的、未经作者声明的结论。所有公开榜分都是 public 部分（约 4,000 条隐藏测试的 40%，`0.001` 约等于 1–2 条 passage）。

## 0. 榜单事实

`kaggle competitions leaderboard --download`（2026-11-15）**[官方]**：5,937 支队伍，中位数 0.602。

| 阈值 | 队伍数 |
|---|---:|
| ≥ 0.774（第 1） | 1 |
| ≥ 0.75 | 12 |
| ≥ 0.72 | 46 |
| ≥ 0.705 | 103 |
| ≥ 0.696 | 217 |
| ≥ 0.684 | 938 |

公开代码前沿是 0.696，第 1 名 0.774，前 1% 约 0.705。**0.68→0.77 之间的方法没有公开**：前 5 名的账号没有任何公开 notebook（`kaggle kernels list --user` 逐个查过，要么 Not found，要么只有无关的 ARC/vLLM notebook）。公开可以被抄的天花板就是 0.696，且已停滞三天。

## 1. 本地 0.684 逐段解剖

### 1.1 执行顺序（`infer.py` 主导）

```
comp_stack.py（CPU 进程，先跑）
   └─ 写 comp_logit.npy + comp_meta.json
infer.py
   1. TF-IDF 凝聚聚类 → test 分组 Gte（阈值 0.15，组上限 18）
   2. qwen_groups.py（Qwen3-Embedding 余弦 0.70）→ 参照分组 G_q
   3. 5 个 l24 成员 + 1 个 encoder 成员：plain / twin(centred) / gauss 三项
   4. blend(LP) = 所有 term 求和 ÷ 活跃 term 数 → P
   5. comp：P ← 0.85·P + 0.15·COMP_P        （概率空间）
   6. GSQ：P ← 0.97·P + 0.03·Q27             （概率空间）
   7. native：logP ← 0.6·logP + 0.4·log p_fold，逐折 Sinkhorn 后 5 折概率平均
   8. 决策 = argmax[(1−0.30)·log P + 0.30·log Q]   Q = 向 0.65·自身边缘 + 0.35·训练先验 的 Sinkhorn 极限
```

### 1.2 三个必须先纠正的事实

**(a) `w=3.0/2.0/1.0/2.5/1.0/1.0` 不是集成权重，是每个成员的 logit 温度。** `blend()` 里 `lpn = total / max(1, active)`，即各成员 log-prob 矩阵**等权平均**（几何平均）；`spec['w']` 只缩放该成员自己的 log-softmax。`--sk_div 1` 是默认值，0.684 的运行命令没有覆盖它。所谓 "3.0×" 实际是"更尖锐→在几何平均里更占优"。

**(b) "ATLAS V9" 在这个 notebook 里是死代码。** 运行命令没有 `--atlas_mode`，默认 `off`；`atlas_infer.py` 只是被打包进来。真正干活的是 `infer.py` 内部的 `l24` 读出（它是 ATLAS 读出形状的重新实现）。0.680/0.681 那一线才用 `--atlas_w 0.10~0.20`。

**(c) 简报里的"prefit stylometric core（TF-IDF / distortion / SVC 投票）"权重是 0。** 运行命令写死 `--tfidf_w 0 --distort_w 0 --svc_w 0`，`ntl660_readout.core_votes` 只在 `FAST_COMMIT` 的 smoke test 里被调用。同理，cell 2 里从 Open Pool 抽出的 `extra_matched.parquet`（`--extra_w 0.5`）只喂给这三个已关闭的头，在 0.684 全流程里**完全不参与预测**。外部数据在 0.684 中的唯一入口是 `comp_stack.py` 自己加载的 extra passages，以及作者离线训练好的 adapter。

### 1.3 各段在做什么

| 段 | 机制 | 训练/冻结 |
|---|---|---|
| `l24` 成员（q14v5/q8v5/ll8v5/n14b_clean/q25_14） | 底座截断到前 24/27 层；取 `[2,4,6,8,12,16,20,24]` 层 hidden，**去掉第 0 个 token** 后按 mask 均值池化，逐层 LayerNorm，拼成 8×4096；两个线性头（plain / centred） | 权重与头**冻结**（作者离线训练）；每个成员 6 个 adapter（5 折 + 1 全量）**逐个原地加进权重、提特征、再减掉**，6 个头输出等权平均 |
| `twin`（centred） | 对同一 prompt 组内的 8×4096 特征减组均值后再过 `centred` 头；只对"落在 2..18 条组里"的行生效，系数 `0.35·w` | 头冻结，分组在推理时算 |
| `gauss` | 二维混合效应：`x = μ_class + P_prompt + ε`，类内协方差 + prompt 协方差分别 Ledoit-Wolf 收缩，广义特征分解白化，再算类条件高斯后验 | **推理时在官方 train 特征上现拟**（CPU，秒级），`q14v5: 0.70`、`q8v5: 0.44` |
| `n14b_clean` | 同一 Qwen3-14B 底座上叠两个 LoRA，`OQ4:0.72 + N14B:0.28`，关掉 final norm，补 eos | 冻结 |
| `deb` | DeBERTa-v3-large 编码器读出（`--adapt_w 1.0` 使其可有 centred 项），`w=1.0` | 冻结 |
| `comp_stack` | 见下 | **推理时现拟** |
| `native`（Qwen3-Embedding-8B） | 36 层全开，取 `(3,6,9,12,18,24,30,36)` 层**最后一个 token** 的 hidden（最后一层先 L2 归一），8 个可学 LayerNorm，rank-16 LoRA 打在全部 36 层×7 个投影上（`SCALE=2.0`），指令前缀 `Instruct: Identify the language model...`；长文 512 token 覆盖式滑窗并按 token 加权汇总；plain + centred 两个头，按 `plain_scale`/`centered_scale`/`selected_centered_weight` 校准 | 5 折权重全**冻结**（CHEN Han 训练），校准参数来自 `calibration/fold{k}.json` |
| `GSQ-27B` | Qwen3.8-27B 的 3-bit pack-quantized（compressed-tensors, group 128）线性注意力主干 + 120k 训练出的读出 | 冻结，3% 概率混合 |
| Sinkhorn | 100 次矩阵缩放，目标 = `0.65·自身边缘 + 0.35·训练类先验`，再与原始 log-prob 按 0.30 混合 | 无参数 |

`comp_stack.py` 的成员权重（`W`，非零项）：`stylegram 1.516`、`tokattn_deb 0.185`、`lora24_public 0.177`、`lora24_claude 0.177`、`lora20_q25_instyle 0.17`、`tokattn_q35 0.153`、`desk 0.132`、`lora24_seed7 0.113`、`ultrabert 0.095`、`stylometry_lgb 0.063`、`debv3l 0.05`、`electl 0.032`。合计约 2.85，**stylegram 一个占 53%**，而 comp_stack 整体只占最终概率 15% → stylegram ≈ 最终决策的 8%。`GATE_LAM = {claude_style: 1.5, claude_lora: 1.0, family: 0.25}` 是族内重分配门。

### 1.4 有效 term 数（推断，读代码计数）

`blend` 的分母 = 活跃 term 数。0.684 有 **13 个 term**：q14v5 3（plain+twin+gauss）、q8v5 3、ll8v5 2、n14b 2、q25_14 2、deb 1。所以 q8v5 这一个槽位 = 3/13 ≈ 23% 的几何混合，其中 plain 的尺度是 2.0（13 个里最大）。

**由此得出一个我认为关键但作者没写的结论 [推断]**：13 个 log-prob 平均之后父分支的 logit 尺度被压到单成员的 ~1/13，而 native 以 0.40 的份额混进**自身已校准（O(1~3)）**的 log 概率。也就是说 `native_share=0.40` 在 logit 空间里远不止 40%，父分支主要起尾部 tie-break 和先验作用。这正好解释 0.30→0.674、0.40→0.680、0.50→0.679 的形状：40% 已经是 native 主导、父分支刚好还能纠错的点。

## 2. 组件表

成本一列按 2×T4、9 小时上限、约 4,000 条隐藏 test 计。

| 组件 | 做什么 | 成本 | 训练/冻结 | 分数证据 |
|---|---|---|---|---|
| Qwen3-14B V5 l24 ×6 adapter | 多深度 hidden → 线性读出 | 最大单项（14B×6 次前向） | 冻结（dhruvtibarewal） | 无单独消融；父分支主体 |
| Qwen3-8B V5 l24 ×6 adapter | 同上 | ~6.7h 总栈里的主要部分 | 冻结（同作者） | 无单独消融 |
| LLaMA-3.1-8B V5 | 同上，异族骨干 | 中 | 冻结 | 无单独消融 |
| N14B clean（OQ4 0.72+N14B 0.28） | 双 LoRA 14B 读出 | 中 | 冻结（nuridudak） | "移除破坏性 v1 adapter" 是作者声明，**无 A/B** |
| Qwen2.5-14B | 异族 14B 读出 | 中 | 冻结 | 无 |
| DeBERTa-v3-large 编码器 | token 级编码读出 | 小 | 冻结 | 无 |
| **Qwen3-Embedding-8B native 5 折** | 36 层多深度 + rank16 LoRA + plain/centred 头，40% 份额 | 大（36 层全跑，双卡切分） | 冻结（concyclics） | **0.666→0.680**；份额 0.30→0.674 / 0.40→0.680 / 0.50→0.679；独立 0.679 |
| **GSQ-27B（3%）** | 3-bit 27B 特征 + 120k 读出 | 大（27B 量化前向） | 冻结（lightningv08） | 3%→0.684 / 6%→0.682 / 12%→0.681（同一代码） |
| comp_stack（15%） | 12 个预拟合稀疏/风格头 + 4 个族门 + Claude 门，stylegram 主导 | 中（CPU，现拟） | **推理时现拟**（SVC/logreg/gate） | 无 comp_w 消融公开；0.10/0.20 两版在跑，结果未公布 |
| gauss 头 | prompt 混合效应 + 类条件高斯 | 小（CPU） | **现拟** | 无 |
| Sinkhorn 0.30 / mix 0.35 | 类边缘向先验拉 | 可忽略 | 无参数 | 社区称 ±0.003；areebbb 称更高 target mix −0.002 |
| **GLM-5.3 / -Flash 真实似然（0.69x 线）** | 按 GLM 质量排序取前 20–30%，160 token 下两模型逐 token logp → 54 维特征 → 多项 LR → `argmax(z + log p_member/T')` | +0.7h（在 6.05–6.7h 栈之上）；Flash 0.011 s/token-row、GLM 0.020 s/token-row + 23/30 min 分段装载 | LR **在 train 上现拟**，引擎冻结 | **0.682→0.689**；5 次同码 0.682/0.688/0.690/0.691/0.694；OOF +0.032（20–30% 覆盖）；glm 召回 0.27→0.63、flash 0.45→0.65 |

## 3. 从 0.65 到 0.68x 再到 0.69x，按证据强度排序

1. **Qwen3-Embedding-8B native 多深度专家进 40% 份额。** 唯一有"加一个专家 → 跳一档"证据的组件：lightningv08 的 0.666 加权栈 + 该专家 = 0.680（alexchilton 在自己账号复现 0.680）；该专家单独一个 notebook 就 0.679。份额消融在两个独立账号上一致（0.30 掉 0.006、0.50 掉 0.001）。**这是 0.65→0.68 的主引擎。**
2. **GLM-5.3 / Flash 真实似然对比重排。** 0.682→0.689（作者测），alexchilton 五次同码均值 ≈0.689（0.684 基线两次 0.684）；OOF +0.032 全部落在 GLM 两类。**这是 0.68→0.69 的唯一真实来源**，其余都是噪声内的微调。
3. **异构成员堆叠 + Sinkhorn + typography/风格门（0.651→0.668，10/25→11/04）。** 十几步、每步 +0.001~0.003，单步全部落在 0.009 的噪声带内；但跨作者单调、且反复被独立复现，整体 +0.017 可信。这解释了为什么"再加一个 LoRA 骨干"没用于是仍然一路涨：涨的是**多样性**，不是某个成员。
4. **Qwen3.8-27B LoRA 12% + target mix 0.50（0.681 线 → 0.682/0.683）。** +0.001~0.002。同线的 WeMM-4B 7% 也是 +0.002。都贴着噪声带。
5. **ATLAS 20%（0.660→0.663）。** 注意 ATLAS 本身是**生成式**打分器（`mean_lp` + `exp(−gap/T)` 四温度列，`SourceLinear` 按来源路由的 LoRA），不是分类头。在 0.684 里它被关掉了。
6. **GSQ-27B 3%。** 证据方向是反的：份额从 3% 加到 6%/12% 分数下降（0.684/0.682/0.681）。它存在但不涨分，别在它身上花时间。
7. **族内质量保持的风格残差（pengyipeng1 0.687、kirilltym 移植）。** 四个小 LR 只学标点/虚词/文本形状，族内重分配 ≤10%，族总质量不变。作者自述 0.684→0.687，**未被独立复现**，且同一作者自陈"没有建立可靠因果增益"。

**反向证据（同样是结论）**：给强集成再加 LoRA 骨干/成员 ≤0.004（alexchilton 17 组 A/B）；ModernBERT 10% −0.011、20% −0.014（prvsiyan）；按 prompt 簇做 Hungarian 分配 −0.009~−0.017（多次）。

## 4. 与我们 Q8 读出正交的槽位

我们的 Q8 与 q8v5 **结构完全同形**（Qwen3-8B-Base 前 24 层、`[2,4,6,8,12,16,20,24]` 均值池化、逐层 LN 拼接、rank16/alpha32、7 投影、384 token），本地 OOF（2,699 条、150 prompt 五折）：plain 0.6420、centred 0.6883、`P+.35C+.22G` 0.6917；最弱类 glm-5.3 0.21、claude-fable-5 0.43、gemini-3.7-flash 0.48。

| 槽位 | 界面要求 | 我们的可插拔性 | 判断 |
|---|---|---|---|
| `q8v5`（3/13 term） | `adapter/adapter_config.json`+`adapter_model.safetensors`、`heads.pt`（`norms.{0..7}`、`plain.*`、`centered.*`）、`meta.json` 的 `labels` | **完全同形，字面意义 drop-in**（改 `AQ8V5` 路径即可） | 可替换，但这是**最冗余**的槽（已有 q14v5/ll8v5/n14b/q25），且"加/换 LoRA 成员 ≤0.004"是反复测过的结论。预期中性 |
| `native`（40% 份额） | `fold{k}/trainables.safetensors` + `calibration/fold{k}.json`（`plain_scale`/`centered_scale`/`selected_centered_weight`），36 层、深度 `(3,6,9,12,18,24,30,36)`、最后一 token 池化 | 需要按该形状**重训**（换 Qwen3-Embedding-8B、全 36 层、8 个深度、最后一 token），接口不需改 | **最高杠杆的单槽**。一个 0.679 的专家能吃掉 40% 份额，说明这个槽位对"专家质量"最敏感 |
| `gauss` 头 | 同一 `heads.pt` 特征 | 同形 | 可换，但收益在 0.001 级 |
| `comp_stack`（15%） | 自有 12 个预拟合头 + 门，`add_member(key, prob)` 接口 | 需要伪造一个 `key` 并改 `W`；纯 CPU、现拟 | 可插，但作者侧的 `W` 是手调常数，插入新成员要自带权重与归一 |
| `GSQ`（3%） | `gsq_stage.run()` 返回 18 维概率 | 无接口意义 | 忽略 |
| GLM 似然层 | 独立进程 + `report.json`，在栈之后 | **不需底座，与我们完全正交** | 已有人做；剩下的开放权重类别才是增量 |

**结论性判断**：
- 把 Q8 adapter 塞进 `q8v5` 是最容易、也最可能是零收益的做法：它只值 3/13 的几何混合，而同族已有三个骨干。
- 真正正交且高杠杆的两条：(1) 把 Q8 重构成 native 槽的形状（Qwen3-Embedding-8B / 36 层 / 8 深度 / 最后一 token / 5 折 prompt 分组 / plain+centred 双头），直接吃 40% 份额；(2) 生成式似然，针对**还没被做过的开放权重类**（glm 两类已被做掉；`qwen3.8-max` 有 27B 权重与 GSQ 资产，`kimi-k3` 是否开放权重公开材料里没写死）。生成式似然与我们任何判别式 adapter 都零相关，是唯一不与其他队伍抢同一信号的维度。
- 我们的 centred 单支 OOF 0.688 高于 0.684 榜分，但**不能**当作会涨分的证据：alexchilton 的 0.6865 嵌套 CV → 榜上 0.56 是这类误判的标本；GoPain 给出的修正做法（每题只取 2 条回答做 OOF）把与榜分的相关性从 0.13 提到 0.91。

## 5. 复现口径提醒

- 0.684 是**确定性**的：同一份代码两次都是 0.684。带 GLM 似然层的 0.682–0.694 是**非确定性**的（时间预算决定覆盖多少条），所以那条线的单次分数不能当排名。
- 标题不可信：`ozertuu/0-689-...-apex` 现在的正文写 0.696；`lbjyang/ntl-700-fixed160-prefetch` 标题写 700 但正文没有任何实测分数；`heliosli/name-that-llm-lv683-a15` 的 notebook 正文仍是 0.684 原文。
- 若干 notebook 里的 "NeurIPS 2024" 是别处复制来的样板文字，本赛截止 2027-01-14、奖金 50,000 USD。

## 6. 本项目对 0.696 与 Q8 的实验安排

公开可抄的最高分是 0.696（`feiyu26/ntl-0684-v1-graft` v1；`areebbb/0-696-this-actually-works` 是它的逐行镜像并钉死了挂载顺序）。0.696 相对 0.684 的增量只有一件事：在 0.684 栈之后加一个 **GLM-5.3 / GLM-5.3-Flash 真实似然重打分**阶段，用 `ntl-segrun` 把 IQ3_XXS GGUF 分层装进两张 T4，按固定时间预算只覆盖栈认为属于 GLM 的那部分文本。其余建模代码不变。

本项目据此安排三组对照，全部在 qingyu1232（主队）上跑：

| 组 | 代码 | 目的 |
|---|---|---|
| A | `jiyujia/ntl-0696-mirror` + 提交 ref 58380844 | 复现公开 0.696，确认这条线在我们账号上的真实分数 |
| B | `qingyu1232/ntl-q8v5-submit` | 0.684 栈 + 本项目 Q8 六件套替换 `q8v5` 槽位，隔离 Q8 的槽位增益 |
| C | A 的代码 + Q8 槽位替换 | Q8 与 0.696 线的叠加，回答“能否结合、结合后是否仍正” |

判读口径：单次提交噪声在 0.012 量级（同一份 0.682–0.694 的五次同码结果），所以 A−0.684 只有 0.012，单次不可区分；B−0.684 若小于 0.005 就不能算实质提升。真正可判的是 C−A，因为两者只差 `q8v5` 槽位。

`q8v5` 在 0.684 的 13 个 term 里占 3 个（plain/twin/gauss），名义份额 8.75%（singleton 行）–11.50%（grouped 行），经 native 40%、GSQ 3%、comp 15% 稀释后有效 log-prob 份额约 4.3–5.7%。因此“成员 OOF 提高 2.8pp”不能线性外推；这也是本项目必须先做槽位消融、再决定是否把 Q8 往 `native`（40% 份额）迁移的原因。

## 7. 实测：0.696 在本账号复现

`areebbb/0-696-this-actually-works`（feiyu26 v1 的逐行镜像）在本队账号上以 **submission ref 58380844 跑出 public LB 0.696**，与公开声称一致，比我们此前的 0.684 高 **0.012**。这两个数正好等于本项目反复观察到的单次提交噪声量级，所以 0.684→0.696 的差异本身不能当作“某个组件有效”的因果证据；它只能说明这条公开线的上限确实更高。

同批提交了 0.684 栈 + 本项目 Q8 六件套的受控替换（ref 58396783，只改 `adapter_dir` 指向六件套，其余 31 个 cell 逐字保留），用于隔离 `q8v5` 槽位的增益。
