# 冲榜研究与首轮同题配对对照

## 公开前沿

2026-11-23 下午复核，Kaggle 比赛 Notebook 按分数排序的最高可查关联公开分为 **0.720**，对应 RanJeet Gupta 的 Sibling Odds v1（`372780350`），比此前的 0.719 高 0.1 个百分点。它来自 Areeb 的 0.719 版本，对 Claude、GPT、Gemini 分别计算 GLM 似然阶段前后的分数变化，在族内中心化、截断后以 0.55 加权加入原有族内分数。原前两名分差低于 0.8 且新旧首选同族时才允许交换，同时保留 GLM→Fable 的保护。两条关联分数给出完整方案的差值，单次差值尚无法估计随机波动或改判规则的独立收益。另有标题写 `0723` 的 Capacity Transport，复核时没有 Kaggle 关联分数。[按成绩排序的公开代码](https://www.kaggle.com/competitions/name-that-llm/code?sortBy=scoreDescending) · [Sibling Odds v1](https://www.kaggle.com/code/ranjeet258/name-that-llm-sibling-odds?scriptVersionId=372780350) · [Capacity Transport](https://www.kaggle.com/code/gengsr/ntl-geng-0723-capacity-transport)

同次复核，榜首 YK **0.809**，次席 NguyenThanhNhan **0.802**，未找到这两个成绩对应的公开 Notebook。0.719 的可追溯评分版本是 Dennis 的 Guarded Fable Return 和 Areeb v3 的同源复现。并列的 Deep Forensic EDA 使用同源评分代码，其数据分析与评分流程分开。Ozertuu 文中提到的 0.720+ 属于实验目标，条目关联最高为 0.718。[排行榜](https://www.kaggle.com/competitions/name-that-llm/leaderboard) · [Dennis 评分版](https://www.kaggle.com/code/dennis6886/glm-user-prefix-guarded-fable-return?scriptVersionId=372174726) · [Areeb 评分版](https://www.kaggle.com/code/areebbb/0-719-preserve-glm-to-fable-corrections?scriptVersionId=372551661)

Dennis 使用 Padilla v5 的完整集成和 Low 用户前缀 GLM。原首选为 GLM 且似然改判为 Fable 时，Fable 对最佳 GLM 的分差须增加至少 0.30、最终至少为 0.25。满足后保护该判决免受族内残差覆盖，随后仍可被字形规则改判。Areeb 报告 0.716 → 0.719，围绕它修改候选队列和阈值的结果在 0.716–0.718。保留该版本作为已启动的 ATLAS 配对实验基准。本队已有成绩为 0.717，离榜首 0.092 BA，相当于减少约 **33% 的当前宏平均错误权重**。在官方近均衡面板上，9.2 个百分点约等于 248 条净改对，这个换算只适用于该面板。[本队提交](https://www.kaggle.com/competitions/name-that-llm/submissions) · [社区消融](https://www.kaggle.com/code/areebbb/0-719-preserve-glm-to-fable-corrections?scriptVersionId=372551661)

## 过往结果对新研究的约束

- **新读出要参与主要判决。** Q8 外部训练使单成员明显变强，六件套替换旧集成小权重槽后公开榜仍下降 0.3pp。READER 用 Native 折外回放选方案，正式评分下降 0.9pp。下一轮同时看独立读出能力与完整底座中的净改对，逐阶段记录路由，而后决定融合入口。[过往实验](retired.md) · [回放与线上对照](status_20261123.md)
- **同题控制有效，推理组条件有迁移风险。** Q8 上中心化五折均提高，平均 +4.63pp。外部同题组是否可靠、隐藏文本能否聚成组及组里有哪些类别会直接影响推理。训练可用同题作为监督，部署保留逐条打分能力。若评估组关系，须重建 1、2、4、8 条组成的稀疏测试面板，重跑分组和推理。社区有人报告稀疏 OOF 与榜单更一致，本项目尚未在完整底座上测得相同关系。[Q8 配对复算](../runs/research_status_20261123/q8_oof_diagnostics.json) · [社区稀疏验证](https://www.kaggle.com/competitions/name-that-llm/discussion/753177)
- **大幅提升需要覆盖多组错误。** Q8 保存的三分支 OOF 中，GLM＋Fable、三个 Gemini Flash、GPT-5.6 三类及 Kimi＋GPT-5.5 四组，真类在前三且错误首选也在该组内的错例分别为 175、123、86、23，共 407 条，含 213 条不足 100 词的回答。完美改判在该 Q8 参照上约有 15.1pp 空间。每组可达性必须在完整集成重新测量，强制同族重排也无法补足真类出候选的问题。[Q8 折外预测](../runs/q8_official_20261113/oof) · [本次复算](../runs/paired_attribution_20261123/local_data/pair_diagnostics.json)

## 一手研究带来的可检验假设

| 研究 | 论文条件与结果 | 在本赛题的用法 |
|---|---|---|
| Idiosyncrasies，ICML 2025 | 五个生成模型归因可达 97.1% accuracy，四个相近 Qwen2.5 尺寸为 59.8%。语言模型选择写什么也有身份信号。 | 原文走主读出，词面掩码与风格视图单独消融。衡量相近型号，避免由供应商级结果推断版本级效果。 |
| DeTeCtive 与主题控制研究 | DeTeCtive 在旧 TuringBench 的 20 类任务为 82.75% accuracy，强 RoBERTa 对照 81.73%。人类作者研究表明，控制同一对话或采同主题异作者负例有助于削弱选题捷径。 | 可用真实同题回答构造成对训练。同批数据及同训练量下测损失本身，分离数据与目标改动。 |
| READER v3 | Qwen3.5 在 Agent500 单回答为 50.4%，96.2% 来自已知同一作者的 100 个交互。代理读出可见原 prompt。 | 利用表示差异的思路有价值。这个累积结果依赖作者分组和额外输入，比赛推理条件不同。 |
| ATLAS，2026 预印本 | 五个 LLM 家族联合域与版本偏移时，74.27% accuracy 对同台对照 65.27%。每源 1 万条的消融中，固定分支后逐 token 截断聚合从 66.95% 到 72.07%。主实验用四张 H100。 | 作为第二条路线，用少数难类训练真类全 token 似然，再在同一检查点比较平均与截断。家族结果只支持该机制待检验。 |

[Idiosyncrasies](https://proceedings.mlr.press/v267/sun25z.html) · [DeTeCtive](https://arxiv.org/html/2410.20964#S4.SS3) · [同对话控制](https://aclanthology.org/2022.repl4nlp-1.26/) · [同主题反事实负例](https://doi.org/10.1145/3626772.3657956) · [READER](https://arxiv.org/html/2606.10794v3) · [ATLAS](https://arxiv.org/html/2610.19583v1)

## 首轮原创实验

**假设。** 同题同源的两份真实回答，实际长度接近。在保留 18 类交叉熵的同时，要求两条回答的正确标签分配比互换标签得分更高。成对差分抵消这两条回答共享的类偏置，例如同题共有的主题先验。模型编码每条文本一次，推理时可对任意单条文段产生 18 类分数。训练优先采样 Fable–GLM、Gemini Flash、GPT-5.6、Kimi–GPT-5.5；候选可跨供应商。这个目标也可能压低正确首选的分差，因此独立 BA 与实际改判均要测。[实现](../src/ntllm/paired_attribution.py) · [配置](../configs/paired_attribution_pilot.json)

**同批对照。** 用同一 ELECTRA-large 权重、上部八层 query/value LoRA、分类头初始化、样本序列和 360 步更新。A 使用 18 类交叉熵，B 加入 0.5 权重的成对排序损失。每组 2,880 对，基准输入上限 256 token。测外部预训练直接迁移，以及在每折的其他题上拟合线性头后的折外迁移。保留两组逐行分数以做配对分析。首轮测试目标是隔离配对损失的效应，大模型和更长训练属于后续独立变量。[骨干权重来源](https://www.kaggle.com/code/lucasboesen/name-that-llm-electra-large-weights) · [首轮设计实测](../runs/paired_attribution_20261123/local_data/sample_report.json)

**同题证据与来源约束。** 本地 Open Pool v15 预检在清洗、来源隔离和题组排除后有 82,441 条训练候选，来自 23 个主要来源。一次划定的训练序列抽到 5,386 条独有正文、2,019 道题。来源隔离对照留出 drjcfmd 的主要来源样本（17 类）和 David Vertex（2 类），排除其规范题号和重合正文。drjcfmd 也发布过其他来源的合集，主要来源筛选只隔离其原始采样。David 的 Gemini 数据有相同请求和返回版本核验，适合观察 3.7/3.8 区分。新版 Open Pool 增加了由全量官方标签训练的筛选器选出的记录，首轮排除相关来源与重建官方题的数据，采样不使用 `match_weight`。同规范题号只提供内容控制。Prvsiyan v8 中有 362 个准入的 18 类题组，按实际发出的请求重新分组后为零，主要差异之一是逐行目标词数指令。若后续取得新公开数据，应记录逐字请求、实际返回版本、日期、路由和长度设定，长度要求对各标签交叉平衡。[Open Pool v18](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/18) · [Prvsiyan](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset) · [David Vertex](https://www.kaggle.com/datasets/davidpiln/ntl-vertex-gemini-gen/versions/2)

**CPU 先导。** 复用本地抽样，在 32,768 维字符 n-gram 线性模型上仅用外部数据训练，交叉熵与加入同题损失在官方集直接迁移的 BA 为 0.12481 与 0.12778，十个难对的成对顺序表现有升有降。两组绝对表现都低，字符代理缺乏独立使用依据。GPU 对照另用可训练编码器和官方训练折校准。[CPU 指标](../runs/paired_attribution_20261123/local_data/sparse_pair_proxy.json)

**T4 实测。** ELECTRA tokenizer 将大小写折叠，正文截到 256 token。两组各训练 360 步，主要来源、顺序和训练量一致。固定五折下，配对损失多改对 189 条、改错 233 条，五折校准后共掉 1.63pp，按题组重采样的 95% 区间为 −3.07 至 −0.18pp。源间效果方向不同。David Vertex 的同请求 Gemini 3.7/3.8 成对顺序从 57.41% 到 68.06%，同一类对在官方折外从 68.00% 到 62.67%。这一训练配方可能利用了随来源变化的信号。[CE 指标](../runs/paired_attribution_20261123/gpu/paired_attribution/ce_metrics.json) · [配对指标](../runs/paired_attribution_20261123/gpu/paired_attribution/paired_metrics.json) · [配对比较](../runs/paired_attribution_20261123/gpu_comparison.json)

| 第一轮评估 | CE | CE＋同题配对 |
|---|---:|---:|
| 外部训练后直接迁移至官方 BA | 0.22448 | 0.23378 |
| 官方题组五折校准 BA | **0.31304** | 0.29677 |
| 官方五折校准，不足 100 词 | **0.28104** | 0.26230 |
| 官方五折前三覆盖 | **0.67133** | 0.66875 |
| drjcfmd 主要来源留出，17 类 BA | 0.23578 | 0.24608 |
| David Vertex 留出，2 个真类的 BA | **0.44583** | 0.42500 |
| 与 Q8 三分支固定 30% 对数分数融合 | 0.69129 | 0.69277 |

Q8 参照为 0.69166，加入配对读出后净改对 3 条，区间跨零。限制在难组的低分差行后，两组都在 Fable–GLM 净改对 5 条，Gemini 三个 Flash 分别净改错 6 条和 5 条。若把相等计为半次，Gemini 3.7 比 3.8 更长的比例在官方同题上为 60.0%，在 David Vertex 上为 45.0%，在抽样前的实际训练来源上为 46.5%。这个差异值得按词长及来源复测，模型排序变化的具体机制仍待验证。当前配方缺乏作为主读出或通用重排的支持。[固定融合](../runs/paired_attribution_20261123/gpu_comparison.json) · [难组诊断](../runs/paired_attribution_20261123/gpu_routed_diagnostics.json) · [词长变化](../runs/paired_attribution_20261123/local_data/gemini_surface_shift.json)

**交叉熵续训设计。** 在首轮模型上补一批同规则的外部样本，以交叉熵继续训练 1,080 步。累计样本曝光从 5,760 到 23,040。按本地池复算，独有正文从 5,386 到 17,978，涉及 4,386 道规范题。续训以较低的学习率重新初始化优化器，差值包含新增样本、更多更新和优化器重启的共同作用。[续训配置](../configs/paired_attribution_continue.json) · [曝光复算](../runs/paired_attribution_20261123/local_data/continuation_sample_report.json)

| CE 续训评估 | 360 步 | 累计 1,440 步 |
|---|---:|---:|
| 外部训练后直接迁移至官方 BA | 0.22448 | **0.30964** |
| 官方题组五折校准 BA | 0.31304 | **0.37488** |
| 官方五折校准，不足 100 词 | 0.28104 | **0.33563** |
| 官方五折前三覆盖 | 0.67133 | **0.72800** |
| drjcfmd 主要来源留出，17 类 BA | 0.23578 | **0.31275** |
| David Vertex 留出，2 个真类的 BA | **0.44583** | 0.28750 |
| 与 Q8 三分支固定 30% 对数分数融合 | 0.69129 | **0.69240** |

独立读出增加 **6.18pp**，五折都提高，按 prompt 重采样的配对区间为 **+4.37 至 +8.00pp**。两种留源的变化方向相反。相对 Q8 的 0.69166，续训件固定融合净改对 **2 条**，区间为 **−0.48 至 +0.63pp**。续训件在 Q8 首选错误时单独答对 157 条，两者首选用真值选择可到 0.74984。两者前三候选取并集覆盖 0.95109，比 Q8 单用时多覆盖 60 条。数据与更新能提升成员能力，下一步应把资源投向对基线错例有竞争力的证据，并评估它能否在真实可用的路由下兑现。[续训指标](../runs/paired_attribution_20261123/gpu_continue/paired_attribution/ce_continued_metrics.json) · [配对区间](../runs/paired_attribution_20261123/gpu_continuation_comparison.json) · [互补性复算](../runs/paired_attribution_20261123/gpu_continuation_complementarity.json)

## 规模化路线

1. **用完整集成校准研究目标。** 以冻结的 0.719 路线建同口径逐条分数参照。完整栈里见过目标折标签的成员需要以折内版本重算，或放在独立留源集评估。记录实际候选、分差、改判、两路 GLM 覆盖以及短文和留源表现。用冻结参照挑出值得训练的混淆，并保留扩大候选召回的路径。训练路由时在内部折拟合，用未见题的外层折测净改判。
2. **换成更强的读出和更接近赛题的训练面板。** 续训使弱读出明显进步，固定融合收益仍接近零。下一轮优先评估已有的强骨干和身份、请求、长度可核对的跨来源样本，以交叉熵为首个控制。把跨题、跨来源的同类对比留作单变量，并用未见来源和短文检查是否靠得住。任何需要按题使用多条正文的实验都应补单条和稀疏面板对照。
3. **用第二种证据验证剩余难例。** 在 4–6 个高混淆标签、每类约 2,000–5,000 条同题控制回答上试 ATLAS 类源条件似然，每步用真类全 token NLL。旧 1.7B 已有共享下投影、类别上投影和软前缀，但用少量官方文本、TF-IDF 选负类与混合目标，排序 AUC 为 0.4676。新对照固定检查点测逐 token 截断的边际贡献，部署只给少量竞争候选评分，测净改对和吞吐。GLM 的真实似然继续作参照。[旧似然读出](../runs/source_likelihood_20261121/readout/source_likelihood_readout.json) · [ATLAS 消融](https://arxiv.org/html/2610.19583v1#S5.SS3)
4. **用差异来源验证上限。** 同题拓扑作为可选补充时，只学习置信加权的邻居残差。在同一模型下模拟缺候选、错组和组内重复作者，避免把官方近全类题组中的排斥优势带入不同构成的隐藏集。用完整流程的固定参照识别实际增长空间后再增加训练预算。[赛题数据说明](https://www.kaggle.com/competitions/name-that-llm/data)
