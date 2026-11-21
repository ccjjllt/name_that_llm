# 公开方案进展（2026-11-21）

截至 2026-11-21，Team Aillis 以 **0.781** 领先。本队当前最佳提交为 sanpier v2 的原样复现 **0.709**（`58650152`），较先前 lightningv08 的 Soft Typography 公开版（`58453323`，0.702）高 0.7 个百分点。比赛截止时间为 2027-01-14 23:59 UTC，指标为 balanced accuracy。[Kaggle 排行榜](https://www.kaggle.com/competitions/name-that-llm/leaderboard) · [比赛说明](https://www.kaggle.com/competitions/name-that-llm) · [本队提交](https://www.kaggle.com/competitions/name-that-llm/submissions)

Kaggle 成绩排序首位已更新为 José Padilla 的 **0.716 Robust GLM-5.3 Likelihood + User Prefix**。作者报告同一代码两次得分 0.716 和 0.715。其队伍在官方公共榜上的最高分是 0.717，公共榜接口只返回队伍提交编号和分数。当前可运行公开版是 v5，代码单元与 v4 相同，本队原样提交 `58659450`，正待评分。[José Padilla v5](https://www.kaggle.com/code/estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix?scriptVersionId=371376285) · [Kaggle 排行榜](https://www.kaggle.com/competitions/name-that-llm/leaderboard) · [本队提交](https://www.kaggle.com/competitions/name-that-llm/submissions)

0.716 沿用 0.701 集成，重算同一批 2,699 条官方回答的两路 GLM 似然。上下文改为低推理力度，加上指定段落词数、无标题的用户请求，正文最多读 160 token。GLM 按两类总概率排序，每个权重分段结束后根据剩余时间缩小候选范围，并仅融合两路都成功得分的回答。作者报告在同一融合权重下，按 prompt 分组的训练集折外 balanced accuracy 平均增 1.37 个百分点。该版同时改了前缀和超时行为，隐藏榜单分数的增量无法归因于其中一项。[公开方法与结果](https://www.kaggle.com/code/estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix?scriptVersionId=371376285) · [公开训练特征](https://www.kaggle.com/code/estopadilla/ntl-glm-prefix-ab)

此前排名首位的是 areebbb 的 **0.711 Fingerprints + Sibling Specialists**（NTL-52，第 21 版）。它沿用 sanpier 的 320-token GLM 似然头，优先计算 Fable 与 GLM 竞争的文本，保留满足似然增幅条件的 Fable 改判，并将字形规则中的概率乘数设为 0.15。页面正文仍记录上一版 256-token 的 0.706。0.711 来自页面标题与 Kaggle 的成绩排序。本队已按公开版原样提交 `58655969`，等待 Kaggle 评分。[Areeb Ali v21](https://www.kaggle.com/code/areebbb/0-711-fingerprints-sibling-specialis?scriptVersionId=371951077) · [Kaggle 提交记录](https://www.kaggle.com/competitions/name-that-llm/submissions)

sanpier 的 **0.709** 在 0.702 集成上把 GLM-5.3 与 GLM-5.3-Flash 的真实似然窗口从 160 增至 320 token，重新拟合原有 54 维特征的逻辑回归头，并把处理预算从 90,000 提到 130,000 个 token 位置。候选按 GLM 概率质量排序，覆盖上限为 15%。作者报告似然头自身的按 prompt 分组交叉验证从 0.183 到 0.217。作者和本项目的原样复现均获 0.709。[sanpier v2](https://www.kaggle.com/code/sanpier/0-709-320-token-glm-likelihood-qwen-ensemble?scriptVersionId=371885729) · [本队提交](https://www.kaggle.com/competitions/name-that-llm/submissions)

Riad Mohamed 作者队伍最高为 **0.710**，页面标题含 0.730+ 字样。该版本先做保留家族总概率的族内重排，再按两个 GLM 的合计概率排序。由源码可知，族内重排保留这项排序量，因此主要可能影响 GLM 融合后的类别判决。末段还按 `−0.40 × log(预测类别边缘)` 调整 logits，并保护原 GLM 获胜行。[Riad Mohamed 公开 Notebook](https://www.kaggle.com/code/riadmohamed42/name-that-llm-apex-320-glm-supreme-sovereign-gold) · [Kaggle 排行榜](https://www.kaggle.com/competitions/name-that-llm/leaderboard)
