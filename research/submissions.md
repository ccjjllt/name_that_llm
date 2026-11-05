# 比赛实验记录

2026-11-04 复现 Omer 的公开集成方案，公开榜得到 **0.668**。该结果作为后续研究的整体参照。[原方案 v2](https://www.kaggle.com/code/omerfarukmerey/0-668-verifier-matched-rows-pool-prompts) · [本项目结果](https://www.kaggle.com/code/ccjjllt/name-that-llm-public-0668-reproduction?scriptVersionId=365647412)

| 实验 | 数据 | 配方 | Public LB |
|---|---|---|---:|
| 公开集成复现 | 官方 2,699 条文本、Open Pool v13 的 190,588 条文本，以及原方案其余外部数据 | 多模型 LoRA 特征、n-gram 分类器、互补集成、ATLAS、软边际调整、文风重排与 pairwise verifier | **0.668** |

此次使用原方案的建模参数。后续数据实验以 v13 为该成绩的版本参照，v14 的 211,369 行属于新增数据条件，需要单独比较其影响。[Open Pool v13](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/13) · [Open Pool v14](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/14)

公开示例只有 3 行，完整成绩来自隐藏测试。末尾重排代码存在会影响执行路径的问题，现有分数只能支持整套方案的结果，尚不足以确定各项重排的独立贡献。下一步用同一组折外预测分别消融文风重排、同题特征和 verifier。[公开方案与提分研究](public-solutions.md)
