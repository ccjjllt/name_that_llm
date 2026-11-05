# 实验记录

## 2026-11-04：TF-IDF + Logistic Regression 起点

目的：建立可复用的本地分组验证和实验产物，获得本项目自己的起始结果。没有调参，也没有提交 Kaggle。

- Run：`20261104T123209Z_tfidf_lr_844505`
- 数据：官方训练集 2,699 条，150 个 prompt，18 个类别；没有额外数据。
- 切分：`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`，按 `prompt_id` 分组。
- 特征：词 1–2 gram，最多 25,000 维；字符 3–5 gram，最多 35,000 维；`min_df=2`，`lowercase=False`，`sublinear_tf=True`。
- 模型：Logistic Regression，`C=2.0`，`max_iter=500`。
- 实验主体耗时：24.605 秒，包含交叉验证、全量拟合和示例预测，不含环境安装与模块导入。

| 指标 | 实测结果 |
|---|---:|
| OOF balanced accuracy | 0.312252 |
| 分折均值 | 0.312248 |
| 分折标准差 | 0.010420 |
| <60 词（555 条） | 0.232902 |
| 60–99 词（767 条） | 0.289523 |
| 100–159 词（723 条） | 0.339890 |
| ≥160 词（654 条） | 0.356587 |

较弱类别的 recall：`claude-fable-5` 0.0805、`glm-5.3-flash` 0.1067、`glm-5.3` 0.1667。较强类别：`gemini-3.1-pro-preview` 0.6533、`claude-opus-5` 0.6267。

观察：短文本明显更难，各型号表现差异较大。下一步优先在相同切分下比较字符 1–5 gram 与简单排版/节奏特征；先看这些变化是否改善短文本和弱类别，再考虑更大模型。社区报告的不同配置与切分分数不直接作为本次实验的对照。

产物：

- [完整指标](../runs/20261104T123209Z_tfidf_lr_844505/metrics.json)
- [配置与环境版本](../runs/20261104T123209Z_tfidf_lr_844505/config.json)
- [OOF 概率](../runs/20261104T123209Z_tfidf_lr_844505/oof.parquet)
- [混淆矩阵](../runs/20261104T123209Z_tfidf_lr_844505/confusion_matrix.csv)
- 同目录包含 `model.joblib` 与 `submission.parquet`。示例预测仅对应公开的 3 行，不代表隐藏测试成绩。

运行命令（本次限制 BLAS/OpenMP 为单线程）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --frozen python -m ntllm.baseline --config configs/tfidf.toml --fit-full
```

OOF 覆盖全部 2,699 个唯一 ID，每个 prompt 的回答均在同一验证折中；概率与示例预测产物已实际读取。没有新增工程测试套件。
