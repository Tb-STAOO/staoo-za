# Staoo-za: 一维波形二分类工程（螺栓断裂检测）

本工程用于完成“一维波形数据二分类”实验，支持并对比 3 种模型：

- `cnn1d`
- `inceptiontime`
- `convnext1d`

其中 `convnext1d` 已参考仓库实现做结构改进（LayerNorm1D、DropPath、GRN、AttentionPooling），并完成调参提升。

## 1. 工程结构

```text
yes_no_classifier_project/
├─ train_binary_classifier.py          # 训练入口（支持三模型）
├─ generate_assignment_outputs.py      # 生成指标汇总与图表报告
├─ requirements.txt
├─ README.md
├─ model/                              # 训练后的模型权重
│  ├─ cnn1d/best_model.pth
│  ├─ inceptiontime/best_model.pth
│  └─ convnext1d/best_model.pth
├─ output/                             # 评估输出与报告
│  ├─ cnn1d/metrics.json
│  ├─ inceptiontime/metrics.json
│  ├─ convnext1d/metrics.json
│  ├─ model_comparison.csv
│  ├─ model_comparison.json
│  └─ report/
│     ├─ metrics_summary.csv
│     ├─ assignment_report.md
│     └─ figures/
├─ error_analysis/                     # 错误样本分析输出
│  ├─ tables/
│  └─ figures/
├─ ERROR_ANALYSIS_REPORT.md            # 错误样本分析报告（Markdown）
├─ REPORT_MAIN_PDF_STYLE.md            # 主实验报告（Markdown）
├─ REPORT_MAIN_PDF_STYLE.pdf           # 主实验报告（PDF）
└─ REPORT_MAIN_PDF_STYLE.docx          # 主实验报告（DOCX）
```

## 2. 数据格式

默认数据集路径：`F:/data/newdata`

目录结构要求：

```text
data_root/
├─ train/
│  ├─ yes/*.txt
│  └─ no/*.txt
└─ test/
   ├─ yes/*.txt
   └─ no/*.txt
```

每个 `txt` 文件按行存储波形点，支持两列格式：

- `index value`（推荐）
- 或单列 `value`

## 3. 环境安装

```bash
conda create -n yesno_cls python=3.10 -y
conda activate yesno_cls
pip install -r requirements.txt
```

## 4. 训练命令

### 4.1 训练单个模型

```bash
python train_binary_classifier.py --data-root F:/data/newdata --model cnn1d
python train_binary_classifier.py --data-root F:/data/newdata --model inceptiontime
python train_binary_classifier.py --data-root F:/data/newdata --model convnext1d
```

### 4.2 一次性训练三模型并对比

```bash
python train_binary_classifier.py --data-root F:/data/newdata --model all --epochs 80 --max-retries 3 --target-acc 0.90
```

### 4.3 ConvNeXt1D 推荐参数（已验证提升）

```bash
python train_binary_classifier.py --data-root F:/data/newdata --model convnext1d \
  --epochs 140 --max-retries 5 --target-acc 0.90 \
  --lr 0.0003 --batch-size 32 --weight-decay 0.00005 --dropout 0.1 \
  --convnext-dim1 32 --convnext-dim2 64 --convnext-dim3 128 \
  --convnext-depth1 2 --convnext-depth2 2 --convnext-depth3 2 \
  --convnext-kernel-size 7 --convnext-drop-path 0.15 \
  --convnext-layer-scale 1e-6 --convnext-use-grn 1 --convnext-attention-pooling 1 \
  --augment-shift 5 --augment-noise 0.003
```

## 5. 生成实验报告输出

```bash
python generate_assignment_outputs.py
```

会生成：

- `output/report/metrics_summary.csv`
- `output/report/assignment_report.md`
- `output/report/figures/*.png`

## 6. 当前结果（测试集）

- `cnn1d`: Acc = `0.92`
- `inceptiontime`: Acc = `0.92`
- `convnext1d`: Acc = `0.91`

对比总表：`output/model_comparison.csv`

## 7. 报告与分析文件

- 主报告（PDF）：`REPORT_MAIN_PDF_STYLE.pdf`
- 主报告（DOCX）：`REPORT_MAIN_PDF_STYLE.docx`
- 主报告（Markdown）：`REPORT_MAIN_PDF_STYLE.md`
- 错误样本分析报告：`ERROR_ANALYSIS_REPORT.md`
- 错误样本图表与表格：`error_analysis/`

---

如需复现实验，优先执行：

1. 安装环境
2. `--model all` 训练
3. 运行 `generate_assignment_outputs.py` 生成汇总报告
