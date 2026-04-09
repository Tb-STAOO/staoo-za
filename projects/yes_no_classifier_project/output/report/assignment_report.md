# 一维波形二分类实验报告（自动生成）

- 生成时间: 2026-04-04 17:42:32
- 数据集路径: `F:/data/newdata`
- 模型数量: 3（cnn1d / inceptiontime / convnext1d）

## 1. 评价指标对比

| Rank | Model | Test Acc | Precision | Recall | F1 | ROC-AUC | Best Seed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | cnn1d | 0.9200 | 0.9038 | 0.9400 | 0.9216 | 0.9662 | 42 |
| 2 | inceptiontime | 0.9200 | 0.9200 | 0.9200 | 0.9200 | 0.9760 | 42 |
| 3 | convnext1d | 0.9100 | 0.9184 | 0.9000 | 0.9091 | 0.9592 | 42 |

## 2. 最优模型

- 最优模型: **cnn1d**，Test Acc=0.9200，F1=0.9216，ROC-AUC=0.9662
- 混淆矩阵: `[[90, 10], [6, 94]]`

## 3. 作业要求输出文件

- 指标总表: `F:\yes_no_classifier_project\output\report\metrics_summary.csv`
- 模型对比JSON: `F:\yes_no_classifier_project\output\model_comparison.json`
- 模型对比CSV: `F:\yes_no_classifier_project\output\model_comparison.csv`
- 每模型训练曲线图与混淆矩阵图: `output/report/figures/`
- 每模型原始详细指标: `output/<model>/metrics.json`

## 4. 各模型图像文件

### cnn1d
- 训练曲线: `F:\yes_no_classifier_project\output\report\figures\cnn1d_training_curve.png`
- 混淆矩阵: `F:\yes_no_classifier_project\output\report\figures\cnn1d_confusion_matrix.png`
### inceptiontime
- 训练曲线: `F:\yes_no_classifier_project\output\report\figures\inceptiontime_training_curve.png`
- 混淆矩阵: `F:\yes_no_classifier_project\output\report\figures\inceptiontime_confusion_matrix.png`
### convnext1d
- 训练曲线: `F:\yes_no_classifier_project\output\report\figures\convnext1d_training_curve.png`
- 混淆矩阵: `F:\yes_no_classifier_project\output\report\figures\convnext1d_confusion_matrix.png`