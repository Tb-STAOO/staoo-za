# Error Sample Analysis Report

## 1. Objective
- Analyze misclassified test samples for three models.
- Provide structured outputs: summary table, per-model misclassified lists, and waveform figures.

## 2. Data and Models
- Test dataset root: `F:\data\newdata`
- Models: `cnn1d, inceptiontime, convnext1d`
- Test set size: 200 samples (100 no + 100 yes).

## 3. Error Summary
- Summary CSV: `F:\yes_no_classifier_project\error_analysis\tables\error_summary.csv`

| Model | Accuracy | Error Rate | Errors | no->yes | yes->no | Avg Wrong Confidence |
|---|---:|---:|---:|---:|---:|---:|
| cnn1d | 0.9200 | 0.0800 | 16/200 | 10 | 6 | 0.6698 |
| inceptiontime | 0.9200 | 0.0800 | 16/200 | 8 | 8 | 0.6679 |
| convnext1d | 0.9100 | 0.0900 | 18/200 | 8 | 10 | 0.7602 |

## 4. Per-Model Error Analysis
### 4.1 cnn1d
- Checkpoint: `F:\yes_no_classifier_project\model\cnn1d\best_model.pth`
- Misclassified list (CSV): `F:\yes_no_classifier_project\error_analysis\tables\cnn1d_misclassified.csv`
- Error pattern: `no->yes=10`, `yes->no=6`
- Typical error waveform plots:
  - `F:\yes_no_classifier_project\error_analysis\figures\cnn1d_error_rank01_idx036.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\cnn1d_error_rank02_idx065.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\cnn1d_error_rank03_idx038.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\cnn1d_error_rank04_idx039.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\cnn1d_error_rank05_idx181.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\cnn1d_error_rank06_idx077.png`

### 4.2 inceptiontime
- Checkpoint: `F:\yes_no_classifier_project\model\inceptiontime\best_model.pth`
- Misclassified list (CSV): `F:\yes_no_classifier_project\error_analysis\tables\inceptiontime_misclassified.csv`
- Error pattern: `no->yes=8`, `yes->no=8`
- Typical error waveform plots:
  - `F:\yes_no_classifier_project\error_analysis\figures\inceptiontime_error_rank01_idx036.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\inceptiontime_error_rank02_idx181.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\inceptiontime_error_rank03_idx038.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\inceptiontime_error_rank04_idx037.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\inceptiontime_error_rank05_idx171.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\inceptiontime_error_rank06_idx026.png`

### 4.3 convnext1d
- Checkpoint: `F:\yes_no_classifier_project\model\convnext1d\best_model.pth`
- Misclassified list (CSV): `F:\yes_no_classifier_project\error_analysis\tables\convnext1d_misclassified.csv`
- Error pattern: `no->yes=8`, `yes->no=10`
- Typical error waveform plots:
  - `F:\yes_no_classifier_project\error_analysis\figures\convnext1d_error_rank01_idx036.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\convnext1d_error_rank02_idx168.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\convnext1d_error_rank03_idx038.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\convnext1d_error_rank04_idx133.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\convnext1d_error_rank05_idx039.png`
  - `F:\yes_no_classifier_project\error_analysis\figures\convnext1d_error_rank06_idx077.png`

## 5. Conclusion and Suggestions
- Best model by accuracy: `cnn1d` (0.9200).
- Highest error model: `convnext1d` (0.0900 error rate).
- Suggested next steps:
  - Add class-wise threshold tuning on validation set.
  - Add hard-sample mining for high-confidence wrong predictions.
  - Apply stronger regularization only when overfitting appears.

## 6. Output Paths
- Root folder: `F:\yes_no_classifier_project\error_analysis`
- Report file: `F:\yes_no_classifier_project\ERROR_ANALYSIS_REPORT.md`
