# Image index

All notebook images below were extracted directly from the persisted outputs of `notebooks/etislat.ipynb`. Figures 18 and 19 summarize notebook metrics without changing the notebook.

| Figure | File | Purpose |
|---:|---|---|
| 1 | `fig-01-target-distribution.png` | Shows the churn class imbalance |
| 2 | `fig-02-tenure-distribution.png` | Shows customer-tenure shape and concentration |
| 3 | `fig-03-tenure-boxplot.png` | Checks tenure spread and outliers |
| 4 | `fig-04-monthly-charges-distribution.png` | Shows monthly-charge distribution |
| 5 | `fig-05-monthly-charges-boxplot.png` | Checks monthly-charge spread and outliers |
| 6 | `fig-06-total-charges-distribution.png` | Shows accumulated-charge skew |
| 7 | `fig-07-total-charges-boxplot.png` | Checks accumulated-charge extremes |
| 8 | `fig-08-categorical-feature-distributions.png` | Audits all category frequencies |
| 9 | `fig-09-categorical-features-vs-churn.png` | Compares category counts by churn status |
| 10 | `fig-10-numeric-features-vs-churn-boxplots.png` | Compares numeric feature distributions by target |
| 11 | `fig-11-numeric-features-by-churn-kde.png` | Shows churn-conditioned numeric density |
| 12 | `fig-12-spearman-numeric-correlation.png` | Rank correlation across numeric inputs |
| 13 | `fig-13-cramers-v-categorical-association.png` | Effect-size matrix for categorical associations |
| 14 | `fig-14-logistic-oof-confusion-matrix.png` | Initial Logistic out-of-fold errors |
| 15 | `fig-15-threshold-precision-recall-f1.png` | Shows the operating-threshold tradeoff |
| 16 | `fig-16-exploratory-ensemble-oof-confusion-matrix.png` | Exploratory ensemble OOF outcomes |
| 17 | `fig-17-exploratory-ensemble-oof-roc.png` | Exploratory ensemble ranking curve |
| 18 | `fig-18-final-model-comparison.png` | Final OOF model comparison from notebook values |
| 19 | `fig-19-final-test-confusion-matrix.png` | Final untouched-holdout confusion matrix |

## Application and report evidence

| File | Purpose |
|---|---|
| `ui/single-customer-assessment-complete.png` | Live localhost UI with a complete 19/19 conversational assessment, risk score, threshold, signals, action, and report button |
| `demo/single-customer-report.png` | Rendered preview of the generated single-customer PDF |
| `demo/bulk-findings-summary.png` | Rendered first-page summary of the 10-customer bulk report |
| `demo/bulk-findings-detail.png` | Rendered row-by-row findings and recommended actions from the bulk report |
