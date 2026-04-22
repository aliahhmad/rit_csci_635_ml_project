import pandas
import matplotlib.pyplot as plotter
import numpy as np
import seaborn as sns

models = ['logistic_regression', 'neural_network', 'xgboost']
date = '2026-04-22'

summaries = []
thresholds = []

for model in models:
    summary_df = pandas.read_csv(f"results/{model}_{date}_summary.csv")
    summary_df = summary_df.set_index("metric").T
    summary_df["Model"] = model.replace("_", " ").title()
    summaries.append(summary_df)
    threshold_df = pandas.read_csv(f"results/{model}_{date}_thresholds.csv")
    threshold_df["Model"] = model.replace("_", " ").title()
    thresholds.append(threshold_df)

all_summaries_combined = pandas.concat(summaries, ignore_index=True)
all_thresholds_combined = pandas.concat(thresholds, ignore_index=True)

all_data = []
for _, row in all_summaries_combined.iterrows():
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        all_data.append({
            'Model': row['Model'],
            'Metric': metric,
            'Score': row[metric]
        })

all_data_df = pandas.DataFrame(all_data)
all_data_df['Score'] = pandas.to_numeric(all_data_df['Score'], errors='coerce')

# Plotting Basic Model Evaluation Metrics
plotter.figure(figsize=(10, 6))
sns.barplot(data=all_data_df, x='Metric', y='Score', hue='Model', palette='Set1')
plotter.title('Model Performance Comparison at Threshold 0.5')
plotter.ylim(0.5, 1.0)
plotter.grid(axis='y', linestyle='--', alpha=0.7)
plotter.savefig('results/plots/model_comparison_bar.png')
plotter.show()

# Plotting Precision-Recall Curve
plotter.figure(figsize=(8, 6))
sns.lineplot(data=all_thresholds_combined, x='recall', y='precision', hue='Model', marker='o')
plotter.title('Precision-Recall Curves by Model')
plotter.xlabel('Recall')
plotter.ylabel('Precision')
plotter.grid(True)
plotter.savefig('results/plots/precision_recall_curve.png')
plotter.show()

# F-1 Score for different thresholds
plotter.figure(figsize=(8, 6))
sns.lineplot(data=all_thresholds_combined, x='threshold', y='f1_score', hue='Model')
plotter.title('F1-Score across Different Probability Thresholds')
plotter.ylabel('F1 Score')
plotter.xlabel('Threshold')
plotter.savefig('results/plots/f1_threshold_comparison.png')
plotter.show()

# Summary CSV
final_table = all_summaries_combined[['Model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
final_table = final_table.sort_values(by='roc_auc', ascending=False)
print(final_table.to_string(index=False))
final_table.to_csv('results/plots/final_model_comparison_table.csv', index=False)

# Confusion Matrices
fig, axes = plotter.subplots(1, 3, figsize=(18, 5))
for i, model in enumerate(models):
    df = pandas.read_csv(f"results/{model}_{date}_summary.csv").set_index('metric')
    matrix = np.array([[float(df.loc['tn', 'value']), float(df.loc['fp', 'value'])], 
                       [float(df.loc['fn', 'value']), float(df.loc['tp', 'value'])]])
    
    sns.heatmap(matrix, annot=True, fmt=',.0f', ax=axes[i], cmap='Greens', cbar=False)
    axes[i].set_title(f"CM: {model.replace('_', ' ').title()}")
    axes[i].set_xticklabels(['Pred Default', 'Pred Paid'])
    axes[i].set_yticklabels(['Actual Default', 'Actual Paid'])
plotter.tight_layout()
plotter.savefig('results/plots/confusion_matrices_grid.png')

# ROC AUC Curve
plotter.figure(figsize=(10, 6))
for model in models:
    sub = all_thresholds_combined[all_thresholds_combined['Model'] == model.replace('_', ' ').title()]
    fpr = sub['fp'] / (sub['fp'] + sub['tn'])
    tpr = sub['recall']
    plotter.plot(fpr, tpr, marker='.', label=model.replace('_', ' ').title())

plotter.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guess')
plotter.title('ROC Curve: Ability to Separate Good vs Bad Borrowers')
plotter.xlabel('False Positive Rate (Risk of Bad Approvals)')
plotter.ylabel('True Positive Rate (Success in Good Approvals)')
plotter.legend()
plotter.savefig('results/plots/roc_curve_comparison.png')