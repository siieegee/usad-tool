import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")

# Load evaluation report
with open('evaluation_report.json', 'r') as f:
    eval_data = json.load(f)

# Input metrics (from user-provided results)
models = [
    {
        'name': 'DBSCAN',
        'accuracy': 0.6323,
        'precision': 0.7456,
        'recall': 0.4285,
        'f1': 0.5442,
        #'auc': 0.6880,
        #'separation': 1.2351,
    },
    {
        'name': 'KMeans',
        'accuracy': 0.5127,
        'precision': 0.5125,
        'recall': 0.9996,
        'f1': 0.6776,
        #'auc': 0.2965,
        #'separation': 0.6770,
    },
    {
        'name': 'Proposed Model',
        'accuracy': eval_data['test_metrics']['accuracy'],
        'precision': eval_data['test_metrics']['precision'],
        'recall': eval_data['test_metrics']['recall'],
        'f1': eval_data['test_metrics']['f1_score'],
        #'auc': eval_data['roc_auc']['test'],
        #'separation': eval_data['distance_statistics']['test_separation'],
    },
]


def plot_all_to_single_file(output_path: str = 'compare_models_overview.png'):
    metrics = [
        ('accuracy', 'Accuracy by Model', 'Accuracy'),
        ('precision', 'Precision by Model', 'Precision'),
        ('recall', 'Recall by Model', 'Recall'),
        ('f1', 'F1-Score by Model', 'F1-Score'),
        #('auc', 'AUC-ROC by Model', 'AUC-ROC'),
        #('separation', 'Separation Ratio by Model', 'Separation Ratio'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    names = [m['name'] for m in models]

    for ax, (key, title, ylabel) in zip(axes, metrics):
        values = [m[key] for m in models]
        x_pos = range(len(names))
        bars = ax.bar(x_pos, values, color=colors, width=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        # y-limit: cap at >=1.0 for prob-like metrics
        ylim_top = max(1.0, max(values) * 1.1)
        ax.set_ylim(0, ylim_top)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    # Save a single image containing all four comparisons
    plot_all_to_single_file('compare_models_overview.png')


if __name__ == '__main__':
    main()