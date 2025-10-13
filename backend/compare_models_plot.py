import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

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
        'accuracy': 0.6770,
        'precision': 0.6628,
        'recall': 0.7521,
        'f1': 0.7046,
        #'auc': 0.7401,
        #'separation': 1.3483,
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

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    names = [m['name'] for m in models]

    for ax, (key, title, ylabel) in zip(axes, metrics):
        values = [m[key] for m in models]
        bars = ax.bar(names, values, color=colors)
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