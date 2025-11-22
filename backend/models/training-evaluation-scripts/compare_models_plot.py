"""
Program Title:
compare_models_plot.py – Unified Model Performance Comparison Plot Generator

Programmers:
Cristel Jane Baquing, Angelica Jean Evangelista, James Tristan Landa, Kharl Chester Velasco

Where the Program Fits in the General System Design:
This module belongs to the Visualization and Reporting Component of the USAD (UnSupervised Anomaly 
Detection) system. It processes evaluation outputs from baseline, clustering, and proposed hybrid models, 
and generates comparative performance visualizations across multiple metrics. These plots help developers 
and analysts assess differences in model behavior and improvements achieved by optimized or hybrid 
approaches.

Date Written and Revised:
Original version: November 22, 2025
Last revised: November 22, 2025

Purpose:
To generate a single consolidated PNG visualization comparing multiple models (DBSCAN, KMeans, and 
Proposed Model) using performance metrics sourced from evaluation_report.json and manually supplied 
baseline results.  
The visualization supports:
• Accuracy comparison  
• Precision comparison  
• Recall comparison  
• F1-score comparison  
The output image is automatically saved in visualizations/.

Data Structures, Algorithms, and Control:
• Data Structures:
  - JSON evaluation report loaded from evaluation-reports/evaluation_report.json  
  - Python dictionaries storing model names and metric values  
  - Automatically generated directories for visualization outputs  

• Algorithms:
  - Iterative metric extraction and bar-plot generation using Matplotlib  
  - Automatic label formatting and dynamic y-axis scaling  
  - Grid-style visualization layout with four metric-specific subplots in a single figure  

• Control:
  - Resolves base directory using relative script path  
  - Safely creates the visualizations directory if missing  
  - main() orchestrates the loading of evaluation metrics and the creation of the consolidated figure  
  - Saves the final PNG file compare_models_overview.png into the visualizations/ folder  
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")

# Directory structure constants
EVALUATION_REPORTS_DIR = "evaluation-reports"
VISUALIZATIONS_DIR = "visualizations"


def get_base_models_dir():
    """Get the base models directory (parent of training-evaluation-scripts)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def get_evaluation_reports_dir():
    """Get the evaluation reports directory"""
    return os.path.join(get_base_models_dir(), EVALUATION_REPORTS_DIR)


def get_visualizations_dir():
    """Get the visualizations directory"""
    base_dir = get_base_models_dir()
    viz_dir = os.path.join(base_dir, VISUALIZATIONS_DIR)
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    return viz_dir


# Load evaluation report
reports_dir = get_evaluation_reports_dir()
with open(os.path.join(reports_dir, 'evaluation_report.json'), 'r') as f:
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
    viz_dir = get_visualizations_dir()
    output_path = os.path.join(viz_dir, 'compare_models_overview.png')
    plot_all_to_single_file(output_path)


if __name__ == '__main__':
    main()