"""
Visualize Feature Ranges Report

Creates comprehensive visualizations for the feature ranges calculation report,
including feature ranges, statistics, and validation metrics.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Directory structure constants
RANGE_REPORTS_DIR = "range-reports"
VISUALIZATIONS_DIR = "visualizations"


def get_base_models_dir():
    """Get the base models directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def get_range_reports_dir():
    """Get the range reports directory"""
    base_dir = get_base_models_dir()
    reports_dir = os.path.join(base_dir, RANGE_REPORTS_DIR)
    return reports_dir


def get_visualizations_dir():
    """Get the visualizations directory"""
    base_dir = get_base_models_dir()
    viz_dir = os.path.join(base_dir, VISUALIZATIONS_DIR)
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    return viz_dir


def load_report(report_path):
    """Load the feature ranges report JSON file"""
    with open(report_path, 'r') as f:
        return json.load(f)


def plot_feature_ranges(report, output_path):
    """Plot 1: Feature Normal Ranges with Mean and Statistics"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    feature_ranges = report['feature_ranges']
    feature_stats = report['feature_statistics']
    
    features = list(feature_ranges.keys())
    feature_names = [feature_ranges[f]['name'] for f in features]
    
    y_pos = np.arange(len(features))
    
    # Extract data
    means = [feature_stats[f]['mean'] for f in features]
    stds = [feature_stats[f]['std'] for f in features]
    normal_mins = [feature_ranges[f]['normal_min'] for f in features]
    normal_maxs = [feature_ranges[f]['normal_max'] for f in features]
    mins = [feature_stats[f]['min'] for f in features]
    maxs = [feature_stats[f]['max'] for f in features]
    
    # Plot normal ranges as horizontal bars
    range_widths = [normal_maxs[i] - normal_mins[i] for i in range(len(features))]
    range_starts = normal_mins
    
    # Create horizontal bar chart
    bars = ax.barh(y_pos, range_widths, left=range_starts, 
                   height=0.6, alpha=0.6, color='lightgreen', 
                   label='Normal Range (±1.5σ)')
    
    # Plot mean points
    ax.scatter(means, y_pos, color='darkblue', s=100, zorder=5, 
              label='Mean', marker='o')
    
    # Plot min/max as error bars
    for i, (mean, std, min_val, max_val) in enumerate(zip(means, stds, mins, maxs)):
        ax.plot([min_val, max_val], [i, i], 'k-', linewidth=1, alpha=0.3)
        ax.scatter([min_val, max_val], [i, i], color='red', s=30, zorder=4, marker='|')
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel('Feature Value', fontsize=11, fontweight='bold')
    ax.set_title('Feature Normal Ranges with Mean and Distribution', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add text annotations for mean values
    for i, (mean, normal_min, normal_max) in enumerate(zip(means, normal_mins, normal_maxs)):
        ax.text(mean, i, f'  {mean:.2f}', va='center', fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_feature_statistics(report, output_path):
    """Plot 2: Feature Statistics Box Plot (Percentiles)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    feature_stats = report['feature_statistics']
    feature_ranges = report['feature_ranges']
    
    features = list(feature_stats.keys())
    feature_names = [feature_ranges[f]['name'] for f in features]
    
    # Prepare data for box plot
    box_data = []
    positions = []
    labels = []
    
    for i, feat in enumerate(features):
        stats = feature_stats[feat]
        # Create box plot data: [min, p25, p50, p75, max]
        box_data.append([
            stats['min'],
            stats['p25'],
            stats['p50'],  # median
            stats['p75'],
            stats['max']
        ])
        positions.append(i)
        labels.append(feature_names[i])
    
    # Create box plot manually
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, 
                   patch_artist=True, vert=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Add mean markers
    means = [feature_stats[f]['mean'] for f in features]
    ax.scatter(positions, means, color='red', s=80, zorder=5, 
              marker='D', label='Mean', edgecolors='darkred', linewidths=1)
    
    # Customize
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Feature Value', fontsize=11, fontweight='bold')
    ax.set_title('Feature Distribution Statistics (Box Plot)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_validation_metrics(report, output_path):
    """Plot 3: Validation Performance Metrics"""
    if 'validation_metrics' not in report or report['validation_metrics'] is None:
        print("[WARNING] Skipping validation metrics plot: No validation metrics available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    metrics = report['validation_metrics']
    
    # Plot 1: Primary Metrics Bar Chart
    ax1 = axes[0]
    primary_metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score']
    }
    
    colors = ['#2ecc71' if v >= 0.8 else '#e74c3c' if v >= 0.6 else '#f39c12' 
              for v in primary_metrics.values()]
    
    bars = ax1.bar(primary_metrics.keys(), primary_metrics.values(), 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, (name, value) in zip(bars, primary_metrics.items()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Primary Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target (0.8)')
    ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Minimum (0.6)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Confusion Matrix Heatmap
    ax2 = axes[1]
    cm = metrics['confusion_matrix']
    cm_matrix = np.array([
        [cm['tp'], cm['fn']],
        [cm['fp'], cm['tn']]
    ])
    
    # Create heatmap
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Fraudulent', 'Genuine'],
               yticklabels=['Suspicious', 'Normal'],
               ax=ax2, cbar_kws={'label': 'Count'}, 
               annot_kws={'size': 14, 'weight': 'bold'})
    
    ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Actual Label', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Predicted Label', fontsize=10, fontweight='bold')
    
    # Add text annotations
    total = cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']
    ax2.text(0.5, -0.15, f'Total Reviews: {total}', 
            transform=ax2.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_comprehensive_dashboard(report, output_path):
    """Plot 4: Comprehensive Dashboard with All Information"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    feature_ranges = report['feature_ranges']
    feature_stats = report['feature_statistics']
    features = list(feature_ranges.keys())
    feature_names = [feature_ranges[f]['name'] for f in features]
    
    # 1. Feature Ranges (Top Left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    y_pos = np.arange(len(features))
    means = [feature_stats[f]['mean'] for f in features]
    normal_mins = [feature_ranges[f]['normal_min'] for f in features]
    normal_maxs = [feature_ranges[f]['normal_max'] for f in features]
    range_widths = [normal_maxs[i] - normal_mins[i] for i in range(len(features))]
    
    ax1.barh(y_pos, range_widths, left=normal_mins, height=0.5, 
            alpha=0.6, color='lightgreen', label='Normal Range')
    ax1.scatter(means, y_pos, color='darkblue', s=60, zorder=5, label='Mean')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_names, fontsize=8)
    ax1.set_xlabel('Feature Value', fontsize=10)
    ax1.set_title('Feature Normal Ranges', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Validation Metrics (Top Right)
    if 'validation_metrics' in report and report['validation_metrics']:
        ax2 = fig.add_subplot(gs[0, 2])
        metrics = report['validation_metrics']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        colors = ['#2ecc71' if v >= 0.8 else '#e74c3c' if v >= 0.6 else '#f39c12' 
                 for v in metric_values]
        bars = ax2.bar(metric_names, metric_values, color=colors, alpha=0.8)
        for bar, val in zip(bars, metric_values):
            ax2.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=9)
        ax2.set_title('Performance Metrics', fontsize=10, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Confusion Matrix (Middle Right)
    if 'validation_metrics' in report and report['validation_metrics']:
        ax3 = fig.add_subplot(gs[1, 2])
        cm = report['validation_metrics']['confusion_matrix']
        cm_matrix = np.array([
            [cm['tp'], cm['fn']],
            [cm['fp'], cm['tn']]
        ])
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fraud', 'Genuine'],
                   yticklabels=['Susp', 'Normal'],
                   ax=ax3, cbar=False, annot_kws={'size': 10, 'weight': 'bold'})
        ax3.set_title('Confusion Matrix', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Actual', fontsize=8)
        ax3.set_ylabel('Predicted', fontsize=8)
    
    # 4. Feature Statistics Summary (Bottom, spans 3 columns)
    ax4 = fig.add_subplot(gs[2, :])
    
    # Create a table-like visualization
    stats_data = []
    for feat in features:
        stats = feature_stats[feat]
        stats_data.append([
            feature_names[features.index(feat)],
            f"{stats['mean']:.2f}",
            f"{stats['std']:.2f}",
            f"{stats['min']:.2f}",
            f"{stats['max']:.2f}",
            f"{normal_mins[features.index(feat)]:.2f}",
            f"{normal_maxs[features.index(feat)]:.2f}"
        ])
    
    table = ax4.table(cellText=stats_data,
                      colLabels=['Feature', 'Mean', 'Std Dev', 'Min', 'Max', 
                                'Normal Min', 'Normal Max'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.axis('off')
    ax4.set_title('Feature Statistics Summary', fontsize=12, fontweight='bold', pad=10)
    
    # Add methodology info
    if 'methodology' in report:
        method = report['methodology']
        method_text = (f"Method: {method.get('range_method', 'N/A')}, "
                      f"Std Multiplier: {method.get('std_multiplier', 'N/A')}, "
                      f"Features: {method.get('feature_count', 'N/A')}")
        fig.text(0.5, 0.02, method_text, ha='center', fontsize=9, 
                style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Feature Ranges Calculation & Validation Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_range_comparison(report, output_path):
    """Plot 5: Compare Normal Ranges vs Actual Min/Max"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    feature_ranges = report['feature_ranges']
    feature_stats = report['feature_statistics']
    
    features = list(feature_ranges.keys())
    feature_names = [feature_ranges[f]['name'] for f in features]
    
    y_pos = np.arange(len(features))
    
    # Extract data
    normal_mins = [feature_ranges[f]['normal_min'] for f in features]
    normal_maxs = [feature_ranges[f]['normal_max'] for f in features]
    actual_mins = [feature_stats[f]['min'] for f in features]
    actual_maxs = [feature_stats[f]['max'] for f in features]
    means = [feature_stats[f]['mean'] for f in features]
    
    # Plot actual range
    actual_ranges = [actual_maxs[i] - actual_mins[i] for i in range(len(features))]
    ax.barh(y_pos, actual_ranges, left=actual_mins, height=0.4, 
           alpha=0.4, color='lightcoral', label='Actual Range (Min-Max)')
    
    # Plot normal range
    normal_range_widths = [normal_maxs[i] - normal_mins[i] for i in range(len(features))]
    ax.barh(y_pos, normal_range_widths, left=normal_mins, height=0.3, 
           alpha=0.7, color='lightgreen', label='Normal Range (±1.5σ)')
    
    # Plot mean
    ax.scatter(means, y_pos, color='darkblue', s=80, zorder=5, 
              label='Mean', marker='D', edgecolors='white', linewidths=1.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel('Feature Value', fontsize=11, fontweight='bold')
    ax.set_title('Normal Range vs Actual Data Range Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def main():
    """Main function to generate all visualizations"""
    print("=" * 80)
    print("FEATURE RANGES VISUALIZATION")
    print("=" * 80)
    
    # Get paths
    reports_dir = get_range_reports_dir()
    viz_dir = get_visualizations_dir()
    
    report_path = os.path.join(reports_dir, 'feature_ranges_report.json')
    
    if not os.path.exists(report_path):
        print(f"\nError: Report file not found at: {report_path}")
        print("Please run feature_range_calculation.py first to generate the report.")
        return
    
    print(f"\nLoading report from: {report_path}")
    report = load_report(report_path)
    print("[OK] Report loaded successfully")
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    
    # 1. Feature Ranges
    output1 = os.path.join(viz_dir, 'feature_ranges.png')
    plot_feature_ranges(report, output1)
    
    # 2. Feature Statistics
    output2 = os.path.join(viz_dir, 'feature_statistics.png')
    plot_feature_statistics(report, output2)
    
    # 3. Validation Metrics
    output3 = os.path.join(viz_dir, 'validation_metrics.png')
    plot_validation_metrics(report, output3)
    
    # 4. Comprehensive Dashboard
    output4 = os.path.join(viz_dir, 'feature_ranges_dashboard.png')
    plot_comprehensive_dashboard(report, output4)
    
    # 5. Range Comparison
    output5 = os.path.join(viz_dir, 'range_comparison.png')
    plot_range_comparison(report, output5)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {viz_dir}")
    print("\nGenerated files:")
    print(f"  1. feature_ranges.png - Feature normal ranges visualization")
    print(f"  2. feature_statistics.png - Feature distribution statistics")
    print(f"  3. validation_metrics.png - Validation performance metrics")
    print(f"  4. feature_ranges_dashboard.png - Comprehensive dashboard")
    print(f"  5. range_comparison.png - Normal vs actual range comparison")


if __name__ == "__main__":
    main()

