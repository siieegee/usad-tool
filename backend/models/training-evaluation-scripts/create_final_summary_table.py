"""
Create Final Summary Table/Visualization

Generates a comprehensive table and visualization showing all final values
for feature ranges and validation criteria.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

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


def create_summary_table_visualization(report, output_path):
    """Create a comprehensive table visualization with all final values"""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, 
                          left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # ========== PART 1: FEATURE RANGES TABLE ==========
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    ax1.set_title('FINAL FEATURE RANGES - Normal Values for Each Feature', 
                 fontsize=16, fontweight='bold', pad=20)
    
    feature_ranges = report['feature_ranges']
    feature_stats = report['feature_statistics']
    
    # Prepare table data
    table_data = []
    for feat_key, feat_data in feature_ranges.items():
        stats = feature_stats[feat_key]
        table_data.append([
            feat_data['name'],
            f"{stats['mean']:.3f}",
            f"{stats['std']:.3f}",
            f"{feat_data['normal_min']:.3f}",
            f"{feat_data['normal_max']:.3f}",
            f"{stats['min']:.3f}",
            f"{stats['max']:.3f}"
        ])
    
    # Create table
    table1 = ax1.table(cellText=table_data,
                      colLabels=['Feature Name', 'Mean', 'Std Dev', 
                                'Normal Min', 'Normal Max', 
                                'Actual Min', 'Actual Max'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2.2)
    
    # Style header row
    for i in range(7):
        table1[(0, i)].set_facecolor('#2E86AB')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows (alternating colors)
    for i in range(1, len(table_data) + 1):
        for j in range(7):
            if i % 2 == 0:
                table1[(i, j)].set_facecolor('#F0F0F0')
            else:
                table1[(i, j)].set_facecolor('#FFFFFF')
            table1[(i, j)].set_text_props(size=9)
    
    # Highlight normal range columns
    for i in range(1, len(table_data) + 1):
        table1[(i, 3)].set_facecolor('#90EE90')  # Normal Min - light green
        table1[(i, 4)].set_facecolor('#90EE90')  # Normal Max - light green
    
    # ========== PART 2: VALIDATION METRICS TABLE ==========
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    ax2.set_title('VALIDATION PERFORMANCE METRICS', 
                 fontsize=14, fontweight='bold', pad=15)
    
    if 'validation_metrics' in report and report['validation_metrics']:
        metrics = report['validation_metrics']
        cm = metrics['confusion_matrix']
        
        # Primary metrics
        primary_data = [
            ['Accuracy', f"{metrics['accuracy']:.4f}", f"{metrics['accuracy']*100:.2f}%"],
            ['Precision', f"{metrics['precision']:.4f}", f"{metrics['precision']*100:.2f}%"],
            ['Recall (Sensitivity)', f"{metrics['recall']:.4f}", f"{metrics['recall']*100:.2f}%"],
            ['F1-Score', f"{metrics['f1_score']:.4f}", f"{metrics['f1_score']*100:.2f}%"],
            ['Specificity', f"{metrics['specificity']:.4f}", f"{metrics['specificity']*100:.2f}%"],
            ['False Alarm Rate', f"{metrics['false_alarm_rate']:.4f}", f"{metrics['false_alarm_rate']*100:.2f}%"],
            ['Detection Rate', f"{metrics['detection_rate']:.4f}", f"{metrics['detection_rate']*100:.2f}%"]
        ]
        
        table2 = ax2.table(cellText=primary_data,
                          colLabels=['Metric', 'Value', 'Percentage'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.5, 0.25, 0.25])
        
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 2.5)
        
        # Style header
        for i in range(3):
            table2[(0, i)].set_facecolor('#2E86AB')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code metrics based on performance
        for i in range(1, len(primary_data) + 1):
            metric_name = primary_data[i-1][0]
            value = float(primary_data[i-1][1])
            
            # Color coding
            if metric_name in ['False Alarm Rate']:
                # Lower is better
                if value < 0.2:
                    color = '#90EE90'  # Good
                elif value < 0.4:
                    color = '#FFD700'  # Acceptable
                else:
                    color = '#FF6B6B'  # Poor
            else:
                # Higher is better
                if value >= 0.85:
                    color = '#90EE90'  # Excellent
                elif value >= 0.70:
                    color = '#FFD700'  # Good
                elif value >= 0.60:
                    color = '#FFA500'  # Acceptable
                else:
                    color = '#FF6B6B'  # Poor
            
            for j in range(3):
                table2[(i, j)].set_facecolor(color)
                table2[(i, j)].set_text_props(size=9, weight='bold')
    else:
        ax2.text(0.5, 0.5, 'No validation metrics available', 
                ha='center', va='center', fontsize=12, style='italic')
    
    # ========== PART 3: CONFUSION MATRIX ==========
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    ax3.set_title('CONFUSION MATRIX', 
                 fontsize=14, fontweight='bold', pad=15)
    
    if 'validation_metrics' in report and report['validation_metrics']:
        cm = metrics['confusion_matrix']
        
        # Create confusion matrix table
        cm_data = [
            ['', 'Predicted: Fraudulent', 'Predicted: Genuine'],
            ['Actual: Fraudulent', f"{cm['tp']}", f"{cm['fn']}"],
            ['Actual: Genuine', f"{cm['fp']}", f"{cm['tn']}"]
        ]
        
        table3 = ax3.table(cellText=cm_data[1:],  # Skip header row for table
                          colLabels=cm_data[0],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.4, 0.3, 0.3])
        
        table3.auto_set_font_size(False)
        table3.set_fontsize(11)
        table3.scale(1, 2.5)
        
        # Style header
        for i in range(3):
            table3[(0, i)].set_facecolor('#2E86AB')
            table3[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style confusion matrix cells
        # TP (True Positive) - Green
        table3[(1, 1)].set_facecolor('#90EE90')
        table3[(1, 1)].set_text_props(size=12, weight='bold')
        
        # FN (False Negative) - Red
        table3[(1, 2)].set_facecolor('#FF6B6B')
        table3[(1, 2)].set_text_props(size=12, weight='bold')
        
        # FP (False Positive) - Orange
        table3[(2, 1)].set_facecolor('#FFA500')
        table3[(2, 1)].set_text_props(size=12, weight='bold')
        
        # TN (True Negative) - Light Green
        table3[(2, 2)].set_facecolor('#90EE90')
        table3[(2, 2)].set_text_props(size=12, weight='bold')
        
        # Row labels
        table3[(1, 0)].set_facecolor('#E8E8E8')
        table3[(1, 0)].set_text_props(weight='bold')
        table3[(2, 0)].set_facecolor('#E8E8E8')
        table3[(2, 0)].set_text_props(weight='bold')
        
        # Add summary text
        total = cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']
        summary_text = (f"Total Reviews: {total}\n"
                       f"Correct Predictions: {cm['tp'] + cm['tn']} ({((cm['tp'] + cm['tn'])/total*100):.1f}%)\n"
                       f"Incorrect Predictions: {cm['fp'] + cm['fn']} ({((cm['fp'] + cm['fn'])/total*100):.1f}%)")
        
        ax3.text(0.5, -0.15, summary_text, transform=ax3.transAxes,
                ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        ax3.text(0.5, 0.5, 'No confusion matrix available', 
                ha='center', va='center', fontsize=12, style='italic')
    
    # ========== PART 4: METHODOLOGY & SUMMARY ==========
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Methodology info
    if 'methodology' in report:
        method = report['methodology']
        method_text = (
            f"METHODOLOGY: {method.get('range_method', 'N/A').upper().replace('_', ' ')}\n"
            f"Standard Deviation Multiplier: {method.get('std_multiplier', 'N/A')}\n"
            f"Total Features Analyzed: {method.get('feature_count', 'N/A')}"
        )
    else:
        method_text = "Methodology information not available"
    
    # Performance assessment
    if 'validation_metrics' in report and report['validation_metrics']:
        metrics = report['validation_metrics']
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1_score']
        
        # Determine overall assessment
        if accuracy >= 0.85 and precision >= 0.80 and recall >= 0.85:
            assessment = "EXCELLENT - All criteria met. Ranges approved for production."
            assessment_color = '#2E7D32'
        elif accuracy >= 0.80 and precision >= 0.75 and recall >= 0.80:
            assessment = "GOOD - Performance is acceptable. Ranges approved for production."
            assessment_color = '#558B2F'
        elif accuracy >= 0.70 and precision >= 0.60 and recall >= 0.60:
            assessment = "ACCEPTABLE - Performance meets minimum thresholds. Consider refinement."
            assessment_color = '#F57C00'
        else:
            assessment = "POOR - Performance below acceptable thresholds. Ranges need refinement."
            assessment_color = '#C62828'
        
        performance_text = (
            f"PERFORMANCE ASSESSMENT:\n{assessment}\n\n"
            f"KEY FINDINGS:\n"
            f"  • Accuracy: {accuracy*100:.2f}% {'✓' if accuracy >= 0.85 else '⚠' if accuracy >= 0.70 else '✗'}\n"
            f"  • Precision: {precision*100:.2f}% {'✓' if precision >= 0.80 else '⚠' if precision >= 0.70 else '✗'}\n"
            f"  • Recall: {recall*100:.2f}% {'✓' if recall >= 0.85 else '⚠' if recall >= 0.70 else '✗'}\n"
            f"  • F1-Score: {f1*100:.2f}% {'✓' if f1 >= 0.80 else '⚠' if f1 >= 0.70 else '✗'}\n\n"
            f"RECOMMENDATIONS:\n"
        )
        
        recommendations = []
        if precision < 0.80:
            recommendations.append("  • Too many false positives - consider expanding normal ranges")
        if recall < 0.85:
            recommendations.append("  • Too many false negatives - consider narrowing normal ranges")
        if accuracy < 0.85:
            recommendations.append("  • Overall accuracy needs improvement - review feature ranges")
        if not recommendations:
            recommendations.append("  • No major issues detected. Ranges are performing well.")
        
        performance_text += "\n".join(recommendations)
    else:
        performance_text = "Validation metrics not available. Cannot assess performance."
        assessment_color = '#757575'
    
    # Combine text
    full_text = f"{method_text}\n\n{performance_text}"
    
    ax4.text(0.5, 0.5, full_text, transform=ax4.transAxes,
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', 
                     edgecolor=assessment_color, linewidth=2, alpha=0.9),
            family='monospace')
    
    ax4.set_title('METHODOLOGY & PERFORMANCE ASSESSMENT', 
                 fontsize=14, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle('FEATURE RANGES - FINAL VALUES SUMMARY', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def create_csv_summary(report, output_path):
    """Create a CSV file with all final values"""
    feature_ranges = report['feature_ranges']
    feature_stats = report['feature_statistics']
    
    # Feature ranges data
    feature_data = []
    for feat_key, feat_data in feature_ranges.items():
        stats = feature_stats[feat_key]
        feature_data.append({
            'Feature Name': feat_data['name'],
            'Mean': round(stats['mean'], 4),
            'Std Dev': round(stats['std'], 4),
            'Normal Min': round(feat_data['normal_min'], 4),
            'Normal Max': round(feat_data['normal_max'], 4),
            'Actual Min': round(stats['min'], 4),
            'Actual Max': round(stats['max'], 4),
            '5th Percentile': round(stats['p5'], 4),
            '25th Percentile': round(stats['p25'], 4),
            'Median (50th)': round(stats['p50'], 4),
            '75th Percentile': round(stats['p75'], 4),
            '95th Percentile': round(stats['p95'], 4)
        })
    
    feature_df = pd.DataFrame(feature_data)
    
    # Validation metrics data
    if 'validation_metrics' in report and report['validation_metrics']:
        metrics = report['validation_metrics']
        cm = metrics['confusion_matrix']
        
        metrics_data = [{
            'Metric': 'Accuracy',
            'Value': round(metrics['accuracy'], 4),
            'Percentage': f"{metrics['accuracy']*100:.2f}%"
        }, {
            'Metric': 'Precision',
            'Value': round(metrics['precision'], 4),
            'Percentage': f"{metrics['precision']*100:.2f}%"
        }, {
            'Metric': 'Recall (Sensitivity)',
            'Value': round(metrics['recall'], 4),
            'Percentage': f"{metrics['recall']*100:.2f}%"
        }, {
            'Metric': 'F1-Score',
            'Value': round(metrics['f1_score'], 4),
            'Percentage': f"{metrics['f1_score']*100:.2f}%"
        }, {
            'Metric': 'Specificity',
            'Value': round(metrics['specificity'], 4),
            'Percentage': f"{metrics['specificity']*100:.2f}%"
        }, {
            'Metric': 'False Alarm Rate',
            'Value': round(metrics['false_alarm_rate'], 4),
            'Percentage': f"{metrics['false_alarm_rate']*100:.2f}%"
        }, {
            'Metric': 'Detection Rate',
            'Value': round(metrics['detection_rate'], 4),
            'Percentage': f"{metrics['detection_rate']*100:.2f}%"
        }]
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Confusion matrix
        cm_data = [{
            'Type': 'True Positive (TP)',
            'Count': cm['tp'],
            'Description': 'Correctly identified fraudulent reviews'
        }, {
            'Type': 'True Negative (TN)',
            'Count': cm['tn'],
            'Description': 'Correctly identified genuine reviews'
        }, {
            'Type': 'False Positive (FP)',
            'Count': cm['fp'],
            'Description': 'Genuine reviews wrongly flagged as suspicious'
        }, {
            'Type': 'False Negative (FN)',
            'Count': cm['fn'],
            'Description': 'Fraudulent reviews that slipped through'
        }]
        
        cm_df = pd.DataFrame(cm_data)
        
        # Save to Excel with multiple sheets (if openpyxl is available)
        try:
            excel_path = output_path.replace('.csv', '.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                feature_df.to_excel(writer, sheet_name='Feature Ranges', index=False)
                metrics_df.to_excel(writer, sheet_name='Validation Metrics', index=False)
                cm_df.to_excel(writer, sheet_name='Confusion Matrix', index=False)
            print(f"[OK] Saved Excel file: {excel_path}")
        except ImportError:
            print(f"[INFO] openpyxl not installed. Skipping Excel export. CSV file available.")
        except Exception as e:
            print(f"[WARNING] Could not create Excel file: {e}. CSV file available.")
    
    # Also save CSV
    feature_df.to_csv(output_path, index=False)
    print(f"[OK] Saved CSV file: {output_path}")


def main():
    """Main function"""
    print("=" * 80)
    print("FINAL VALUES SUMMARY - TABLE & VISUALIZATION")
    print("=" * 80)
    
    # Get paths
    reports_dir = get_range_reports_dir()
    viz_dir = get_visualizations_dir()
    
    report_path = os.path.join(reports_dir, 'feature_ranges_report.json')
    
    if not os.path.exists(report_path):
        print(f"\nError: Report file not found at: {report_path}")
        return
    
    print(f"\nLoading report from: {report_path}")
    report = load_report(report_path)
    print("[OK] Report loaded successfully")
    
    # Create visualization
    print("\nGenerating final summary table visualization...")
    output_viz = os.path.join(viz_dir, 'final_values_summary.png')
    create_summary_table_visualization(report, output_viz)
    
    # Create CSV/Excel summary
    print("\nGenerating CSV/Excel summary...")
    output_csv = os.path.join(reports_dir, 'final_values_summary.csv')
    create_csv_summary(report, output_csv)
    
    print("\n" + "=" * 80)
    print("SUMMARY GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nFiles created:")
    print(f"  1. {output_viz}")
    print(f"  2. {output_csv}")
    print(f"  3. {output_csv.replace('.csv', '.xlsx')}")


if __name__ == "__main__":
    main()

