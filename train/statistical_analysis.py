#!/usr/bin/env python3
"""
Statistical Analysis Script for Ablation Study
Performs statistical tests to determine significance of improvements
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_results_from_directories(baseline_dir, hybrid_dir):
    """Load evaluation results from multiple directories"""
    results = {
        'baseline': [],
        'hybrid': []
    }
    
    # Load baseline results
    baseline_files = list(Path(baseline_dir).glob('evaluation_summary.txt'))
    if baseline_files:
        baseline_results = parse_summary_file(baseline_files[0])
        results['baseline'] = baseline_results
    
    # Load hybrid results
    hybrid_files = list(Path(hybrid_dir).glob('evaluation_summary.txt'))
    if hybrid_files:
        hybrid_results = parse_summary_file(hybrid_files[0])
        results['hybrid'] = hybrid_results
    
    return results

def parse_summary_file(filepath):
    """Parse evaluation summary file"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Dice=' in line:
                parts = line.strip().split(':')
                epoch = int(parts[0].split()[1])
                metrics_part = parts[1].strip()
                
                # Extract Dice score
                dice_match = next((m for m in metrics_part.split(',') if 'Dice=' in m), None)
                if dice_match:
                    dice_score = float(dice_match.split('=')[1])
                    
                    # Extract inference time if available
                    time_match = next((m for m in metrics_part.split(',') if 'Avg Inference Time=' in m), None)
                    inference_time = float(time_match.split('=')[1].replace('s', '')) if time_match else None
                    
                    results.append({
                        'epoch': epoch,
                        'dice': dice_score,
                        'inference_time': inference_time
                    })
    
    return results

def perform_statistical_tests(baseline_results, hybrid_results):
    """Perform statistical significance tests"""
    
    # Extract Dice scores
    baseline_dice = [r['dice'] for r in baseline_results if r['dice'] is not None]
    hybrid_dice = [r['dice'] for r in hybrid_results if r['dice'] is not None]
    
    if not baseline_dice or not hybrid_dice:
        return None
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(baseline_dice, hybrid_dice, equal_var=False)
    
    # Perform Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(baseline_dice, hybrid_dice)
    
    # Calculate effect sizes
    mean_baseline = np.mean(baseline_dice)
    mean_hybrid = np.mean(hybrid_dice)
    std_baseline = np.std(baseline_dice)
    std_hybrid = np.std(hybrid_dice)
    
    # Cohen's d
    pooled_std = np.sqrt((std_baseline**2 + std_hybrid**2) / 2)
    cohens_d = (mean_hybrid - mean_baseline) / pooled_std if pooled_std > 0 else 0
    
    # Hedges' g (small sample bias correction)
    n_baseline = len(baseline_dice)
    n_hybrid = len(hybrid_dice)
    correction_factor = 1 - 3/(4*(n_baseline + n_hybrid - 2) - 1)
    hedges_g = cohens_d * correction_factor if correction_factor > 0 else cohens_d
    
    # Confidence intervals
    baseline_ci = stats.t.interval(0.95, len(baseline_dice)-1, 
                                  loc=mean_baseline, scale=stats.sem(baseline_dice))
    hybrid_ci = stats.t.interval(0.95, len(hybrid_dice)-1,
                                 loc=mean_hybrid, scale=stats.sem(hybrid_dice))
    
    return {
        'baseline_stats': {
            'mean': mean_baseline,
            'std': std_baseline,
            'n': n_baseline,
            'ci_95': baseline_ci
        },
        'hybrid_stats': {
            'mean': mean_hybrid,
            'std': std_hybrid,
            'n': n_hybrid,
            'ci_95': hybrid_ci
        },
        't_test': {
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'degrees_freedom': n_baseline + n_hybrid - 2
        },
        'mann_whitney': {
            'statistic': u_stat,
            'p_value': u_pvalue,
            'significant': u_pvalue < 0.05
        },
        'effect_sizes': {
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'interpretation': interpret_effect_size(abs(cohens_d)),
            'improvement_percentage': ((mean_hybrid - mean_baseline) / mean_baseline * 100) if mean_baseline > 0 else 0
        }
    }

def interpret_effect_size(d):
    """Interpret Cohen's d effect size"""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def create_statistical_report(stats_results, output_dir):
    """Create comprehensive statistical report"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not stats_results:
        print("No statistical results to report")
        return
    
    # 1. Create text report
    report_path = output_dir / 'statistical_report.txt'
    with open(report_path, 'w') as f:
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline U-Net (n={stats_results['baseline_stats']['n']}):\n")
        f.write(f"  Mean Dice Score: {stats_results['baseline_stats']['mean']:.4f}\n")
        f.write(f"  Standard Deviation: {stats_results['baseline_stats']['std']:.4f}\n")
        f.write(f"  95% Confidence Interval: [{stats_results['baseline_stats']['ci_95'][0]:.4f}, "
                f"{stats_results['baseline_stats']['ci_95'][1]:.4f}]\n\n")
        
        f.write(f"HybridSSCloudUNet (n={stats_results['hybrid_stats']['n']}):\n")
        f.write(f"  Mean Dice Score: {stats_results['hybrid_stats']['mean']:.4f}\n")
        f.write(f"  Standard Deviation: {stats_results['hybrid_stats']['std']:.4f}\n")
        f.write(f"  95% Confidence Interval: [{stats_results['hybrid_stats']['ci_95'][0]:.4f}, "
                f"{stats_results['hybrid_stats']['ci_95'][1]:.4f}]\n\n")
        
        f.write("STATISTICAL TESTS\n")
        f.write("-" * 40 + "\n")
        f.write("Independent Samples t-test:\n")
        f.write(f"  t-statistic: {stats_results['t_test']['statistic']:.4f}\n")
        f.write(f"  p-value: {stats_results['t_test']['p_value']:.6f}\n")
        f.write(f"  Degrees of Freedom: {stats_results['t_test']['degrees_freedom']}\n")
        f.write(f"  Statistically Significant: {stats_results['t_test']['significant']}\n\n")
        
        f.write("Mann-Whitney U Test (non-parametric):\n")
        f.write(f"  U-statistic: {stats_results['mann_whitney']['statistic']:.2f}\n")
        f.write(f"  p-value: {stats_results['mann_whitney']['p_value']:.6f}\n")
        f.write(f"  Statistically Significant: {stats_results['mann_whitney']['significant']}\n\n")
        
        f.write("EFFECT SIZE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Cohen's d: {stats_results['effect_sizes']['cohens_d']:.4f}\n")
        f.write(f"Hedges' g: {stats_results['effect_sizes']['hedges_g']:.4f}\n")
        f.write(f"Effect Size Interpretation: {stats_results['effect_sizes']['interpretation']}\n")
        f.write(f"Improvement Percentage: {stats_results['effect_sizes']['improvement_percentage']:.2f}%\n\n")
        
        f.write("CONCLUSION\n")
        f.write("-" * 40 + "\n")
        if stats_results['t_test']['significant']:
            if stats_results['hybrid_stats']['mean'] > stats_results['baseline_stats']['mean']:
                f.write("✓ HybridSSCloudUNet demonstrates statistically significant improvement over Baseline U-Net.\n")
                f.write(f"  Average improvement: {stats_results['effect_sizes']['improvement_percentage']:.2f}%\n")
                f.write(f"  Effect size: {stats_results['effect_sizes']['interpretation']} ({stats_results['effect_sizes']['cohens_d']:.4f})\n")
            else:
                f.write("✗ HybridSSCloudUNet shows statistically significant worse performance than Baseline U-Net.\n")
        else:
            f.write("○ No statistically significant difference detected between models.\n")
            f.write("  The observed differences could be due to random variation.\n")
    
    print(f"Statistical report saved to {report_path}")
    
    # 2. Create visualization
    create_statistical_visualization(stats_results, output_dir)
    
    # 3. Save JSON data
    json_path = output_dir / 'statistical_results.json'
    with open(json_path, 'w') as f:
        json.dump(stats_results, f, indent=2, default=float)
    
    print(f"JSON data saved to {json_path}")
    
    # 4. Create LaTeX table
    latex_path = output_dir / 'statistical_results.tex'
    create_latex_statistical_table(stats_results, latex_path)
    
    print(f"LaTeX table saved to {latex_path}")

def create_statistical_visualization(stats_results, output_dir):
    """Create statistical visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Box plot comparison
    ax1 = axes[0, 0]
    baseline_data = [r['dice'] for r in baseline_results if r['dice'] is not None]
    hybrid_data = [r['dice'] for r in hybrid_results if r['dice'] is not None]
    
    box_data = [baseline_data, hybrid_data]
    box_labels = ['Baseline U-Net', 'HybridSSCloudUNet']
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax1.set_ylabel('Dice Score', fontsize=12)
    ax1.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add p-value annotation
    p_val = stats_results['t_test']['p_value']
    if p_val < 0.001:
        sig_text = f"p < 0.001"
    else:
        sig_text = f"p = {p_val:.3f}"
    
    y_max = max(max(baseline_data), max(hybrid_data))
    ax1.text(1.5, y_max * 0.95, sig_text, ha='center', va='bottom', fontsize=12, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # Confidence interval plot
    ax2 = axes[0, 1]
    models = ['Baseline U-Net', 'HybridSSCloudUNet']
    means = [stats_results['baseline_stats']['mean'], stats_results['hybrid_stats']['mean']]
    ci_lower = [stats_results['baseline_stats']['ci_95'][0], stats_results['hybrid_stats']['ci_95'][0]]
    ci_upper = [stats_results['baseline_stats']['ci_95'][1], stats_results['hybrid_stats']['ci_95'][1]]
    
    x_pos = np.arange(len(models))
    ax2.errorbar(x_pos, means, yerr=[np.array(means) - np.array(ci_lower), 
                                      np.array(ci_upper) - np.array(means)], 
                 fmt='o', capsize=5, markersize=8, color='black')
    ax2.bar(x_pos, means, alpha=0.7, color=['lightblue', 'lightcoral'])
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models)
    ax2.set_ylabel('Dice Score (Mean ± 95% CI)', fontsize=12)
    ax2.set_title('Confidence Interval Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Effect size visualization
    ax3 = axes[1, 0]
    effect_size = abs(stats_results['effect_sizes']['cohens_d'])
    effect_labels = ['Negligible (<0.2)', 'Small (0.2-0.5)', 'Medium (0.5-0.8)', 'Large (>0.8)']
    effect_ranges = [0, 0.2, 0.5, 0.8, 2.0]
    
    colors = ['lightgray', 'lightblue', 'cornflowerblue', 'royalblue']
    
    for i in range(len(effect_ranges)-1):
        if effect_ranges[i] <= effect_size < effect_ranges[i+1]:
            ax3.bar(0, effect_size, color=colors[i], alpha=0.7)
            break
    
    ax3.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_ylim(0, 1.0)
    ax3.set_xticks([])
    ax3.set_ylabel("Cohen's d", fontsize=12)
    ax3.set_title('Effect Size Visualization', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax3.text(0, effect_size + 0.03, f'd = {effect_size:.3f}', 
             ha='center', va='bottom', fontsize=11)
    
    # Improvement percentage
    ax4 = axes[1, 1]
    improvement = stats_results['effect_sizes']['improvement_percentage']
    
    colors = ['red', 'orange', 'green', 'darkgreen']
    if improvement < 0:
        bar_color = colors[0]
    elif improvement < 5:
        bar_color = colors[1]
    elif improvement < 10:
        bar_color = colors[2]
    else:
        bar_color = colors[3]
    
    ax4.bar(0, improvement, color=bar_color, alpha=0.7)
    ax4.axhline(y=0, color='black', linewidth=0.8)
    
    ax4.set_xlim(-0.5, 0.5)
    ax4.set_ylim(min(improvement - 5, -5), max(improvement + 5, 10))
    ax4.set_xticks([])
    ax4.set_ylabel('Improvement (%)', fontsize=12)
    ax4.set_title('Performance Improvement', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax4.text(0, improvement + (0.5 if improvement >= 0 else -0.5), 
             f'{improvement:+.2f}%', ha='center', va='bottom' if improvement >= 0 else 'top',
             fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / 'statistical_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Statistical visualization saved to {plot_path}")

def create_latex_statistical_table(stats_results, output_path):
    """Create LaTeX table with statistical results"""
    latex = """\\begin{table}[h]
\\centering
\\caption{Statistical Analysis Results: Baseline U-Net vs HybridSSCloudUNet}
\\label{tab:statistical_analysis}
\\begin{tabular}{lcc}
\\toprule
Metric & Baseline U-Net & HybridSSCloudUNet \\\\
\\midrule
"""
    
    latex += f"Sample Size (n) & {stats_results['baseline_stats']['n']} & {stats_results['hybrid_stats']['n']} \\\\\n"
    latex += f"Mean Dice Score & {stats_results['baseline_stats']['mean']:.4f} & {stats_results['hybrid_stats']['mean']:.4f} \\\\\n"
    latex += f"Standard Deviation & {stats_results['baseline_stats']['std']:.4f} & {stats_results['hybrid_stats']['std']:.4f} \\\\\n"
    latex += f"95\\% CI Lower & {stats_results['baseline_stats']['ci_95'][0]:.4f} & {stats_results['hybrid_stats']['ci_95'][0]:.4f} \\\\\n"
    latex += f"95\\% CI Upper & {stats_results['baseline_stats']['ci_95'][1]:.4f} & {stats_results['hybrid_stats']['ci_95'][1]:.4f} \\\\\n"
    latex += f"\\midrule\n"
    latex += f"t-statistic & \\multicolumn{{2}}{{c}}{{{stats_results['t_test']['statistic']:.4f}}} \\\\\n"
    latex += f"p-value & \\multicolumn{{2}}{{c}}{{{stats_results['t_test']['p_value']:.6f}}} \\\\\n"
    latex += f"Significant (p < 0.05) & \\multicolumn{{2}}{{c}}{{{stats_results['t_test']['significant']}}} \\\\\n"
    latex += f"Mann-Whitney U & \\multicolumn{{2}}{{c}}{{{stats_results['mann_whitney']['statistic']:.2f}}} \\\\\n"
    latex += f"Mann-Whitney p & \\multicolumn{{2}}{{c}}{{{stats_results['mann_whitney']['p_value']:.6f}}} \\\\\n"
    latex += f"Cohen's d & \\multicolumn{{2}}{{c}}{{{stats_results['effect_sizes']['cohens_d']:.4f}}} \\\\\n"
    latex += f"Hedges' g & \\multicolumn{{2}}{{c}}{{{stats_results['effect_sizes']['hedges_g']:.4f}}} \\\\\n"
    latex += f"Improvement & \\multicolumn{{2}}{{c}}{{{stats_results['effect_sizes']['improvement_percentage']:.2f}\\%}} \\\\\n"
    latex += f"Effect Size & \\multicolumn{{2}}{{c}}{{{stats_results['effect_sizes']['interpretation']}}} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    with open(output_path, 'w') as f:
        f.write(latex)

def main():
    parser = argparse.ArgumentParser(description='Perform statistical analysis for ablation study')
    parser.add_argument('--baseline-dir', type=str, required=True,
                       help='Directory containing baseline U-Net results')
    parser.add_argument('--hybrid-dir', type=str, required=True,
                       help='Directory containing HybridSSCloudUNet results')
    parser.add_argument('--output-dir', type=str, default='statistical_results',
                       help='Output directory for statistical analysis')
    
    args = parser.parse_args()
    
    print("Loading results...")
    results = load_results_from_directories(args.baseline_dir, args.hybrid_dir)
    
    baseline_results = results['baseline']
    hybrid_results = results['hybrid']
    
    if not baseline_results:
        print(f"Warning: No baseline results found in {args.baseline_dir}")
        return
    if not hybrid_results:
        print(f"Warning: No hybrid results found in {args.hybrid_dir}")
        return
    
    print(f"Baseline samples: {len(baseline_results)}")
    print(f"Hybrid samples: {len(hybrid_results)}")
    
    print("\nPerforming statistical tests...")
    stats_results = perform_statistical_tests(baseline_results, hybrid_results)
    
    if stats_results:
        print(f"\nStatistical Analysis Results:")
        print(f"  Baseline Mean: {stats_results['baseline_stats']['mean']:.4f}")
        print(f"  Hybrid Mean: {stats_results['hybrid_stats']['mean']:.4f}")
        print(f"  t-test p-value: {stats_results['t_test']['p_value']:.6f}")
        print(f"  Cohen's d: {stats_results['effect_sizes']['cohens_d']:.4f}")
        print(f"  Improvement: {stats_results['effect_sizes']['improvement_percentage']:.2f}%")
        
        if stats_results['t_test']['significant']:
            print(f"\n✓ Statistically significant difference detected (p < 0.05)")
        else:
            print(f"\n○ No statistically significant difference (p = {stats_results['t_test']['p_value']:.3f})")
        
        print(f"\nCreating report...")
        create_statistical_report(stats_results, args.output_dir)
        
        print(f"\nAnalysis complete. Results saved to {args.output_dir}/")
    else:
        print("Error: Could not perform statistical analysis")

if __name__ == '__main__':
    # Global variables for visualization function
    baseline_results = []
    hybrid_results = []
    
    main()