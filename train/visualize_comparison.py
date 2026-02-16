#!/usr/bin/env python3
"""
Comparison Visualization Script for Ablation Study
Creates visualizations comparing Baseline U-Net vs HybridSSCloudUNet
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from unet import UNet
from unet.unet_model import HybridSSCloudUNet
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files

def load_evaluation_results(results_dir):
    """Load evaluation results from a results directory"""
    results = {
        'dice_scores': [],
        'inference_times': [],
        'per_class_iou': {},
        'epochs': []
    }
    
    # Find all evaluation summary files
    summary_files = list(Path(results_dir).glob('evaluation_summary.txt'))
    if not summary_files:
        raise FileNotFoundError(f"No evaluation_summary.txt found in {results_dir}")
    
    summary_file = summary_files[0]
    
    # Parse the summary file
    with open(summary_file, 'r') as f:
        for line in f:
            if 'Epoch' in line:
                parts = line.strip().split(':')
                epoch = int(parts[0].split()[1])
                metrics_part = parts[1].strip()
                
                # Extract Dice score
                dice_match = next((m for m in metrics_part.split(',') if 'Dice=' in m), None)
                dice_score = float(dice_match.split('=')[1]) if dice_match else None
                
                # Extract inference time
                time_match = next((m for m in metrics_part.split(',') if 'Avg Inference Time=' in m), None)
                inference_time = float(time_match.split('=')[1].replace('s', '')) if time_match else None
                
                if dice_score is not None:
                    results['dice_scores'].append(dice_score)
                    results['inference_times'].append(inference_time)
                    results['epochs'].append(epoch)
    
    # Load per-class IoU for the best epoch
    if results['epochs']:
        best_epoch = results['epochs'][np.argmax(results['dice_scores'])]
        iou_file = Path(results_dir) / f'per_class_iou_epoch{best_epoch}.txt'
        if iou_file.exists():
            with open(iou_file, 'r') as f:
                for line in f:
                    if 'Class' in line and 'IoU =' in line:
                        parts = line.strip().split(':')
                        class_id = int(parts[0].split()[1])
                        iou_str = parts[1].split('=')[1].strip()
                        if 'NaN' not in iou_str:
                            iou_value = float(iou_str.split()[0])
                            results['per_class_iou'][class_id] = iou_value
    
    return results

def count_model_parameters(model_name, use_transformer=True, use_attention=True):
    """Count parameters for different model configurations"""
    # Create model instance
    if model_name == 'baseline':
        # Baseline U-Net (no transformer, no attention)
        model = UNet(n_channels=3, n_classes=68, bilinear=False)
    elif model_name == 'hybrid':
        # HybridSSCloudUNet with specified components
        model = HybridSSCloudUNet(n_channels=3, n_classes=68, 
                                  use_transformer=use_transformer, 
                                  use_attention=use_attention)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def create_comparison_charts(baseline_results, hybrid_results, output_dir, baseline_params=None, hybrid_params=None):
    """Create comparison charts between baseline and hybrid models"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dice Score Comparison Chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dice Score Progress
    ax1 = axes[0, 0]
    if baseline_results['epochs'] and baseline_results['dice_scores']:
        ax1.plot(baseline_results['epochs'], baseline_results['dice_scores'], 
                'b-', label='Baseline U-Net', linewidth=2, marker='o', markersize=4)
    if hybrid_results['epochs'] and hybrid_results['dice_scores']:
        ax1.plot(hybrid_results['epochs'], hybrid_results['dice_scores'], 
                'r-', label='HybridSSCloudUNet', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Dice Score', fontsize=12)
    ax1.set_title('Dice Score Progression', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Inference Time Comparison
    ax2 = axes[0, 1]
    if baseline_results['epochs'] and baseline_results['inference_times']:
        ax2.plot(baseline_results['epochs'], baseline_results['inference_times'], 
                'b-', label='Baseline U-Net', linewidth=2, marker='o', markersize=4)
    if hybrid_results['epochs'] and hybrid_results['inference_times']:
        ax2.plot(hybrid_results['epochs'], hybrid_results['inference_times'], 
                'r-', label='HybridSSCloudUNet', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Inference Time (seconds)', fontsize=12)
    ax2.set_title('Inference Time Progression', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Per-Class IoU Comparison (Bar Chart)
    ax3 = axes[1, 0]
    if baseline_results['per_class_iou'] and hybrid_results['per_class_iou']:
        classes = sorted(set(list(baseline_results['per_class_iou'].keys()) + 
                           list(hybrid_results['per_class_iou'].keys())))
        
        baseline_iou = [baseline_results['per_class_iou'].get(c, 0) for c in classes]
        hybrid_iou = [hybrid_results['per_class_iou'].get(c, 0) for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax3.bar(x - width/2, baseline_iou, width, label='Baseline U-Net', alpha=0.8)
        ax3.bar(x + width/2, hybrid_iou, width, label='HybridSSCloudUNet', alpha=0.8)
        
        # Only show every 10th label for readability
        if len(classes) > 20:
            tick_positions = list(range(0, len(classes), max(1, len(classes) // 20)))
            tick_labels = [str(classes[pos]) for pos in tick_positions]
            ax3.set_xticks(tick_positions)
            ax3.set_xticklabels(tick_labels)
        
        ax3.set_xlabel('Class ID', fontsize=12)
        ax3.set_ylabel('IoU Score', fontsize=12)
        ax3.set_title('Per-Class IoU Comparison', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Parameter Count Comparison (Pie Chart)
    ax4 = axes[1, 1]
    
    # Use provided parameter counts or calculate them
    if baseline_params is None:
        baseline_total, baseline_trainable = count_model_parameters('baseline')
    else:
        baseline_total, baseline_trainable = baseline_params
    
    if hybrid_params is None:
        hybrid_total, hybrid_trainable = count_model_parameters('hybrid', 
                                                               use_transformer=True,
                                                               use_attention=True)
    else:
        hybrid_total, hybrid_trainable = hybrid_params
    
    sizes = [baseline_total, hybrid_total]
    labels = ['Baseline U-Net', 'HybridSSCloudUNet']
    colors = ['#1f77b4', '#ff7f0e']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Total Parameter Count Comparison', fontsize=14, fontweight='bold')
    
    # Add text box with detailed info
    textstr = '\n'.join([
        f'Baseline U-Net:',
        f'  Total: {baseline_total:,}',
        f'  Trainable: {baseline_trainable:,}',
        f'',
        f'HybridSSCloudUNet:',
        f'  Total: {hybrid_total:,}',
        f'  Trainable: {hybrid_trainable:,}',
        f'',
        f'Increase: {((hybrid_total - baseline_total) / baseline_total * 100):.1f}%'
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(1.5, 0.5, textstr, transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    comparison_chart_path = output_dir / 'ablation_comparison.png'
    plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison chart saved to {comparison_chart_path}")
    
    # 2. Create Summary Statistics Table
    summary_stats = {}
    
    if baseline_results['dice_scores']:
        summary_stats['baseline'] = {
            'best_dice': max(baseline_results['dice_scores']),
            'avg_dice': np.mean(baseline_results['dice_scores']),
            'std_dice': np.std(baseline_results['dice_scores']),
            'best_epoch': baseline_results['epochs'][np.argmax(baseline_results['dice_scores'])],
            'avg_inference_time': np.mean(baseline_results['inference_times']),
            'std_inference_time': np.std(baseline_results['inference_times']),
            'total_params': baseline_total,
            'trainable_params': baseline_trainable
        }
    
    if hybrid_results['dice_scores']:
        summary_stats['hybrid'] = {
            'best_dice': max(hybrid_results['dice_scores']),
            'avg_dice': np.mean(hybrid_results['dice_scores']),
            'std_dice': np.std(hybrid_results['dice_scores']),
            'best_epoch': hybrid_results['epochs'][np.argmax(hybrid_results['dice_scores'])],
            'avg_inference_time': np.mean(hybrid_results['inference_times']),
            'std_inference_time': np.std(hybrid_results['inference_times']),
            'total_params': hybrid_total,
            'trainable_params': hybrid_trainable
        }
    
    # Save summary statistics
    summary_path = output_dir / 'comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("ABLATION STUDY COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        if 'baseline' in summary_stats:
            f.write("BASELINE U-Net\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Dice Score: {summary_stats['baseline']['best_dice']:.4f}\n")
            f.write(f"Average Dice Score: {summary_stats['baseline']['avg_dice']:.4f} ± {summary_stats['baseline']['std_dice']:.4f}\n")
            f.write(f"Best Epoch: {summary_stats['baseline']['best_epoch']}\n")
            f.write(f"Average Inference Time: {summary_stats['baseline']['avg_inference_time']:.4f}s ± {summary_stats['baseline']['std_inference_time']:.4f}s\n")
            f.write(f"Total Parameters: {baseline_total:,}\n")
            f.write(f"Trainable Parameters: {baseline_trainable:,}\n\n")
        
        if 'hybrid' in summary_stats:
            f.write("HYBRIDSSCloudUNet\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Dice Score: {summary_stats['hybrid']['best_dice']:.4f}\n")
            f.write(f"Average Dice Score: {summary_stats['hybrid']['avg_dice']:.4f} ± {summary_stats['hybrid']['std_dice']:.4f}\n")
            f.write(f"Best Epoch: {summary_stats['hybrid']['best_epoch']}\n")
            f.write(f"Average Inference Time: {summary_stats['hybrid']['avg_inference_time']:.4f}s ± {summary_stats['hybrid']['std_inference_time']:.4f}s\n")
            f.write(f"Total Parameters: {hybrid_total:,}\n")
            f.write(f"Trainable Parameters: {hybrid_trainable:,}\n\n")
        
        if 'baseline' in summary_stats and 'hybrid' in summary_stats:
            f.write("IMPROVEMENT ANALYSIS\n")
            f.write("-" * 30 + "\n")
            dice_improvement = summary_stats['hybrid']['best_dice'] - summary_stats['baseline']['best_dice']
            f.write(f"Dice Score Improvement: {dice_improvement:.4f} ({dice_improvement/summary_stats['baseline']['best_dice']*100:.1f}%)\n")
            
            time_increase = summary_stats['hybrid']['avg_inference_time'] - summary_stats['baseline']['avg_inference_time']
            f.write(f"Inference Time Increase: {time_increase:.4f}s ({time_increase/summary_stats['baseline']['avg_inference_time']*100:.1f}%)\n")
            
            param_increase = hybrid_total - baseline_total
            f.write(f"Parameter Count Increase: {param_increase:,} ({param_increase/baseline_total*100:.1f}%)\n")
            
            efficiency_ratio = dice_improvement / (time_increase / summary_stats['baseline']['avg_inference_time'])
            f.write(f"Efficiency Ratio (Improvement/Time Cost): {efficiency_ratio:.4f}\n")
    
    print(f"Summary statistics saved to {summary_path}")
    
    # 3. Create Latex Table for Thesis
    latex_path = output_dir / 'ablation_results.tex'
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Study Results: Baseline U-Net vs HybridSSCloudUNet}\n")
        f.write("\\label{tab:ablation_results}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Metric & Baseline U-Net & HybridSSCloudUNet \\\\\n")
        f.write("\\midrule\n")
        
        if 'baseline' in summary_stats and 'hybrid' in summary_stats:
            f.write(f"Dice Score (Best) & {summary_stats['baseline']['best_dice']:.4f} & {summary_stats['hybrid']['best_dice']:.4f} \\\\\n")
            f.write(f"Dice Score (Avg ± Std) & {summary_stats['baseline']['avg_dice']:.4f} ± {summary_stats['baseline']['std_dice']:.4f} & {summary_stats['hybrid']['avg_dice']:.4f} ± {summary_stats['hybrid']['std_dice']:.4f} \\\\\n")
            f.write(f"Inference Time (s) & {summary_stats['baseline']['avg_inference_time']:.4f} ± {summary_stats['baseline']['std_inference_time']:.4f} & {summary_stats['hybrid']['avg_inference_time']:.4f} ± {summary_stats['hybrid']['std_inference_time']:.4f} \\\\\n")
            f.write(f"Parameters (Total) & \\num{{{baseline_total:,}}} & \\num{{{hybrid_total:,}}} \\\\\n")
            f.write(f"Parameters (Trainable) & \\num{{{baseline_trainable:,}}} & \\num{{{hybrid_trainable:,}}} \\\\\n")
            f.write(f"Best Epoch & {summary_stats['baseline']['best_epoch']} & {summary_stats['hybrid']['best_epoch']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {latex_path}")
    
    return summary_stats

def main():
    parser = argparse.ArgumentParser(description='Visualize ablation study comparison')
    parser.add_argument('--baseline-dir', type=str, required=True,
                       help='Directory containing baseline U-Net results')
    parser.add_argument('--hybrid-dir', type=str, required=True,
                       help='Directory containing HybridSSCloudUNet results')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory for comparison charts')
    parser.add_argument('--transformer', action='store_true', default=True,
                       help='Hybrid model uses transformer (default: True)')
    parser.add_argument('--attention', action='store_true', default=True,
                       help='Hybrid model uses attention (default: True)')
    
    args = parser.parse_args()
    
    print("Loading evaluation results...")
    baseline_results = load_evaluation_results(args.baseline_dir)
    hybrid_results = load_evaluation_results(args.hybrid_dir)
    
    print(f"Baseline results loaded: {len(baseline_results['epochs'])} epochs")
    print(f"Hybrid results loaded: {len(hybrid_results['epochs'])} epochs")
    
    print("\nCreating comparison visualizations...")
    
    # Calculate parameter counts
    baseline_total, baseline_trainable = count_model_parameters('baseline')
    hybrid_total, hybrid_trainable = count_model_parameters('hybrid', 
                                                           use_transformer=args.transformer,
                                                           use_attention=args.attention)
    
    summary_stats = create_comparison_charts(
        baseline_results, 
        hybrid_results, 
        args.output_dir,
        baseline_params=(baseline_total, baseline_trainable),
        hybrid_params=(hybrid_total, hybrid_trainable)
    )
    
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPARISON COMPLETE")
    print("=" * 60)
    
    if 'baseline' in summary_stats and 'hybrid' in summary_stats:
        baseline_best = summary_stats['baseline']['best_dice']
        hybrid_best = summary_stats['hybrid']['best_dice']
        improvement = hybrid_best - baseline_best
        baseline_params = summary_stats['baseline']['total_params']
        hybrid_params = summary_stats['hybrid']['total_params']
        
        print(f"\nKey Findings:")
        print(f"  • Baseline U-Net Best Dice: {baseline_best:.4f}")
        print(f"  • HybridSSCloudUNet Best Dice: {hybrid_best:.4f}")
        print(f"  • Improvement: {improvement:.4f} ({improvement/baseline_best*100:.1f}%)")
        print(f"  • Parameter Increase: {(hybrid_params - baseline_params)/baseline_params*100:.1f}%")
        
        if improvement > 0:
            print(f"\n✓ HybridSSCloudUNet outperforms baseline by {improvement:.4f}")
        else:
            print(f"\n✗ HybridSSCloudUNet performs worse than baseline by {-improvement:.4f}")
    
    print(f"\nAll results saved to: {args.output_dir}/")
    print("Files created:")
    print(f"  • ablation_comparison.png - Comparison charts")
    print(f"  • comparison_summary.txt - Detailed statistics")
    print(f"  • ablation_results.tex - LaTeX table for thesis")

if __name__ == '__main__':
    main()