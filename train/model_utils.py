"""
Model Utility Functions for Ablation Study
Provides parameter counting, memory tracking, and statistical analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import time
import psutil
import os

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in a PyTorch model
    
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def get_model_size_mb(model: nn.Module) -> float:
    """
    Estimate model size in megabytes
    
    Note: This is an approximation based on parameter count
    """
    param_count = sum(p.numel() for p in model.parameters())
    # Assuming float32 parameters (4 bytes each)
    size_bytes = param_count * 4
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics
    
    Returns:
        Dictionary with memory usage in MB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        'shared_mb': memory_info.shared / (1024 * 1024),  # Shared Memory
        'percent': process.memory_percent()  # Percentage of total memory
    }

def measure_inference_time(model: nn.Module, input_tensor: torch.Tensor, 
                          num_iterations: int = 100, warmup: int = 10) -> Dict[str, float]:
    """
    Measure inference time for a model
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor for inference
        num_iterations: Number of iterations to average over
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Measure inference time
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'p95': float(np.percentile(times, 95)),
        'p99': float(np.percentile(times, 99))
    }

def calculate_flops(model: nn.Module, input_size: Tuple[int, ...]) -> float:
    """
    Calculate FLOPs (Floating Point Operations) for a model
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (e.g., (batch_size, channels, height, width))
    
    Returns:
        Estimated FLOPs count
    """
    from thop import profile
    
    if input_size[0] is None:
        # Use batch size 1 for estimation
        input_size = (1,) + input_size[1:]
    
    dummy_input = torch.randn(*input_size)
    
    try:
        flops, params = profile(model, inputs=(dummy_input,))
        return flops
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return 0.0
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        return 0.0

def calculate_compute_memory_ratio(model: nn.Module, input_size: Tuple[int, ...]) -> float:
    """
    Calculate compute-to-memory ratio (FLOPs per parameter)
    
    Returns:
        FLOPs per parameter ratio
    """
    flops = calculate_flops(model, input_size)
    param_count = sum(p.numel() for p in model.parameters())
    
    if param_count > 0:
        return flops / param_count
    return 0.0

def compare_models(model1: nn.Module, model2: nn.Module, 
                  input_size: Tuple[int, ...]) -> Dict[str, Dict]:
    """
    Compare two models across multiple metrics
    
    Returns:
        Dictionary with comparison metrics for both models
    """
    # Measure inference time
    dummy_input = torch.randn(*input_size)
    time1 = measure_inference_time(model1, dummy_input)
    time2 = measure_inference_time(model2, dummy_input)
    
    # Count parameters
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    
    # Calculate FLOPs
    flops1 = calculate_flops(model1, input_size)
    flops2 = calculate_flops(model2, input_size)
    
    # Get memory usage
    mem1 = get_memory_usage()
    
    # Run second model
    _ = model2(dummy_input)
    mem2 = get_memory_usage()
    
    # Calculate compute-to-memory ratios
    cm_ratio1 = calculate_compute_memory_ratio(model1, input_size)
    cm_ratio2 = calculate_compute_memory_ratio(model2, input_size)
    
    return {
        'model1': {
            'inference_time': time1,
            'parameters': params1,
            'flops': flops1,
            'memory_usage': mem1,
            'compute_memory_ratio': cm_ratio1,
            'model_size_mb': get_model_size_mb(model1)
        },
        'model2': {
            'inference_time': time2,
            'parameters': params2,
            'flops': flops2,
            'memory_usage': mem2,
            'compute_memory_ratio': cm_ratio2,
            'model_size_mb': get_model_size_mb(model2)
        },
        'comparison': {
            'speedup': time1['mean'] / time2['mean'] if time2['mean'] > 0 else float('inf'),
            'parameter_increase': (params2['total'] - params1['total']) / params1['total'] * 100,
            'flops_increase': (flops2 - flops1) / flops1 * 100 if flops1 > 0 else float('inf'),
            'memory_increase': (mem2['rss_mb'] - mem1['rss_mb']) / mem1['rss_mb'] * 100,
            'efficiency_ratio': (time1['mean'] / params1['total']) / (time2['mean'] / params2['total']) if params2['total'] > 0 else float('inf')
        }
    }

def save_comparison_report(comparison_data: Dict[str, Dict], output_path: str):
    """
    Save comparison report to file
    """
    with open(output_path, 'w') as f:
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MODEL 1 METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Parameters: {comparison_data['model1']['parameters']['total']:,}\n")
        f.write(f"Trainable Parameters: {comparison_data['model1']['parameters']['trainable']:,}\n")
        f.write(f"Model Size: {comparison_data['model1']['model_size_mb']:.2f} MB\n")
        f.write(f"FLOPs: {comparison_data['model1']['flops']:,.0f}\n")
        f.write(f"Memory Usage: {comparison_data['model1']['memory_usage']['rss_mb']:.2f} MB RSS\n")
        f.write(f"Inference Time: {comparison_data['model1']['inference_time']['mean']*1000:.2f} ms "
                f"(±{comparison_data['model1']['inference_time']['std']*1000:.2f} ms)\n")
        f.write(f"Compute-to-Memory Ratio: {comparison_data['model1']['compute_memory_ratio']:.2f} FLOPs/param\n\n")
        
        f.write("MODEL 2 METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Parameters: {comparison_data['model2']['parameters']['total']:,}\n")
        f.write(f"Trainable Parameters: {comparison_data['model2']['parameters']['trainable']:,}\n")
        f.write(f"Model Size: {comparison_data['model2']['model_size_mb']:.2f} MB\n")
        f.write(f"FLOPs: {comparison_data['model2']['flops']:,.0f}\n")
        f.write(f"Memory Usage: {comparison_data['model2']['memory_usage']['rss_mb']:.2f} MB RSS\n")
        f.write(f"Inference Time: {comparison_data['model2']['inference_time']['mean']*1000:.2f} ms "
                f"(±{comparison_data['model2']['inference_time']['std']*1000:.2f} ms)\n")
        f.write(f"Compute-to-Memory Ratio: {comparison_data['model2']['compute_memory_ratio']:.2f} FLOPs/param\n\n")
        
        f.write("COMPARISON METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Speedup (Model1/Model2): {comparison_data['comparison']['speedup']:.2f}x\n")
        f.write(f"Parameter Increase: {comparison_data['comparison']['parameter_increase']:.1f}%\n")
        f.write(f"FLOPs Increase: {comparison_data['comparison']['flops_increase']:.1f}%\n")
        f.write(f"Memory Increase: {comparison_data['comparison']['memory_increase']:.1f}%\n")
        f.write(f"Efficiency Ratio: {comparison_data['comparison']['efficiency_ratio']:.2f}\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        if comparison_data['comparison']['speedup'] > 1:
            f.write(f"• Model 2 is {comparison_data['comparison']['speedup']:.2f}x FASTER than Model 1\n")
        else:
            f.write(f"• Model 2 is {1/comparison_data['comparison']['speedup']:.2f}x SLOWER than Model 1\n")
        
        if comparison_data['comparison']['parameter_increase'] > 0:
            f.write(f"• Model 2 has {comparison_data['comparison']['parameter_increase']:.1f}% MORE parameters\n")
        else:
            f.write(f"• Model 2 has {-comparison_data['comparison']['parameter_increase']:.1f}% FEWER parameters\n")
        
        if comparison_data['comparison']['efficiency_ratio'] > 1:
            f.write(f"• Model 2 is {comparison_data['comparison']['efficiency_ratio']:.2f}x MORE efficient (performance per parameter)\n")
        else:
            f.write(f"• Model 2 is {1/comparison_data['comparison']['efficiency_ratio']:.2f}x LESS efficient (performance per parameter)\n")

def create_latex_comparison_table(comparison_data: Dict[str, Dict], model1_name: str = "Baseline U-Net",
                                   model2_name: str = "HybridSSCloudUNet") -> str:
    """
    Create LaTeX table for thesis
    """
    latex = """\\begin{table}[h]
\\centering
\\caption{Model Comparison: %s vs %s}
\\label{tab:model_comparison}
\\begin{tabular}{lrr}
\\toprule
Metric & %s & %s \\\\
\\midrule
""" % (model1_name, model2_name, model1_name, model2_name)
    
    # Add metrics to table
    metrics = [
        ("Total Parameters", f"\\num{{{comparison_data['model1']['parameters']['total']:,}}}", 
         f"\\num{{{comparison_data['model2']['parameters']['total']:,}}}"),
        ("Trainable Parameters", f"\\num{{{comparison_data['model1']['parameters']['trainable']:,}}}", 
         f"\\num{{{comparison_data['model2']['parameters']['trainable']:,}}}"),
        ("Model Size (MB)", f"{comparison_data['model1']['model_size_mb']:.2f}", 
         f"{comparison_data['model2']['model_size_mb']:.2f}"),
        ("FLOPs (billions)", f"{comparison_data['model1']['flops']/1e9:.2f}", 
         f"{comparison_data['model2']['flops']/1e9:.2f}"),
        ("Memory Usage (MB)", f"{comparison_data['model1']['memory_usage']['rss_mb']:.1f}", 
         f"{comparison_data['model2']['memory_usage']['rss_mb']:.1f}"),
        ("Inference Time (ms)", f"{comparison_data['model1']['inference_time']['mean']*1000:.2f}", 
         f"{comparison_data['model2']['inference_time']['mean']*1000:.2f}"),
        ("Inference Std Dev (ms)", f"{comparison_data['model1']['inference_time']['std']*1000:.2f}", 
         f"{comparison_data['model2']['inference_time']['std']*1000:.2f}"),
    ]
    
    for metric_name, value1, value2 in metrics:
        latex += f"{metric_name} & {value1} & {value2} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex