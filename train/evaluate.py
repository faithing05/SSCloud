import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import time
import os
import psutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def calculate_iou_per_class(pred_mask, true_mask, num_classes):
    """
    Calculate IoU (Intersection over Union) for each class
    """
    iou_per_class = []
    
    for class_id in range(num_classes):
        pred_class = (pred_mask == class_id)
        true_class = (true_mask == class_id)
        
        intersection = (pred_class & true_class).sum()
        union = (pred_class | true_class).sum()
        
        if union == 0:
            iou_per_class.append(float('nan'))  # No pixels of this class in ground truth
        else:
            iou_per_class.append(intersection.float() / union.float())
    
    return iou_per_class


def calculate_confusion_matrix(pred_mask, true_mask, num_classes):
    """
    Calculate confusion matrix for multi-class segmentation
    """
    # Flatten masks to 1D arrays
    pred_flat = pred_mask.flatten().cpu().numpy()
    true_flat = true_mask.flatten().cpu().numpy()
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_flat, pred_flat, labels=range(num_classes))
    
    return cm


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, return_detailed=False, epoch=None, save_dir=None):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_inference_time = 0
    memory_usage = []
    
    # Initialize arrays for per-class metrics
    num_classes = net.n_classes
    iou_sum_per_class = torch.zeros(num_classes, device=device)
    iou_count_per_class = torch.zeros(num_classes, device=device)
    
    # Initialize confusion matrix
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64) if return_detailed else None
    
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

             # predict the mask with timing and memory tracking
            start_time = time.time()
            mask_pred = net(image)
            end_time = time.time()
            total_inference_time += (end_time - start_time) / image.shape[0]  # per image time
            
            # Track memory usage
            if return_detailed:
                memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred_binary = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred_binary, mask_true.float(), reduce_batch_first=False)
                
                if return_detailed:
                    mask_pred_class = mask_pred_binary.long()
                    iou_values = calculate_iou_per_class(mask_pred_class, mask_true, 2)
                    for class_id, iou in enumerate(iou_values):
                        if not np.isnan(iou.cpu().item() if torch.is_tensor(iou) else iou):
                            iou_sum_per_class[class_id] += iou
                            iou_count_per_class[class_id] += 1
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format for Dice
                mask_true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_argmax = mask_pred.argmax(dim=1)
                mask_pred_onehot = F.one_hot(mask_pred_argmax, net.n_classes).permute(0, 3, 1, 2).float()
                
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred_onehot[:, 1:], mask_true_onehot[:, 1:], reduce_batch_first=False)
                
                if return_detailed:
                    # Calculate per-class IoU
                    for i in range(image.shape[0]):  # Iterate over batch
                        iou_values = calculate_iou_per_class(mask_pred_argmax[i], mask_true[i], num_classes)
                        for class_id, iou in enumerate(iou_values):
                            if not np.isnan(iou.cpu().item() if torch.is_tensor(iou) else iou):
                                iou_sum_per_class[class_id] += iou
                                iou_count_per_class[class_id] += 1
                    
                    # Update confusion matrix
                    cm_batch = calculate_confusion_matrix(mask_pred_argmax, mask_true, num_classes)
                    confusion_mat += cm_batch

    net.train()
    
    avg_dice = dice_score / max(num_val_batches, 1)
    avg_inference_time = total_inference_time / max(num_val_batches, 1)
    
    if return_detailed:
        # Calculate average IoU per class
        avg_iou_per_class = []
        for class_id in range(num_classes):
            if iou_count_per_class[class_id] > 0:
                avg_iou = iou_sum_per_class[class_id] / iou_count_per_class[class_id]
                avg_iou_per_class.append(float(avg_iou))
            else:
                avg_iou_per_class.append(float('nan'))
        
        # Save confusion matrix if epoch is provided
        if epoch is not None and save_dir is not None:
            save_detailed_metrics(confusion_mat, avg_iou_per_class, float(avg_dice), 
                                 avg_inference_time, epoch, save_dir, num_classes)
        
        return avg_dice, avg_iou_per_class, confusion_mat, avg_inference_time
    else:
        return avg_dice


def save_detailed_metrics(confusion_mat, avg_iou_per_class, dice_score, 
                         inference_time, epoch, save_dir, num_classes, 
                         memory_usage=None, parameter_count=None):
    """
    Save detailed evaluation metrics including confusion matrix
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save confusion matrix as image
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix for better visualization
    cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Create heatmap
    ax = sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                    cbar_kws={'label': 'Normalized Frequency'})
    
    plt.title(f'Confusion Matrix - Epoch {epoch}', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    
    # For large number of classes, show only every 10th label
    if num_classes > 20:
        tick_positions = list(range(0, num_classes, max(1, num_classes // 20)))
        tick_labels = [str(pos) for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    
    plt.tight_layout()
    cm_filename = save_path / f'confusion_matrix_epoch{epoch}.png'
    plt.savefig(cm_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Save per-class IoU as text file
    iou_filename = save_path / f'per_class_iou_epoch{epoch}.txt'
    with open(iou_filename, 'w') as f:
        f.write(f'Evaluation Results - Epoch {epoch}\n')
        f.write('=' * 50 + '\n')
        f.write(f'Overall Dice Score: {dice_score:.4f}\n')
        f.write(f'Average Inference Time per image: {inference_time:.4f} seconds\n')
        f.write('\nPer-Class IoU Metrics:\n')
        f.write('-' * 50 + '\n')
        
        for class_id, iou in enumerate(avg_iou_per_class):
            if not np.isnan(iou):
                f.write(f'Class {class_id:3d}: IoU = {iou:.4f}\n')
            else:
                f.write(f'Class {class_id:3d}: IoU = NaN (no ground truth pixels)\n')
        
        # Calculate mean IoU (excluding NaN values)
        valid_ious = [iou for iou in avg_iou_per_class if not np.isnan(iou)]
        if valid_ious:
            mean_iou = np.mean(valid_ious)
            f.write(f'\nMean IoU (excluding NaN): {mean_iou:.4f}\n')
        
        f.write('\nConfusion Matrix saved to: ' + str(cm_filename))
    
    # 3. Save summary statistics
    summary_filename = save_path / 'evaluation_summary.txt'
    with open(summary_filename, 'a') as f:
        f.write(f'Epoch {epoch:3d}: Dice={dice_score:.4f}, '
                f'Avg Inference Time={inference_time:.4f}s\n')
    
    print(f'Detailed metrics saved to {save_path}')
    print(f'  - Confusion matrix: {cm_filename}')
    print(f'  - Per-class IoU: {iou_filename}')
    print(f'  - Average inference time per image: {inference_time:.4f} seconds')