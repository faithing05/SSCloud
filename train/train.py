import argparse
import logging
import os
import random
import sys
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import HybridSSCloudUNet, UNet
from utils.data_loading import BasicDataset, CarvanaDataset, load_image
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')


def _map_mask_to_class_indices(mask_array: np.ndarray, mask_values: List) -> np.ndarray:
    mapped_mask = np.zeros(mask_array.shape[:2], dtype=np.int64)
    for class_idx, mask_value in enumerate(mask_values):
        if mask_array.ndim == 2:
            mapped_mask[mask_array == mask_value] = class_idx
        else:
            mapped_mask[(mask_array == mask_value).all(-1)] = class_idx
    return mapped_mask


def _compute_split_class_stats(dataset, subset_indices: List[int], num_classes: int) -> Dict[str, np.ndarray]:
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    image_counts = np.zeros(num_classes, dtype=np.int64)
    image_presence: List[np.ndarray] = []

    for dataset_idx in tqdm(subset_indices, desc='Analyzing class distribution', unit='img', leave=False):
        image_id = dataset.ids[dataset_idx]
        mask_file = list(dataset.mask_dir.glob(image_id + dataset.mask_suffix + '.*'))
        if len(mask_file) != 1:
            raise RuntimeError(f'Expected exactly one mask for {image_id}, found {len(mask_file)}')

        mask_array = np.asarray(load_image(mask_file[0]))

        mapped_mask = _map_mask_to_class_indices(mask_array, dataset.mask_values)
        class_hist = np.bincount(mapped_mask.reshape(-1), minlength=num_classes)

        pixel_counts += class_hist
        present_classes = class_hist > 0
        image_counts += present_classes.astype(np.int64)
        image_presence.append(present_classes)

    return {
        'pixel_counts': pixel_counts,
        'image_counts': image_counts,
        'image_presence': np.stack(image_presence) if image_presence else np.zeros((0, num_classes), dtype=bool),
    }


def _build_class_weights(pixel_counts: np.ndarray) -> torch.Tensor:
    class_weights = np.zeros_like(pixel_counts, dtype=np.float32)
    nonzero_mask = pixel_counts > 0

    if nonzero_mask.any():
        median_count = float(np.median(pixel_counts[nonzero_mask]))
        class_weights[nonzero_mask] = median_count / pixel_counts[nonzero_mask]

    class_weights = np.clip(class_weights, 0.0, 10.0)
    positive = class_weights > 0
    if positive.any():
        class_weights[positive] = class_weights[positive] / class_weights[positive].mean()

    return torch.tensor(class_weights, dtype=torch.float32)


def _build_oversampling_weights(pixel_counts: np.ndarray, image_presence: np.ndarray) -> torch.Tensor:
    sample_weights = np.ones(image_presence.shape[0], dtype=np.float64)
    if image_presence.shape[0] == 0:
        return torch.tensor(sample_weights, dtype=torch.double)

    total_pixels = float(pixel_counts.sum())
    class_freq = np.divide(pixel_counts, total_pixels, out=np.zeros_like(pixel_counts, dtype=np.float64), where=total_pixels > 0)
    class_rarity = np.zeros_like(class_freq, dtype=np.float64)
    nonzero = class_freq > 0
    class_rarity[nonzero] = 1.0 / class_freq[nonzero]

    # Ignore background when oversampling rare semantic classes
    if class_rarity.shape[0] > 0:
        class_rarity[0] = 0.0

    for idx in range(image_presence.shape[0]):
        present = np.where(image_presence[idx])[0]
        present = present[present != 0]
        if present.size > 0:
            sample_weights[idx] = float(np.mean(class_rarity[present]))

    sample_weights = np.clip(sample_weights, 1e-6, None)
    sample_weights = sample_weights / sample_weights.mean()
    return torch.tensor(sample_weights, dtype=torch.double)


def _save_class_distribution(
        results_dir: str,
        train_stats: Dict[str, np.ndarray],
        val_stats: Dict[str, np.ndarray],
        mask_values: List,
) -> None:
    output_path = Path(results_dir) / 'class_distribution.txt'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _format_split(name: str, stats: Dict[str, np.ndarray], total_images: int) -> List[str]:
        lines = [f'{name} split (images={total_images})']
        total_pixels = int(stats['pixel_counts'].sum())
        lines.append(f'Total pixels: {total_pixels}')
        lines.append('ClassIdx | RawValue | Pixels | Pixel% | ImagesWithClass')

        for class_idx, raw_value in enumerate(mask_values):
            pixels = int(stats['pixel_counts'][class_idx])
            image_count = int(stats['image_counts'][class_idx])
            pixel_pct = (100.0 * pixels / total_pixels) if total_pixels > 0 else 0.0
            lines.append(f'{class_idx:7d} | {str(raw_value):8s} | {pixels:7d} | {pixel_pct:6.3f}% | {image_count:15d}')

        lines.append('')
        return lines

    train_images = int(train_stats['image_presence'].shape[0])
    val_images = int(val_stats['image_presence'].shape[0])

    content_lines = []
    content_lines.extend(_format_split('Train', train_stats, train_images))
    content_lines.extend(_format_split('Validation', val_stats, val_images))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content_lines))

    logging.info(f'Class distribution report saved to {output_path}')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        detailed_eval: bool = False,
        results_dir: str = 'results',
        use_class_weights: bool = True,
        use_rare_oversampling: bool = True,
        save_class_distribution: bool = True,
):
    # Create checkpoint directory within results directory
    checkpoint_dir = Path(results_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_stats = _compute_split_class_stats(dataset, list(train_set.indices), model.n_classes)
    val_stats = _compute_split_class_stats(dataset, list(val_set.indices), model.n_classes)

    if save_class_distribution:
        _save_class_distribution(results_dir, train_stats, val_stats, dataset.mask_values)

    class_weights = None
    if model.n_classes > 1 and use_class_weights:
        class_weights = _build_class_weights(train_stats['pixel_counts']).to(device)
        logging.info(f'Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy().round(4).tolist()}')

    train_sampler = None
    if use_rare_oversampling:
        sample_weights = _build_oversampling_weights(train_stats['pixel_counts'], train_stats['image_presence'])
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        logging.info('Using rare-class oversampling with WeightedRandomSampler')

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=train_sampler is None, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp,
             use_class_weights=use_class_weights, use_rare_oversampling=use_rare_oversampling,
             save_class_distribution=save_class_distribution)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # Use detailed evaluation on the final epoch
                        if detailed_eval and epoch == epochs:
                            val_score, iou_per_class, confusion_mat, inference_time = evaluate(
                                model, val_loader, device, amp, 
                                return_detailed=True, 
                                epoch=epoch, 
                                save_dir=results_dir
                            )
                            
                            # Log per-class IoU for the final epoch
                            logging.info(f'Final Epoch {epoch} - Detailed Evaluation:')
                            logging.info(f'  Dice Score: {val_score:.4f}')
                            logging.info(f'  Average Inference Time per image: {inference_time:.4f} seconds')
                            
                            # Log IoU for each class (limited output for 68 classes)
                            valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
                            if valid_ious:
                                mean_iou = np.mean(valid_ious)
                                logging.info(f'  Mean IoU (excluding NaN): {mean_iou:.4f}')
                            
                            # Log first 10 classes and last 10 classes for visibility
                            logging.info('  Per-Class IoU (first 10 classes):')
                            for class_id in range(min(10, len(iou_per_class))):
                                iou = iou_per_class[class_id]
                                if not np.isnan(iou):
                                    logging.info(f'    Class {class_id:3d}: IoU = {iou:.4f}')
                                else:
                                    logging.info(f'    Class {class_id:3d}: IoU = NaN')
                            
                            if len(iou_per_class) > 10:
                                logging.info('  Per-Class IoU (last 10 classes):')
                                for class_id in range(max(0, len(iou_per_class)-10), len(iou_per_class)):
                                    iou = iou_per_class[class_id]
                                    if not np.isnan(iou):
                                        logging.info(f'    Class {class_id:3d}: IoU = {iou:.4f}')
                                    else:
                                        logging.info(f'    Class {class_id:3d}: IoU = NaN')
                        else:
                            val_score = evaluate(model, val_loader, device, amp)
                        
                        scheduler.step(val_score)
                        
                        if not (detailed_eval and epoch == epochs):
                            logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
            torch.save(state_dict, str(checkpoint_path))
            logging.info(f'Checkpoint {epoch} saved to {checkpoint_path}!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--use-transformer', action='store_true', default=True, help='Enable transformer bottleneck (default: True)')
    parser.add_argument('--no-transformer', dest='use_transformer', action='store_false', help='Disable transformer bottleneck')
    parser.add_argument('--use-attention', action='store_true', default=True, help='Enable attention gates (default: True)')
    parser.add_argument('--no-attention', dest='use_attention', action='store_false', help='Disable attention gates')
    parser.add_argument('--detailed-eval', action='store_true', default=False, help='Enable detailed evaluation with per-class IoU and confusion matrix')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save evaluation results')
    parser.add_argument('--use-class-weights', action='store_true', default=True, help='Enable weighted CrossEntropy loss (default: True)')
    parser.add_argument('--no-class-weights', dest='use_class_weights', action='store_false', help='Disable weighted CrossEntropy loss')
    parser.add_argument('--use-rare-oversampling', action='store_true', default=True, help='Enable rare-class oversampling (default: True)')
    parser.add_argument('--no-rare-oversampling', dest='use_rare_oversampling', action='store_false', help='Disable rare-class oversampling')
    parser.add_argument('--save-class-distribution', action='store_true', default=True, help='Save train/val class distribution report (default: True)')
    parser.add_argument('--no-save-class-distribution', dest='save_class_distribution', action='store_false', help='Disable class distribution report')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    
    # For ablation study: when both transformer and attention are disabled, use original UNet
    if not args.use_transformer and not args.use_attention:
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        logging.info('Using original UNet (transformer and attention disabled)')
    else:
        model = HybridSSCloudUNet(
            n_channels=3, 
            n_classes=args.classes, 
            bilinear=args.bilinear,
            use_transformer=args.use_transformer,
            use_attention=args.use_attention
        )
        logging.info(f'Using HybridSSCloudUNet (transformer: {args.use_transformer}, attention: {args.use_attention})')
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            detailed_eval=args.detailed_eval,
            results_dir=args.results_dir,
            use_class_weights=args.use_class_weights,
            use_rare_oversampling=args.use_rare_oversampling,
            save_class_distribution=args.save_class_distribution,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            detailed_eval=args.detailed_eval,
            results_dir=args.results_dir,
            use_class_weights=args.use_class_weights,
            use_rare_oversampling=args.use_rare_oversampling,
            save_class_distribution=args.save_class_distribution,
        )
