import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from utils.data_loading import BasicDataset
from unet import HybridSSCloudUNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                return_attention_maps=False):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        if return_attention_maps and hasattr(net, 'use_attention') and net.use_attention:
            output, attention_maps = net(img, return_attention_maps=True)
            output = output.cpu()
            # Process attention maps
            attention_maps = [att.cpu() for att in attention_maps]
            # Resize attention maps to match original image size
            attention_maps_resized = []
            for att_map in attention_maps:
                att_resized = F.interpolate(att_map, (full_img.size[1], full_img.size[0]), mode='bilinear')
                attention_maps_resized.append(att_resized.squeeze().numpy())
        else:
            output = net(img).cpu()
            attention_maps_resized = None
        
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    if return_attention_maps and hasattr(net, 'use_attention') and net.use_attention:
        return mask[0].long().squeeze().numpy(), attention_maps_resized
    else:
        return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--save-attention', action='store_true', default=False, help='Save attention maps as heatmaps')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = HybridSSCloudUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        if args.save_attention and hasattr(net, 'use_attention') and net.use_attention:
            result_tuple = predict_img(net=net,
                                     full_img=img,
                                     scale_factor=args.scale,
                                     out_threshold=args.mask_threshold,
                                     device=device,
                                     return_attention_maps=True)
            mask = result_tuple[0]  # Extract mask from tuple
            attention_maps = result_tuple[1]  # Extract attention maps from tuple
        else:
            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)
            attention_maps = None

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')
            
            # Save attention maps if requested
            if args.save_attention and hasattr(net, 'use_attention') and net.use_attention and attention_maps:
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Save each attention map as a heatmap
                for j, att_map in enumerate(attention_maps):
                    att_filename = f'{os.path.splitext(out_filename)[0]}_attention_level{j+1}.png'
                    
                    # Normalize attention map for visualization
                    att_normalized = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
                    
                    # Create heatmap
                    plt.figure(figsize=(10, 10))
                    plt.imshow(att_normalized, cmap='hot', interpolation='nearest')
                    plt.colorbar(label='Attention Weight')
                    plt.title(f'Attention Map Level {j+1}')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(att_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    logging.info(f'Attention map level {j+1} saved to {att_filename}')
                
                # Save combined attention map (average of all levels)
                combined_att = np.mean(np.stack(attention_maps), axis=0)
                combined_filename = f'{os.path.splitext(out_filename)[0]}_attention_combined.png'
                
                combined_normalized = (combined_att - combined_att.min()) / (combined_att.max() - combined_att.min() + 1e-8)
                plt.figure(figsize=(10, 10))
                plt.imshow(combined_normalized, cmap='hot', interpolation='nearest')
                plt.colorbar(label='Average Attention Weight')
                plt.title('Combined Attention Map (Average of All Levels)')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(combined_filename, dpi=150, bbox_inches='tight')
                plt.close()
                
                logging.info(f'Combined attention map saved to {combined_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
