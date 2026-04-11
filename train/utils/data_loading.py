import logging
import warnings
import numpy as np
import torch
from PIL import Image
from multiprocessing import Pool
from os.path import splitext
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


warnings.filterwarnings('once', category=Image.DecompressionBombWarning)


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(mask_path):
    mask = np.asarray(load_image(mask_path))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        image_paths = [path for path in self.images_dir.iterdir() if path.is_file() and not path.name.startswith('.')]
        mask_paths = [path for path in self.mask_dir.iterdir() if path.is_file() and not path.name.startswith('.')]

        image_map = {path.stem: path for path in image_paths}
        mask_map = {}
        for path in mask_paths:
            stem = path.stem
            if self.mask_suffix and stem.endswith(self.mask_suffix):
                stem = stem[:-len(self.mask_suffix)]
            mask_map[stem] = path

        self.ids = sorted([image_id for image_id in image_map.keys() if image_id in mask_map])
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        self.image_files = {image_id: image_map[image_id] for image_id in self.ids}
        self.mask_files = {image_id: mask_map[image_id] for image_id in self.ids}

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(unique_mask_values, [self.mask_files[image_id] for image_id in self.ids]),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]

        if name not in self.image_files or name not in self.mask_files:
            raise RuntimeError(f'Image/mask pair was not indexed for ID {name}')

        mask = load_image(self.mask_files[name])
        img = load_image(self.image_files[name])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
