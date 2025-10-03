import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os

# --- Конфигурация модели ---
MODEL_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth" 
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_sam_model():
    """Загружает модель SAM в память."""
    print(f"Проверка устройства: {DEVICE}")
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Файл модели не найден по пути: {MODEL_CHECKPOINT_PATH}")
    
    print("Загрузка модели Segment Anything...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    print("Модель успешно загружена.")
    return sam

def create_mask_generator(sam_model):
    """Создает генератор масок с заданными параметрами из ноутбука."""
    print("Инициализация генератора масок...")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        box_nms_thresh=0.7,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )
    print("Генератор масок готов к работе.")
    return mask_generator

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
print("Инициализация модуля sam_loader...")
SAM_MODEL = load_sam_model()
MASK_GENERATOR = create_mask_generator(SAM_MODEL)
print("Модуль sam_loader полностью инициализирован.")