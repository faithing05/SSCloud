import os
import numpy as np
from PIL import Image
import pye57
import time
import laspy
import argparse
import json
import cv2 # Импортируем OpenCV

# Для запуска скрипта введите в командную строку следующее:
# python LAZformer.py Vistino20241014_E57\1.e57 annotations.json

# --- СЛОВАРЬ СООТВЕТСТВИЯ КЛАССОВ ---
# Сопоставление имен классов из вашего JSON-файла со стандартными и пользовательскими кодами формата LAS.
CLASS_MAPPING = {
    # --- Прямое соответствие стандартам ASPRS ---
    "Здание": 6,           # 6: Building
    "Земля": 2,            # 2: Ground
    "Растительность": 5,   # 5: High Vegetation
    "Фон": 1,              # 1: Unclassified

    # --- Пользовательские классы (диапазон 64-255) ---
    "Конструкции": 65,     
    "Транспорт": 64,       
    "Человек": 67,         
}


def save_points_to_laz(points, output_path, scale=0.001):
    """
    Сохранение структурированного массива точек в LAZ-файл
    """
    if len(points) == 0:
        print("Нет точек для сохранения.")
        return

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([scale, scale, scale])
    header.offsets = np.min(points['x']), np.min(points['y']), np.min(points['z'])

    las = laspy.LasData(header)

    las.x = points['x']
    las.y = points['y']
    las.z = points['z']
    las.intensity = points['intensity'].astype(np.uint16)
    las.red   = (points['normal_r'].astype(np.uint16) * 256)
    las.green = (points['normal_g'].astype(np.uint16) * 256)
    las.blue  = (points['normal_b'].astype(np.uint16) * 256)
    
    # Добавляем данные о классификации
    las.classification = points['classification'].astype(np.uint8)

    las.write(output_path)
    print(f"LAZ с классификацией сохранён: {output_path}")


def process_e57_with_classes(e57_path, annotation_path, laz_out_path):
    """
    Основная функция для обработки E57: чтение, вычисление нормалей,
    применение классификации из аннотаций и сохранение результата.
    """
    t0 = time.perf_counter()

    print("Чтение E57 файла...")
    e57 = pye57.E57(e57_path)
    scan_index = 0
    data = e57.read_scan(scan_index, intensity=True, row_column=True, ignore_missing_fields=True)
    X, Y, Z = data['cartesianX'], data['cartesianY'], data['cartesianZ']
    intensity, col, row = data['intensity'], data['columnIndex'], data['rowIndex']

    width = int(col.max()) + 1
    height = int(row.max()) + 1
    print(f"Размер сетки скана: {width}x{height}")

    print("Создание 2D-представления облака точек...")
    X_sum = np.zeros((height, width), dtype=np.float64)
    Y_sum = np.zeros((height, width), dtype=np.float64)
    Z_sum = np.zeros((height, width), dtype=np.float64)
    cnt = np.zeros((height, width), dtype=np.int32)
    np.add.at(X_sum, (row, col), X)
    np.add.at(Y_sum, (row, col), Y)
    np.add.at(Z_sum, (row, col), Z)
    np.add.at(cnt, (row, col), 1)
    mask = cnt > 0
    X_img = np.divide(X_sum, cnt, out=np.full_like(X_sum, np.nan), where=mask)
    Y_img = np.divide(Y_sum, cnt, out=np.full_like(Y_sum, np.nan), where=mask)
    Z_img = np.divide(Z_sum, cnt, out=np.full_like(Z_sum, np.nan), where=mask)
    
    # --- НОВЫЙ БЛОК: Загрузка и применение аннотаций ---
    print(f"Загрузка аннотаций из {annotation_path}...")
    with open(annotation_path, 'r') as f:
        annotations_data = json.load(f)

    # Создаем карту классов (пустую, с классом 0 - 'Created, never classified')
    classification_map = np.zeros((height, width), dtype=np.uint8)
    
    # Словарь для сопоставления ID категории из COCO с именем класса
    category_id_to_name = {cat['id']: cat['name'] for cat in annotations_data['categories']}

    # Рисуем полигоны на карте классов
    for ann in annotations_data['annotations']:
        category_id = ann['category_id']
        class_name = category_id_to_name[category_id]
        
        if class_name not in CLASS_MAPPING:
            print(f"ВНИМАНИЕ: Класс '{class_name}' из аннотаций не найден в CLASS_MAPPING. Пропускаем.")
            continue
        
        class_code = CLASS_MAPPING[class_name]
        
        # COCO хранит полигоны в виде [x1, y1, x2, y2, ...]. Преобразуем в формат для OpenCV.
        for seg in ann['segmentation']:
            poly = np.array(seg, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(classification_map, [poly], color=int(class_code))

    # ВАЖНО: Панорама для разметки была перевернута.
    # Переворачиваем карту классов, чтобы она соответствовала исходным данным.
    classification_map = np.flipud(classification_map)
    print("Карта классов создана.")
    # --- КОНЕЦ НОВОГО БЛОКА ---

    # Вычисление нормалей (этот блок можно пропустить, если они не нужны, но оставим)
    # ... (код для вычисления нормалей остается здесь без изменений) ...
    # Для простоты примера, создадим "пустые" нормали
    normal_r = np.full_like(classification_map, 128, dtype=np.uint8)
    normal_g = np.full_like(classification_map, 128, dtype=np.uint8)
    normal_b = np.full_like(classification_map, 128, dtype=np.uint8)

    # Формирование точек для LAZ
    final_mask = mask # Сохраняем все точки, у которых есть координаты
    
    n_pts = np.count_nonzero(final_mask)
    if n_pts == 0:
        print("Не найдено валидных точек для сохранения.")
        return

    pts = np.zeros(n_pts, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                                  ('intensity', 'u2'), ('normal_r', 'u1'),
                                  ('normal_g', 'u1'), ('normal_b', 'u1'),
                                  ('classification', 'u1')]) # <-- Добавили поле

    # Собираем данные только для валидных точек
    pts['x'] = X_img[final_mask]
    pts['y'] = Y_img[final_mask]
    pts['z'] = Z_img[final_mask]
    
    # Интенсивность тоже нужно усреднять
    intensity_sum = np.zeros((height, width), dtype=np.float64)
    np.add.at(intensity_sum, (row, col), intensity)
    intensity_img = np.divide(intensity_sum, cnt, out=np.zeros_like(intensity_sum), where=mask)
    pts['intensity'] = intensity_img[final_mask]
    
    pts['normal_r'] = normal_r[final_mask]
    pts['normal_g'] = normal_g[final_mask]
    pts['normal_b'] = normal_b[final_mask]
    
    # Присваиваем классификацию точкам
    pts['classification'] = classification_map[final_mask]

    save_points_to_laz(pts, laz_out_path)
    print(f"Общее время: {time.perf_counter() - t0:.3f} с")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Конвертирует E57 в LAZ, добавляя классификацию из аннотаций CVAT.")
    parser.add_argument("e57_path", type=str, help="Путь к исходному файлу .e57")
    parser.add_argument("annotation_path", type=str, help="Путь к файлу .json с аннотациями (экспорт из CVAT в формате COCO)")
    args = parser.parse_args()

    # Формируем имя выходного файла
    base_name = os.path.splitext(args.e57_path)[0]
    laz_out_path = base_name + "_classified.laz"

    process_e57_with_classes(args.e57_path, args.annotation_path, laz_out_path)