import os
import numpy as np
import pye57
import time
import laspy
import json
import cv2
import traceback
import zipfile
import xml.etree.ElementTree as ET

# --- СЛОВАРЬ СООТВЕТСТВИЯ КЛАССОВ ---
CLASS_MAPPING = {
    "Здание": 6, "Земля": 2, "Растительность": 5, "Фон": 1,
    "Конструкции": 65, "Транспорт": 64, "Человек": 67, "Обстановка": 66,
}

def save_points_to_laz(points, output_path, scale=0.001):
    """Сохранение структурированного массива точек в LAZ-файл."""
    if len(points) == 0:
        print("Нет точек для сохранения.")
        return
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([scale, scale, scale])
    header.offsets = np.min(points['x']), np.min(points['y']), np.min(points['z'])
    las = laspy.LasData(header)
    las.x, las.y, las.z = points['x'], points['y'], points['z']
    las.intensity = points['intensity'].astype(np.uint16)
    las.red, las.green, las.blue = (points['normal_r'].astype(np.uint16) * 256), (points['normal_g'].astype(np.uint16) * 256), (points['normal_b'].astype(np.uint16) * 256)
    las.classification = points['classification'].astype(np.uint8)
    las.write(output_path)
    print(f"LAZ с классификацией сохранён: {output_path}")

def find_and_parse_labelme_xml_from_zip(root_dir):
    """
    Находит ZIP-архив в директории, извлекает из него XML-файл
    и парсит его как LabelMe аннотацию.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.zip'):
                zip_path = os.path.join(subdir, file)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for filename in zf.namelist():
                            if filename.lower().endswith('.xml'):
                                print(f"  Найден XML '{filename}' внутри '{os.path.basename(zip_path)}'")
                                xml_content = zf.read(filename)
                                return ET.fromstring(xml_content)
                except Exception as e:
                    print(f"  ОШИБКА: не удалось прочитать ZIP-архив {zip_path}: {e}")
                    return None
    return None

def process_e57_with_labelme_classes(e57_path, annotation_folder, laz_out_path):
    """
    Основная функция для обработки E57, применяя классы из LabelMe XML,
    найденного в ZIP-архиве.
    """
    t0 = time.perf_counter()
    print("Чтение E57 файла...")
    e57 = pye57.E57(e57_path)
    data = e57.read_scan(0, intensity=True, row_column=True, ignore_missing_fields=True)
    X, Y, Z, intensity, col, row = data['cartesianX'], data['cartesianY'], data['cartesianZ'], data['intensity'], data['columnIndex'], data['rowIndex']
    width, height = int(col.max()) + 1, int(row.max()) + 1
    print(f"Размер сетки скана: {width}x{height}")

    print("Создание 2D-представления облака точек...")
    X_sum, Y_sum, Z_sum = np.zeros((height, width), dtype=np.float64), np.zeros((height, width), dtype=np.float64), np.zeros((height, width), dtype=np.float64)
    cnt = np.zeros((height, width), dtype=np.int32)
    np.add.at(X_sum, (row, col), X); np.add.at(Y_sum, (row, col), Y); np.add.at(Z_sum, (row, col), Z)
    np.add.at(cnt, (row, col), 1)
    mask = cnt > 0
    X_img, Y_img, Z_img = [np.divide(s, cnt, out=np.full_like(s, np.nan), where=mask) for s in [X_sum, Y_sum, Z_sum]]

    print(f"Загрузка аннотаций из папки {os.path.basename(annotation_folder)}...")
    labelme_root = find_and_parse_labelme_xml_from_zip(annotation_folder)
    if labelme_root is None:
        print("ОШИБКА: Не удалось найти или прочитать XML-файл с аннотациями. Обработка прервана.")
        return

    classification_map = np.zeros((height, width), dtype=np.uint8)

    # --- Парсинг LabelMe XML ---
    for obj in labelme_root.findall('object'):
        class_name_tag = obj.find('name')
        if class_name_tag is None: continue
        class_name = class_name_tag.text

        if class_name not in CLASS_MAPPING:
            print(f"ВНИМАНИЕ: Класс '{class_name}' из XML не найден в CLASS_MAPPING. Пропускаем.")
            continue
        class_code = CLASS_MAPPING[class_name]

        polygon_tag = obj.find('polygon')
        if polygon_tag is None: continue
        
        points = []
        for pt in polygon_tag.findall('pt'):
            try:
                x = int(float(pt.find('x').text))
                y = int(float(pt.find('y').text))
                points.append([x, y])
            except (ValueError, AttributeError):
                continue # Пропускаем некорректные точки
        
        if points:
            poly = np.array(points, dtype=np.int32)
            cv2.fillPoly(classification_map, [poly], color=int(class_code))

    classification_map = np.flipud(classification_map)
    print("Карта классов создана.")
    
    normal_r, normal_g, normal_b = [np.full_like(classification_map, 128, dtype=np.uint8) for _ in range(3)]

    n_pts = np.count_nonzero(mask)
    if n_pts == 0:
        print("Не найдено валидных точек для сохранения.")
        return

    pts = np.zeros(n_pts, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('intensity', 'u2'), ('normal_r', 'u1'), ('normal_g', 'u1'), ('normal_b', 'u1'), ('classification', 'u1')])
    pts['x'], pts['y'], pts['z'] = X_img[mask], Y_img[mask], Z_img[mask]
    
    intensity_sum = np.zeros((height, width), dtype=np.float64)
    np.add.at(intensity_sum, (row, col), intensity)
    intensity_img = np.divide(intensity_sum, cnt, out=np.zeros_like(intensity_sum), where=mask)
    pts['intensity'] = intensity_img[mask]
    
    pts['normal_r'], pts['normal_g'], pts['normal_b'] = normal_r[mask], normal_g[mask], normal_b[mask]
    pts['classification'] = classification_map[mask]

    save_points_to_laz(pts, laz_out_path)
    print(f"Общее время: {time.perf_counter() - t0:.3f} с")

def main_auto_process_separated_folders():
    """Автоматически сопоставляет E57 файлы и аннотации из ZIP/XML и запускает конвертацию."""
    e57_folder_path = r'F:\Desktop\SSCloud\Vistino20241014_E57'
    annotations_base_path = r'F:\Desktop\SSCloud\CVAT_Workspace'

    print("Запуск автоматической обработки с разделенными папками (ZIP/XML)...")
    print(f"  Папка с E57: {e57_folder_path}")
    print(f"  Папка с аннотациями: {annotations_base_path}")

    for i in range(1, 101):
        e57_path = os.path.join(e57_folder_path, f'{i}.e57')
        annotation_folder = os.path.join(annotations_base_path, f'{i}_normals')
        
        if not os.path.exists(e57_path):
            if i > 25: break
            continue
        
        if not os.path.isdir(annotation_folder):
            print(f"\n--- Пропуск индекса {i}: E57 файл найден, но папка аннотаций '{os.path.basename(annotation_folder)}' отсутствует.")
            continue
            
        print(f"\n--- Обработка пары файлов для индекса {i} ---")
        print(f"  E57 файл: {e57_path}")
        
        base_name = os.path.splitext(os.path.basename(e57_path))[0]
        laz_out_path = os.path.join(e57_folder_path, f"{base_name}_classified.laz")

        try:
            # Передаем всю папку, а функция сама найдет внутри ZIP и XML
            process_e57_with_labelme_classes(e57_path, annotation_folder, laz_out_path)
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при обработке {os.path.basename(e57_path)}: {e}")
            traceback.print_exc()

    print("\n--- Автоматическая обработка завершена ---")

if __name__ == '__main__':
    main_auto_process_separated_folders()