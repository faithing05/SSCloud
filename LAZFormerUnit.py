import os
import numpy as np
import pye57
import time
import laspy
import cv2
import xml.etree.ElementTree as ET
from sklearn.cluster import DBSCAN
import glob
import argparse 

# --- ЗАПУСК ---
# python LAZformerUnit.py - обработка всех файлов по основным директориям
# python LAZformerUnit.py 6 - обработка определенного файла по основным директориям

# --- ПУТИ К ОСНОВНЫМ ДИРЕКТОРИЯМ ---
E57_BASE_DIR = 'Vistino20241014_E57'
CVAT_BASE_DIR = 'CVAT_Workspace'


# --- СЛОВАРЬ СООТВЕТСТВИЯ КЛАССОВ ---
CLASS_MAPPING = {
    "Здание": 6, "Земля": 2, "Растительность": 5, "Фон": 1,
    "Конструкции": 65, "Транспорт": 64, "Человек": 67, "Обстановка": 66,
}



def filter_small_clusters(points, voxel_size=0.10, dbscan_eps=0.5, min_cluster_points=1000):
    """
    Удаляет маленькие, шумовые кластеры, сохраняя все крупные и средние.
    
    :param points: Исходный массив точек в полном разрешении.
    :param voxel_size: Размер вокселя для прореживания.
    :param dbscan_eps: Параметр eps для DBSCAN.
    :param min_cluster_points: Минимальное количество точек в прореженном кластере,
                               чтобы он считался "значимым" и был сохранен.
    :return: Отфильтрованный массив точек в ПОЛНОМ разрешении.
    """
    if len(points) == 0:
        return points

    print(f"\nЗапуск фильтрации шумовых кластеров...")
    print(f"Размер вокселя: {voxel_size} м, Мин. точек в кластере: {min_cluster_points}")

    # --- Шаг 1: Воксельное прореживание (как и раньше) ---
    xyz = np.vstack((points['x'], points['y'], points['z'])).transpose()
    voxel_indices = np.floor(xyz / voxel_size).astype(int)
    unique_voxel_indices, unique_inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    
    downsampled_points = np.zeros((len(unique_voxel_indices), 3))
    np.add.at(downsampled_points, unique_inverse_indices, xyz)
    counts = np.bincount(unique_inverse_indices)
    downsampled_points /= counts[:, np.newaxis]

    print(f"Количество точек уменьшено с {len(points)} до {len(downsampled_points)} для анализа.")

    # --- Шаг 2: DBSCAN на прореженных данных ---
    print(f"Запуск DBSCAN (eps={dbscan_eps})...")
    db = DBSCAN(eps=dbscan_eps, min_samples=10, n_jobs=-1).fit(downsampled_points)
    labels = db.labels_

    # --- Шаг 3: Анализ размеров кластеров и фильтрация ---
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0:
        print("ВНИМАНИЕ: DBSCAN не нашел ни одного кластера. Возвращаем исходные точки.")
        return points
        
    # НАХОДИМ ID ВСЕХ КЛАСТЕРОВ, КОТОРЫЕ БОЛЬШЕ ЗАДАННОГО ПОРОГА
    significant_cluster_labels = unique_labels[counts >= min_cluster_points]
    
    if len(significant_cluster_labels) == 0:
        print(f"ВНИМАНИЕ: Не найдено ни одного кластера размером > {min_cluster_points}. Возможно, порог слишком велик.")
        # В этом случае, чтобы не удалить всё, оставляем хотя бы самый большой кластер
        significant_cluster_labels = [unique_labels[counts.argmax()]]
        print(f"Сохраняем только самый большой кластер, так как других крупных нет.")

    print(f"Найдено {len(significant_cluster_labels)} значимых кластеров (из {len(unique_labels)} всего).")

    # Создаем маску для "хороших" вокселей, принадлежащих ЛЮБОМУ из значимых кластеров
    good_voxel_mask = np.isin(labels, significant_cluster_labels)
    good_unique_voxels = unique_voxel_indices[good_voxel_mask]
    good_voxels_set = {tuple(v) for v in good_unique_voxels}

    # Фильтруем исходные точки
    final_mask = np.array([tuple(v) in good_voxels_set for v in voxel_indices], dtype=bool)

    num_kept = np.sum(final_mask)
    num_removed = len(points) - num_kept
    print(f"Фильтрация завершена. Сохранено {num_kept} точек. Удалено {num_removed} (шум и мелкие кластеры).")

    return points[final_mask]

def add_diagnostic_prints(points, title="Статистика облака точек"):
    if len(points) == 0: print(f"--- {title}: Облако пустое ---"); return
    print(f"\n--- {title} ---"); print(f"Количество точек: {len(points)}")
    for axis in ['x', 'y', 'z']:
        coords = points[axis]
        print(f"Координаты {axis.upper()}: min={np.min(coords):.3f}, max={np.max(coords):.3f}, медиана={np.median(coords):.3f}")
    print("----------------------------------")

def parse_labelme_xml(xml_path):
    annotations = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            if obj.find('deleted').text == '1': continue
            class_name = obj.find('name').text.strip()
            for polygon_element in obj.findall('polygon'):
                points = [[int(pt.find('x').text), int(pt.find('y').text)] for pt in polygon_element.findall('pt')]
                if points: annotations.append({'class_name': class_name, 'polygon': points})
    except (ET.ParseError, FileNotFoundError) as e: print(f"Ошибка при работе с XML: {e}")
    return annotations

def save_points_to_laz(points, output_path, scale=0.001):
    if len(points) == 0: print("Нет точек для сохранения."); return
    # Убедимся, что директория для сохранения существует
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория: {output_dir}")

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([scale, scale, scale])
    header.offsets = np.min(points['x']), np.min(points['y']), np.min(points['z'])
    las = laspy.LasData(header)
    las.x, las.y, las.z = points['x'], points['y'], points['z']
    las.intensity = points['intensity'].astype(np.uint16)
    las.red = las.green = las.blue = np.full(len(points), 128 * 256, dtype=np.uint16)
    las.classification = points['classification'].astype(np.uint8)
    las.write(output_path)
    print(f"\nLAZ с классификацией сохранён: {output_path}")


def process_e57_with_classes(e57_path, annotation_path, laz_out_path):
    print(f"\n{'='*80}\n--- Начинаю обработку файла: {os.path.basename(e57_path)} ---\n{'='*80}")
    t0 = time.perf_counter()
    print("Чтение E57 файла...")
    try:
        e57 = pye57.E57(e57_path)
        data = e57.read_scan(0, intensity=True, row_column=True, ignore_missing_fields=True)
    except Exception as e:
        print(f"Ошибка при чтении E57 файла {e57_path}: {e}")
        return

    X, Y, Z, intensity, col, row = data['cartesianX'], data['cartesianY'], data['cartesianZ'], data['intensity'], data['columnIndex'], data['rowIndex']
    width, height = int(col.max()) + 1, int(row.max()) + 1
    print(f"Размер сетки скана: {width}x{height}")

    print("Создание 2D-представления облака точек...")
    cnt = np.zeros((height, width), dtype=np.int32); np.add.at(cnt, (row, col), 1)
    mask = cnt > 0
    def create_img(values):
        sum_val = np.zeros((height, width), dtype=np.float64)
        np.add.at(sum_val, (row, col), values)
        return np.divide(sum_val, cnt, out=np.full_like(sum_val, np.nan), where=mask)
    X_img, Y_img, Z_img = create_img(X), create_img(Y), create_img(Z)
    
    print(f"Загрузка аннотаций из {annotation_path}..."); 
    annotations = parse_labelme_xml(annotation_path)
    if not annotations: 
        print(f"ВНИМАНИЕ: Аннотации не найдены для файла {os.path.basename(e57_path)}. Пропускаю.")
        return

    classification_map = np.zeros((height, width), dtype=np.uint8)
    print("Применение аннотаций к облаку точек...")
    annotations_by_class = {cls: [] for cls in CLASS_MAPPING}
    for ann in annotations:
        if ann['class_name'] in annotations_by_class: annotations_by_class[ann['class_name']].append(ann)
    DRAW_ORDER = ["Фон", "Земля", "Здание", "Растительность", "Обстановка", "Конструкции", "Транспорт", "Человек"]
    for class_name in DRAW_ORDER:
        if class_name in annotations_by_class and class_name in CLASS_MAPPING:
            class_code = CLASS_MAPPING[class_name]
            for ann in annotations_by_class[class_name]:
                cv2.fillPoly(classification_map, [np.array(ann['polygon'], dtype=np.int32)], color=int(class_code))

    classification_map = np.flipud(classification_map)
    
    n_pts = np.count_nonzero(mask)
    if n_pts == 0: 
        print("Не найдено валидных точек. Пропускаю.")
        return
        
    pts = np.zeros(n_pts, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('intensity', 'u2'), ('classification', 'u1')])
    pts['x'], pts['y'], pts['z'] = X_img[mask], Y_img[mask], Z_img[mask]
    pts['intensity'] = create_img(intensity)[mask]; pts['classification'] = classification_map[mask]

    # --- Финальный фильтр --- 
    # Главный параметр для настройки - min_cluster_points
    add_diagnostic_prints(pts, "Статистика ДО фильтрации")
    cleaned_pts = filter_small_clusters(pts, voxel_size=0.10, min_cluster_points=10)
    add_diagnostic_prints(cleaned_pts, "Статистика ПОСЛЕ фильтрации")
    # -----------------------------------------------

    save_points_to_laz(cleaned_pts, laz_out_path)
    print(f"Обработка {os.path.basename(e57_path)} завершена. Общее время: {time.perf_counter() - t0:.3f} с")


if __name__ == '__main__':
    # 1. Возвращаем argparse и делаем аргумент необязательным
    parser = argparse.ArgumentParser(
        description="Конвертирует E57 в LAZ с классификацией. Обрабатывает либо один указанный скан, либо все найденные."
    )
    # 'nargs'='?' делает позиционный аргумент необязательным. 
    # Если он не указан, его значение будет None.
    parser.add_argument(
        "scan_id", 
        type=str, 
        nargs='?', 
        default=None,
        help="Опционально: Порядковый номер скана для обработки (например, '6'). Если не указан, обрабатываются все файлы."
    )
    args = parser.parse_args()

    files_to_process = []

    # 2. Логика выбора режима работы
    if args.scan_id:
        # --- РЕЖИМ ОБРАБОТКИ ОДНОГО ФАЙЛА ---
        print(f"--- РЕЖИМ: Обработка одного скана с ID: {args.scan_id} ---")
        # Собираем полный путь к файлу
        e57_path = os.path.join(E57_BASE_DIR, f"{args.scan_id}.e57")
        # Проверяем, существует ли такой файл
        if not os.path.exists(e57_path):
            print(f"ОШИБКА: Файл не найден по пути: {e57_path}")
            exit() # Выходим из скрипта, если файл не найден
        files_to_process.append(e57_path)

    else:
        # --- РЕЖИМ ОБРАБОТКИ ВСЕХ ФАЙЛОВ ---
        print("--- РЕЖИМ: Обработка всех .e57 файлов ---")
        e57_files = glob.glob(os.path.join(E57_BASE_DIR, '**', '*.e57'), recursive=True)
        
        if not e57_files:
            print(f"Не найдено ни одного .e57 файла в директории {E57_BASE_DIR}")
            exit()

        # Функция для числовой сортировки
        def get_number_from_path(path):
            base_name = os.path.basename(path)
            number_str = os.path.splitext(base_name)[0]
            try: return int(number_str)
            except ValueError: return float('inf')

        e57_files.sort(key=get_number_from_path)
        files_to_process = e57_files
        print("Файлы будут обработаны в следующем порядке:")
        for f in files_to_process:
            print(f" - {os.path.basename(f)}")


    # 3. Единый цикл обработки, который работает с подготовленным списком
    for e57_path in files_to_process:
        base_name = os.path.splitext(os.path.basename(e57_path))[0]
        
        annotation_path = os.path.join(
            CVAT_BASE_DIR,
            f"{base_name}_normals",
            "4_labelme_xml",
            f"{base_name}_normals.xml"
        )
        
        output_dir = os.path.dirname(e57_path) or '.'
        laz_out_path = os.path.join(output_dir, base_name + "_classified.laz")
        
        if not os.path.exists(annotation_path):
            print(f"ПРЕДУПРЕЖДЕНИЕ: Для файла {e57_path} не найден файл аннотации: {annotation_path}. Пропускаем.")
            continue

        process_e57_with_classes(e57_path, annotation_path, laz_out_path)

    print("\n\nОбработка завершена.")