import os
import shutil
import argparse
import logging
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import zipfile

# Запускаем командой:
# python train/convert_labelme.py [имя_архива.zip]
# Если имя архива не указано, будут обработаны все найденные архивы в папке Data_Output.

# --- ГЛАВНЫЕ НАСТРОЙКИ ---

# 1. Путь к вашему рабочему пространству с исходными данными
CVAT_WORKSPACE_PATH = r'F:\Desktop\SSCloud\Data_Output'

# 2. Пути, куда мы сложим подготовленные данные для U-Net
TARGET_IMGS_DIR = r'train\data\imgs'
TARGET_MASKS_DIR = r'train\data\masks'

# 3. Словарь для сопоставления классов
# Объединения:
# - _background_ и Фон -> 0
# - Транспорт, Человек и Обстановка -> 66
CLASS_MAPPING = {
    '_background_': 0,
    'background': 0,
    'Фон': 0,
    'Земля': 2,
    'Растительность': 5,
    'Здание': 6,
    'Конструкции': 65,
    'Транспорт': 66,
    'Человек': 66,
    'Обстановка': 66,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
LOGGER = logging.getLogger(__name__)


def _normalize_label(label):
    """
    Нормализует имя класса: обрезает пробелы, схлопывает внутренние пробелы и приводит к нижнему регистру.
    """
    if label is None:
        return ''

    return ' '.join(label.strip().split()).lower()

def parse_labelme_xml(xml_content, class_mapping):
    """
    Парсит XML-файл формата LabelMe и возвращает NumPy array с маской.
    """
    root = ET.fromstring(xml_content)
    
    # --- Извлекаем размеры изображения ---
    imagesize_tag = root.find('imagesize')
    if imagesize_tag is None:
        raise ValueError("Тег <imagesize> не найден в XML.")
    
    height_tag = imagesize_tag.find('nrows')
    width_tag = imagesize_tag.find('ncols')
    
    if height_tag is None or width_tag is None:
        raise ValueError("Теги <nrows> или <ncols> не найдены в XML.")
        
    height = int(height_tag.text)
    width = int(width_tag.text)
    
    # Создаем пустую маску (холст)
    mask = np.zeros((height, width), dtype=np.uint8)

    # --- Находим все объекты (полигоны) и рисуем их ---
    normalized_class_mapping = {
        _normalize_label(name): class_id
        for name, class_id in class_mapping.items()
    }

    for obj in root.findall('object'):
        # Пропускаем удаленные объекты
        if obj.find('deleted') is not None and obj.find('deleted').text == '1':
            continue

        label_tag = obj.find('name')
        if label_tag is None:
            continue
        
        label = label_tag.text
        normalized_label = _normalize_label(label)

        if normalized_label not in normalized_class_mapping:
            LOGGER.warning(
                "Класс '%s' из XML не найден в CLASS_MAPPING. Пропускается.",
                label,
            )
            continue

        class_id = normalized_class_mapping[normalized_label]
        
        polygon_tag = obj.find('polygon')
        if polygon_tag is None:
            continue
            
        points = []
        for pt in polygon_tag.findall('pt'):
            x = int(float(pt.find('x').text))
            y = int(float(pt.find('y').text))
            points.append((x, y))

        # Рисуем полигон на временном изображении
        temp_img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(temp_img).polygon(points, outline=class_id, fill=class_id)
        
        # "Накладываем" нарисованный полигон на основную маску
        mask = np.maximum(mask, np.array(temp_img))

    return mask

def _find_zip_archives(archive_name=None):
    """
    Ищет ZIP-архивы в рабочих папках CVAT.

    Если передан archive_name, возвращает только архивы с таким именем.
    """
    zip_archives = []

    # Если передан абсолютный/относительный путь к архиву - используем его напрямую.
    if archive_name is not None:
        candidate_path = archive_name
        if not os.path.isabs(candidate_path):
            candidate_path = os.path.abspath(candidate_path)

        if os.path.isfile(candidate_path) and candidate_path.lower().endswith('.zip'):
            return [candidate_path]

        # Частый случай: архив лежит прямо в CVAT_WORKSPACE_PATH.
        workspace_candidate = os.path.join(CVAT_WORKSPACE_PATH, archive_name)
        if os.path.isfile(workspace_candidate) and workspace_candidate.lower().endswith('.zip'):
            return [workspace_candidate]

        # Если передано только имя, ищем это имя в корне рабочего пространства.
        root_candidate = os.path.join(CVAT_WORKSPACE_PATH, os.path.basename(archive_name))
        if os.path.isfile(root_candidate) and root_candidate.lower().endswith('.zip'):
            return [root_candidate]

    # Ищем ZIP в корне рабочего пространства.
    for file in sorted(os.listdir(CVAT_WORKSPACE_PATH)):
        if not file.lower().endswith('.zip'):
            continue

        if archive_name is not None and file != archive_name:
            continue

        zip_archives.append(os.path.join(CVAT_WORKSPACE_PATH, file))

    for top_folder_name in sorted(os.listdir(CVAT_WORKSPACE_PATH)):
        if not top_folder_name.endswith('_normals'):
            continue

        source_dir = os.path.join(CVAT_WORKSPACE_PATH, top_folder_name, '5_upload_to_cvat')
        if not os.path.isdir(source_dir):
            continue

        for file in sorted(os.listdir(source_dir)):
            if not file.lower().endswith('.zip'):
                continue

            if archive_name is not None and file != archive_name:
                continue

            zip_archives.append(os.path.join(source_dir, file))

    # Убираем дубликаты, сохраняя порядок.
    return list(dict.fromkeys(zip_archives))


def _build_xml_image_pairs(archive_names):
    """
    Формирует пары (xml, image) внутри одного ZIP.

    Правило соответствия: XML и изображение должны иметь одинаковое имя без расширения.
    """
    image_by_base = {}
    xml_files = []

    for name in sorted(archive_names):
        if name.endswith('/'):
            continue

        file_name = os.path.basename(name)
        base_name, ext = os.path.splitext(file_name)
        ext = ext.lower()

        if ext in ('.png', '.jpg', '.jpeg'):
            image_by_base[base_name.lower()] = name
        elif ext == '.xml':
            xml_files.append(name)

    pairs = []
    for xml_name in xml_files:
        xml_base_name = os.path.splitext(os.path.basename(xml_name))[0]
        matched_image = image_by_base.get(xml_base_name.lower())
        if matched_image:
            pairs.append((xml_name, matched_image))

    # Фолбэк для архивов, где только один XML и одно изображение без совпадающих имен.
    if not pairs and len(xml_files) == 1 and len(image_by_base) == 1:
        only_xml = xml_files[0]
        only_image = next(iter(image_by_base.values()))
        pairs.append((only_xml, only_image))

    return pairs


def process_data_from_zip(archive_name=None):
    """
    Главная функция: находит ZIP-архивы, извлекает данные и создает датасет.
    """
    LOGGER.info("Процесс начался: подготовка данных из ZIP-архивов (LabelMe XML)")
    os.makedirs(TARGET_IMGS_DIR, exist_ok=True)
    os.makedirs(TARGET_MASKS_DIR, exist_ok=True)

    zip_archives = _find_zip_archives(archive_name)
    
    if not zip_archives:
        if archive_name is None:
            LOGGER.error("Не найдено ни одного ZIP-архива. Проверьте пути.")
        else:
            LOGGER.error("Архив '%s' не найден. Проверьте имя и пути.", archive_name)
        return

    LOGGER.info("Найдено %d ZIP-архивов. Идет обработка...", len(zip_archives))

    processed_count = 0
    skipped_count = 0
    error_count = 0
    processed_archives = 0
    skipped_archives = 0

    progress_bar = tqdm(
        zip_archives,
        desc="Обработка архивов",
        unit="архив",
        total=len(zip_archives),
    )

    for zip_path in progress_bar:
        try:
            with zipfile.ZipFile(zip_path, 'r') as archive:
                file_pairs = _build_xml_image_pairs(archive.namelist())

                if not file_pairs:
                    LOGGER.warning(
                        "В архиве %s не найдены валидные пары xml/img. Пропускается.",
                        os.path.basename(zip_path),
                    )
                    skipped_archives += 1
                    skipped_count += 1
                    progress_bar.set_postfix(processed=processed_count, skipped=skipped_count, errors=error_count)
                    continue

                per_archive_progress = tqdm(
                    file_pairs,
                    desc=f"Файлы {os.path.basename(zip_path)}",
                    unit="файл",
                    leave=False,
                    total=len(file_pairs),
                )

                for xml_filename, img_filename in per_archive_progress:
                    # 1. Извлекаем изображение и сохраняем его
                    with archive.open(img_filename) as img_file:
                        target_img_path = os.path.join(TARGET_IMGS_DIR, os.path.basename(img_filename))
                        with open(target_img_path, 'wb') as f:
                            shutil.copyfileobj(img_file, f)

                    # 2. Извлекаем XML, парсим и создаем маску
                    xml_content = archive.read(xml_filename)
                    mask_array = parse_labelme_xml(xml_content, CLASS_MAPPING)
                    mask_image = Image.fromarray(mask_array.astype(np.uint8))

                    # 3. Сохраняем маску с тем же именем, что и изображение
                    base_name = os.path.splitext(os.path.basename(img_filename))[0]
                    mask_save_path = os.path.join(TARGET_MASKS_DIR, base_name + '.png')
                    mask_image.save(mask_save_path)
                    processed_count += 1

                    progress_bar.set_postfix(processed=processed_count, skipped=skipped_count, errors=error_count)

                processed_archives += 1

        except Exception as e:
            LOGGER.error("Ошибка при обработке архива %s: %s", os.path.basename(zip_path), e)
            error_count += 1
            progress_bar.set_postfix(processed=processed_count, skipped=skipped_count, errors=error_count)
            
    LOGGER.info("Процесс завершен")
    LOGGER.info(
        "Архивы: успешно=%d, пропущено=%d, всего=%d",
        processed_archives,
        skipped_archives,
        len(zip_archives),
    )
    LOGGER.info("Файлы: успешно=%d, пропущено=%d, ошибок=%d", processed_count, skipped_count, error_count)
    LOGGER.info("Все изображения скопированы в: %s", TARGET_IMGS_DIR)
    LOGGER.info("Все маски созданы в: %s", TARGET_MASKS_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Подготовка датасета из ZIP-архивов LabelMe XML.'
    )
    parser.add_argument(
        'archive_name',
        nargs='?',
        default=None,
        help='Имя ZIP-архива (например: upload_to_cvat_labelme.zip). Если не указано, обрабатываются все архивы.'
    )
    args = parser.parse_args()

    process_data_from_zip(args.archive_name)
