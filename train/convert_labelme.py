import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import zipfile

# --- ГЛАВНЫЕ НАСТРОЙКИ ---

# 1. Путь к вашему рабочему пространству с исходными данными
CVAT_WORKSPACE_PATH = r'F:\Desktop\SSCloud\CVAT_Workspace'

# 2. Пути, куда мы сложим подготовленные данные для U-Net
TARGET_IMGS_DIR = r'train\data\imgs'
TARGET_MASKS_DIR = r'train\data\masks'

# 3. Ваш словарь для сопоставления классов
CLASS_MAPPING = {
    '_background_': 0, "Здание": 6, "Земля": 2, "Растительность": 5, "Фон": 1,
    "Конструкции": 65, "Транспорт": 64, "Человек": 67, "Обстановка": 66,
}

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
    for obj in root.findall('object'):
        # Пропускаем удаленные объекты
        if obj.find('deleted') is not None and obj.find('deleted').text == '1':
            continue

        label_tag = obj.find('name')
        if label_tag is None:
            continue
        
        label = label_tag.text
        if label not in class_mapping:
            print(f"ВНИМАНИЕ: Класс '{label}' из XML не найден в CLASS_MAPPING. Пропускается.")
            continue

        class_id = class_mapping[label]
        
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

def process_data_from_zip():
    """
    Главная функция: находит ZIP-архивы, извлекает данные и создает датасет.
    """
    print("--- Начало подготовки данных из ZIP-архивов (формат LabelMe XML) ---")
    os.makedirs(TARGET_IMGS_DIR, exist_ok=True)
    os.makedirs(TARGET_MASKS_DIR, exist_ok=True)

    zip_archives = []
    # Ищем все папки *_normals
    for top_folder_name in sorted(os.listdir(CVAT_WORKSPACE_PATH)):
        if not top_folder_name.endswith('_normals'):
            continue
        
        source_dir = os.path.join(CVAT_WORKSPACE_PATH, top_folder_name, '5_upload_to_cvat')
        if not os.path.isdir(source_dir):
            continue

        for file in os.listdir(source_dir):
            if file.endswith('.zip'):
                zip_archives.append(os.path.join(source_dir, file))
                break 
    
    if not zip_archives:
        print("ОШИБКА: Не найдено ни одного ZIP-архива. Проверьте пути.")
        return

    print(f"Найдено {len(zip_archives)} ZIP-архивов. Начинаю обработку...")

    for zip_path in tqdm(zip_archives, desc="Обработка архивов"):
        try:
            with zipfile.ZipFile(zip_path, 'r') as archive:
                xml_filename = None
                img_filename = None
                for name in archive.namelist():
                    if name.endswith('.xml'):
                        xml_filename = name
                    elif name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_filename = name
                
                if not xml_filename or not img_filename:
                    print(f"\nПРЕДУПРЕЖДЕНИЕ: В архиве {os.path.basename(zip_path)} не найдена пара img/xml. Пропускается.")
                    continue

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

        except Exception as e:
            print(f"\nОшибка при обработке архива {os.path.basename(zip_path)}: {e}")
            
    print("\n--- Подготовка данных успешно завершена! ---")
    print(f"Все изображения скопированы в: {TARGET_IMGS_DIR}")
    print(f"Все маски созданы в: {TARGET_MASKS_DIR}")

if __name__ == '__main__':
    process_data_from_zip()