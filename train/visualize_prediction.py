import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ИСПОЛЬЗОВАНИЕ:
# python _visualize_prediction.py                          (обработать все маски в папке)
# python _visualize_prediction.py -f predicted_mask.png    (обработать одну конкретную маску)

# --- ГЛАВНЫЕ НАСТРОЙКИ ---

# 1. Папка, где лежат предсказанные серые маски и куда сохранятся результаты
IO_DIR = 'data/test/output'

# 2. Ваш словарь классов (для названий в легенде)
CLASS_MAPPING = {
    "Здание": 6, "Земля": 2, "Растительность": 5, "Фон": 1,
    "Конструкции": 65, "Транспорт": 64, "Человек": 67, "Обстановка": 66,
    "_background_": 0,
}

# 3. Палитра цветов (RGB). Каждому ID класса - свой цвет.
COLOR_PALETTE = {
    0: (0, 0, 0),          # _background_ (черный)
    6: (255, 0, 0),        # Здание (красный)
    2: (0, 128, 0),        # Земля (темно-зеленый)
    5: (34, 139, 34),      # Растительность (зеленый)
    1: (128, 128, 128),    # Фон (серый)
    65: (255, 255, 0),     # Конструкции (желтый)
    64: (0, 255, 255),     # Транспорт (бирюзовый)
    67: (255, 0, 255),     # Человек (пурпурный)
    66: (255, 165, 0),     # Обстановка (оранжевый)
}

def colorize_mask(input_path, output_path):
    """Раскрашивает одну серую маску в соответствии с палитрой."""
    try:
        gray_mask_img = Image.open(input_path)
        gray_mask = np.array(gray_mask_img)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл не найден! {input_path}")
        return False
    
    color_mask = np.zeros((gray_mask.shape[0], gray_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in COLOR_PALETTE.items():
        color_mask[gray_mask == class_id] = color

    Image.fromarray(color_mask).save(output_path)
    return True

def create_legend(output_path):
    """Создает изображение с легендой классов и цветов."""
    box_size, padding, font_size, text_offset = 40, 20, 20, 10
    classes_to_show = {name: cid for name, cid in CLASS_MAPPING.items() if name != '_background_'}
    img_height = (len(classes_to_show) * (box_size + padding)) + padding
    img_width = 400
    
    legend_img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(legend_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Шрифт Arial не найден, используется дефолтный шрифт.")
        font = ImageFont.load_default()

    y_pos = padding
    for class_name, class_id in sorted(classes_to_show.items(), key=lambda item: item[0]):
        color = COLOR_PALETTE.get(class_id, (0, 0, 0))
        draw.rectangle([padding, y_pos, padding + box_size, y_pos + box_size], fill=color, outline='black')
        text_position = (padding + box_size + text_offset, y_pos + (box_size - font_size) // 2)
        draw.text(text_position, f"{class_name} (ID: {class_id})", fill='black', font=font)
        y_pos += (box_size + padding)
        
    legend_img.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Раскрашивает предсказанные маски и создает для них легенды.')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='Имя файла конкретной маски для обработки (например, "predicted_mask.png"). '
                             'Если не указан, обрабатываются все маски в папке.')
    args = parser.parse_args()

    os.makedirs(IO_DIR, exist_ok=True)

    if args.filename:
        # --- РЕЖИМ ОБРАБОТКИ ОДНОЙ МАСКИ ---
        print(f"Обрабатываю маску: {args.filename}")
        
        input_file = os.path.join(IO_DIR, args.filename)
        base_name = os.path.splitext(args.filename)[0]
        
        if colorize_mask(input_file, os.path.join(IO_DIR, f"{base_name}_COLOR.png")):
            create_legend(os.path.join(IO_DIR, f"{base_name}_legend.png"))
            print(f"Готово! Результаты для '{base_name}' сохранены.")

    else:
        # --- РЕЖИМ ОБРАБОТКИ ВСЕХ МАСОК ---
        print(f"Обрабатываю все .png маски в папке: {IO_DIR}")
        
        mask_files = [f for f in os.listdir(IO_DIR) if f.endswith('.png') and '_COLOR' not in f and '_legend' not in f]
        
        if not mask_files:
            print("Масок для обработки не найдено.")
        else:
            # Создаем одну общую легенду для всех
            legend_path = os.path.join(IO_DIR, "legend.png")
            create_legend(legend_path)
            print(f"Общая легенда сохранена: {legend_path}")

            for filename in tqdm(mask_files, desc="Раскрашивание масок"):
                base_name = os.path.splitext(filename)[0]
                input_path = os.path.join(IO_DIR, filename)
                output_color_path = os.path.join(IO_DIR, f"{base_name}_COLOR.png")
                
                colorize_mask(input_path, output_color_path)
            print("Все маски успешно раскрашены!")