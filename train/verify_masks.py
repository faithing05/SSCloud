import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- НАСТРОЙКИ ---
# Путь к папке с масками, которые мы хотим проверить
MASKS_DIR = 'data/masks'

# Ожидаемые ID классов из вашего словаря (включая фон 0)
EXPECTED_IDS = {0, 1, 2, 5, 6, 64, 65, 66, 67}

def analyze_masks():
    """
    Анализирует все PNG маски в папке, выводит уникальные значения пикселей (ID классов)
    и проверяет, соответствуют ли они ожидаемым.
    """
    print(f"--- Анализ масок в папке: {MASKS_DIR} ---")
    mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]

    if not mask_files:
        print("ОШИБКА: Маски для анализа не найдены.")
        return

    all_found_ids = set()
    has_unexpected_ids = False

    for mask_file in tqdm(mask_files, desc="Проверка масок"):
        try:
            mask_path = os.path.join(MASKS_DIR, mask_file)
            mask_image = Image.open(mask_path)
            mask_array = np.array(mask_image)
            
            # Находим все уникальные значения пикселей в маске
            unique_ids = np.unique(mask_array)
            all_found_ids.update(unique_ids)
            
            # Проверяем, есть ли в этой маске неожиданные ID
            for uid in unique_ids:
                if uid not in EXPECTED_IDS:
                    print(f"\nВНИМАНИЕ: В файле '{mask_file}' найден неожиданный ID класса: {uid}")
                    has_unexpected_ids = True

        except Exception as e:
            print(f"\nОшибка при чтении файла {mask_file}: {e}")
    
    print("\n--- Итоги анализа ---")
    print(f"Всего в датасете найдены следующие уникальные ID классов: {sorted(list(all_found_ids))}")

    if has_unexpected_ids:
        print("\nРЕЗУЛЬТАТ: ОБНАРУЖЕНЫ НЕОЖИДАННЫЕ ID. Проверьте сообщения выше.")
    elif not all_found_ids.issubset(EXPECTED_IDS):
         print("\nРЕЗУЛЬТАТ: В масках есть ID, которых нет в списке EXPECTED_IDS.")
    else:
        print("\nРЕЗУЛЬТАТ: Все найденные ID классов соответствуют ожидаемым. Маски корректны!")

if __name__ == '__main__':
    analyze_masks()