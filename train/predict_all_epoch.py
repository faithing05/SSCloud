import os
import subprocess
import argparse
from tqdm import tqdm
import re

# ИСПОЛЬЗОВАНИЕ (внутри Docker-контейнера):
# python predict_all.py -i test_input/1_normals.jpg

def run_inference_for_all_checkpoints(input_image, checkpoints_dir, output_dir, scale, num_classes):
    """
    Запускает инференс для каждого чекпоинта в указанной папке.
    """
    print(f"--- Начало пакетного инференса для изображения: {input_image} ---")
    
    # Убедимся, что папки существуют
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(checkpoints_dir):
        print(f"ОШИБКА: Папка с чекпоинтами не найдена: {checkpoints_dir}")
        return
    if not os.path.isfile(input_image):
        print(f"ОШИБКА: Входное изображение не найдено: {input_image}")
        return

    # Находим все .pth файлы
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
    
    # --- Умная сортировка ---
    # Сортируем файлы не по алфавиту ('epoch1', 'epoch10', 'epoch2'), а по номеру эпохи.
    def get_epoch_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else -1

    checkpoint_files.sort(key=get_epoch_number)

    if not checkpoint_files:
        print(f"ОШИБКА: Чекпоинты (.pth файлы) не найдены в папке: {checkpoints_dir}")
        return
        
    print(f"Найдено {len(checkpoint_files)} чекпоинтов для обработки.")
    
    # Получаем базовое имя входного файла для именования результатов
    input_basename = os.path.splitext(os.path.basename(input_image))[0]

    # Проходим по каждому чекпоинту
    for checkpoint_file in tqdm(checkpoint_files, desc="Инференс по чекпоинтам"):
        epoch_num = get_epoch_number(checkpoint_file)
        model_path = os.path.join(checkpoints_dir, checkpoint_file)
        
        # Создаем уникальное имя для выходного файла
        output_filename = f"{input_basename}_epoch_{epoch_num}.png"
        output_path = os.path.join(output_dir, output_filename)

        # Собираем команду для запуска predict.py
        command = [
            'python', 'predict.py',
            '-m', model_path,
            '-i', input_image,
            '-o', output_path,
            '--scale', str(scale),
            '--classes', str(num_classes)
        ]
        
        # Запускаем команду как отдельный процесс
        try:
            # Используем DEVNULL, чтобы не засорять консоль выводом predict.py
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"\nОШИБКА при обработке чекпоинта {checkpoint_file}:")
            print("Возможно, в predict.py все еще есть аргумент, который нужно добавить.")
            print(f"Команда, которая не сработала: {' '.join(command)}")
            break # Прерываем цикл в случае ошибки

    print(f"\n--- Пакетный инференс завершен. Результаты сохранены в: {output_dir} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Запускает инференс для всех чекпоинтов на одном изображении.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Путь к входному изображению (например, "test_input/1_normals.jpg").')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints', help='Папка с сохраненными моделями (.pth).')
    parser.add_argument('--output-dir', type=str, default='data/test/output_all_epochs', help='Папка для сохранения всех предсказанных масок.')
    parser.add_argument('--scale', type=float, default=0.1, help='Коэффициент масштабирования, использованный при обучении.')
    parser.add_argument('--classes', type=int, default=68, help='Количество классов, использованное при обучении.')
    
    args = parser.parse_args()

    run_inference_for_all_checkpoints(
        input_image=args.input,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        scale=args.scale,
        num_classes=args.classes
    )