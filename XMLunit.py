import os
import zipfile
import io

def create_combined_labelme_zip():
    """
    Находит индивидуальные ZIP-архивы с LabelMe аннотациями (XML + JPG),
    извлекает их содержимое и упаковывает всё в один большой ZIP-архив,
    готовый для загрузки в CVAT.
    """
    base_path = r'F:\Desktop\SSCloud\CVAT_Workspace'
    # Имя для финального архива
    output_zip_filename = 'upload_to_cvat_labelme.zip'
    output_zip_path = os.path.join(base_path, output_zip_filename)
    
    print(f"Создание нового архива: {output_zip_path}")

    # Создаем новый ZIP-архив для записи
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as combined_zip:
        files_added_count = 0
        
        print("Поиск и обработка исходных архивов...")
        # Проходим по папкам от 1_normals до 21_normals
        for i in range(1, 22):
            folder_name = f'{i}_normals'
            subfolder_name = '5_upload_to_cvat'
            archive_dir = os.path.join(base_path, folder_name, subfolder_name)
            
            if not os.path.isdir(archive_dir):
                print(f"  ПРЕДУПРЕЖДЕНИЕ: Папка не найдена, пропущена: {archive_dir}")
                continue

            # Ищем ZIP-архив в этой папке
            source_zip_path = None
            for filename in os.listdir(archive_dir):
                if filename.lower().endswith('.zip'):
                    source_zip_path = os.path.join(archive_dir, filename)
                    break
            
            if not source_zip_path:
                print(f"  ПРЕДУПРЕЖДЕНИЕ: ZIP-архив не найден в {archive_dir}")
                continue
            
            print(f"  Обработка архива: {source_zip_path}")
            
            # Читаем содержимое исходного архива и добавляем файлы в новый
            try:
                with zipfile.ZipFile(source_zip_path, 'r') as source_zip:
                    for filename in source_zip.namelist():
                        # Проверяем, что это не папка
                        if not filename.endswith('/'):
                            file_content = source_zip.read(filename)
                            # Добавляем файл с его содержимым в новый архив
                            combined_zip.writestr(filename, file_content)
                            print(f"    -> Добавлен файл: {filename}")
                            files_added_count += 1
            except Exception as e:
                print(f"    -> ОШИБКА при чтении архива {source_zip_path}: {e}")
                
    print("\n========================================================")
    if files_added_count > 0:
        print("Готово! ZIP-архив успешно создан.")
        print(f"Всего добавлено файлов: {files_added_count} ({files_added_count // 2} пар XML+JPG)")
        print(f"Результат сохранен в файл: {output_zip_path}")
        print("\nЗагрузите этот в CVAT, выбрав формат 'LabelMe'.")
    else:
        print("Ошибка: Не было добавлено ни одного файла. Проверьте пути и наличие исходных архивов.")
    print("========================================================")


# Запускаем функцию
if __name__ == "__main__":
    create_combined_labelme_zip()