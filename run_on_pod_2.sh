#chmod +x run_on_pod_2.sh
#!/bin/bash
set -e

# --- ПЕРЕМЕННЫЕ ---
GDRIVE_FILE_ID="1WVDuXBvYy9nE9xKmyC7VVQHqNwNMWMFL"
ARCHIVE_NAME="input_data.rar"
DATA_INPUT_DIR="Data_Input"
DATA_OUTPUT_DIR="Data_Output"
RCLONE_CONFIG="/workspace/rclone_config/rclone.conf"
# Имя вашего конфига в rclone (проверьте через rclone listremotes)
REMOTE_NAME="grive" 

echo "--- 1. Проверка инструментов ---"
# Устанавливаем только если их нет (apt быстро проверит состояние)
if ! command -v unrar &> /dev/null || ! command -v zip &> /dev/null; then
    echo "Установка unrar и zip..."
    apt-get update -qq && apt-get install -y -qq unrar zip curl > /dev/null
fi

if ! pip show gdown &> /dev/null; then
    echo "Установка gdown..."
    pip install gdown -q
fi

echo "--- 2. Скачивание входных данных ---"
# Если папка с картинками уже есть и не пуста — пропускаем всё
if [ -d "$DATA_INPUT_DIR" ] && [ "$(ls -A $DATA_INPUT_DIR)" ]; then
    echo "[ПРОПУСК] Папка $DATA_INPUT_DIR уже существует и содержит файлы."
else
    # Если папки нет, проверяем сам архив
    if [ ! -f "$ARCHIVE_NAME" ]; then
        echo "[СТАРТ] Скачивание архива с Google Drive..."
        gdown "$GDRIVE_FILE_ID" -O "$ARCHIVE_NAME"
    else
        echo "[ПРОПУСК] Архив $ARCHIVE_NAME уже скачан."
    fi

    echo "[СТАРТ] Распаковка архива..."
    mkdir -p "$DATA_INPUT_DIR"
    # Распаковываем файлы
    unrar x -o+ "$ARCHIVE_NAME" 
    
    # Если файлы Vyb_*.jpg распаковались в текущую папку, перемещаем их в Data_Input
    if ls Vyb_*.jpg 1> /dev/null 2>&1; then
        mv Vyb_*.jpg "$DATA_INPUT_DIR/"
        echo "Файлы перемещены в $DATA_INPUT_DIR"
    fi
fi

echo "--- 3. Обработка данных (место для вашей нейросети) ---"
# Здесь должен быть запуск вашего кода, который создаст файлы в Data_Output
# Например: python3 process.py

echo "--- 4. Архивация результатов ---"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NEW_ZIP="Data_Output_${TIMESTAMP}.zip"

# Проверяем, есть ли результаты для архивации
if [ -d "$DATA_OUTPUT_DIR" ] && [ "$(ls -A $DATA_OUTPUT_DIR)" ]; then
    # Проверяем, не лежит ли уже какой-то готовый ZIP (от прошлого неудачного запуска)
    EXISTING_ZIP=$(ls Data_Output_*.zip 2>/dev/null | head -n 1)
    
    if [ -f "$EXISTING_ZIP" ]; then
        echo "[ПРОПУСК] Архив $EXISTING_ZIP уже готов к отправке."
        ZIP_TO_UPLOAD="$EXISTING_ZIP"
    else
        echo "[СТАРТ] Создание архива $NEW_ZIP..."
        zip -r "$NEW_ZIP" "$DATA_OUTPUT_DIR"
        ZIP_TO_UPLOAD="$NEW_ZIP"
    fi
else
    echo "[ИНФО] Папка $DATA_OUTPUT_DIR пуста. Нечего архивировать."
fi

echo "--- 5. Выгрузка в Google Drive ---"
if [ -z "$ZIP_TO_UPLOAD" ]; then
    echo "Нет файлов для загрузки."
else
    echo "[СТАРТ] Загрузка $ZIP_TO_UPLOAD в Google Drive..."
    # Используем grive: или gdrive: в зависимости от вашего конфига
    if rclone --config "$RCLONE_CONFIG" copy "$ZIP_TO_UPLOAD" "${REMOTE_NAME}:/RunPod_Results/" --progress; then
        echo "--- Успешно загружено! ---"
        # Перемещаем в архивную папку на сервере, чтобы не загружать повторно
        mkdir -p uploaded_backups
        mv "$ZIP_TO_UPLOAD" uploaded_backups/
    else
        echo "ОШИБКА: Загрузка не удалась."
        exit 1
    fi
fi

echo "--- Скрипт завершил работу ---"