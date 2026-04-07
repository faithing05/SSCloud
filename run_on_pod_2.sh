#!/bin/bash
# chmod +x run_on_pod_2.sh
set -e

# --- ПЕРЕМЕННЫЕ ---
GDRIVE_FILE_ID="1WVDuXBvYy9nE9xKmyC7VVQHqNwNMWMFL"
ARCHIVE_NAME="input_data.rar"
DATA_INPUT_DIR="Data_Input"
DATA_OUTPUT_DIR="Data_Output"
RCLONE_CONFIG="/root/.config/rclone/rclone.conf"
# Имя вашего конфига в rclone (проверьте через rclone listremotes)
REMOTE_NAME="grive" 

echo "--- 3. Проверка инструментов ---"
if ! command -v zip > /dev/null 2>&1; then
    echo "Утилита zip не найдена. Установка..."
    apt-get update -qq && apt-get install -y -qq zip > /dev/null
fi

if ! command -v rclone > /dev/null 2>&1; then
    echo "ОШИБКА: rclone не найден. Установите rclone и проверьте конфиг: $RCLONE_CONFIG"
    exit 1
fi

if [ ! -f "$RCLONE_CONFIG" ]; then
    echo "ОШИБКА: Конфиг rclone не найден: $RCLONE_CONFIG"
    exit 1
fi

REMOTE_FOUND=0
while IFS= read -r remote; do
    if [ "$remote" = "${REMOTE_NAME}:" ]; then
        REMOTE_FOUND=1
        break
    fi
done < <(rclone --config "$RCLONE_CONFIG" listremotes)

if [ "$REMOTE_FOUND" -ne 1 ]; then
    echo "ОШИБКА: Remote '${REMOTE_NAME}:' не найден в rclone конфиге $RCLONE_CONFIG"
    exit 1
fi

echo "--- 4. Архивация результатов ---"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NEW_ZIP="Data_Output_${TIMESTAMP}.zip"
ZIP_TO_UPLOAD=""

# Проверяем, есть ли результаты для архивации
if [ -d "$DATA_OUTPUT_DIR" ] && [ "$(ls -A "$DATA_OUTPUT_DIR")" ]; then
    # Проверяем, не лежит ли уже какой-то готовый ZIP (от прошлого неудачного запуска)
    EXISTING_ZIP=""
    for candidate in Data_Output_*.zip; do
        if [ -f "$candidate" ]; then
            EXISTING_ZIP="$candidate"
            break
        fi
    done
    
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
