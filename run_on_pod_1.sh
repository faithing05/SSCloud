#!/bin/bash
# chmod +x run_on_pod_1.sh
set -e

# --- ПЕРЕМЕННЫЕ ---
GDRIVE_FILE_ID="1WVDuXBvYy9nE9xKmyC7VVQHqNwNMWMFL"
ARCHIVE_NAME="input_data.rar"
DATA_INPUT_DIR="Data_Input"
RCLONE_CONFIG="/workspace/rclone_config/rclone.conf"

echo "--- 1. Проверка и установка инструментов ---"
# Проверяем наличие unrar и zip, чтобы не делать apt-get update каждый раз
if ! command -v unrar &> /dev/null || ! command -v zip &> /dev/null; then
    echo "Утилиты не найдены. Установка unrar, zip, curl..."
    apt-get update -qq && apt-get install -y -qq unrar zip curl > /dev/null
else
    echo "[ПРОПУСК] unrar и zip уже установлены."
fi

# Проверяем наличие gdown
if ! pip show gdown &> /dev/null; then
    echo "Установка gdown..."
    pip install gdown -q
else
    echo "[ПРОПУСК] gdown уже установлен."
fi

echo "--- 2. Проверка и подготовка данных ---"
# Если папка Data_Input уже существует и в ней есть файлы — пропускаем скачивание и распаковку
if [ -d "$DATA_INPUT_DIR" ] && [ "$(ls -A "$DATA_INPUT_DIR" 2>/dev/null)" ]; then
    echo "[ПРОПУСК] Папка $DATA_INPUT_DIR уже подготовлена. Ничего делать не нужно."
else
    echo "Папка $DATA_INPUT_DIR пуста или отсутствует. Начинаем подготовку..."

    # Проверяем, скачан ли уже сам архив
    if [ ! -f "$ARCHIVE_NAME" ]; then
        echo "[СТАРТ] Скачивание архива с Google Drive..."
        gdown "$GDRIVE_FILE_ID" -O "$ARCHIVE_NAME"
    else
        echo "[ПРОПУСК] Архив $ARCHIVE_NAME уже скачан."
    fi

    echo "[СТАРТ] Распаковка архива..."
    # Распаковываем (флаг -o+ перезаписывает файлы, если они вдруг есть)
    unrar x -o+ "$ARCHIVE_NAME"

    echo "[СТАРТ] Перенос файлов в $DATA_INPUT_DIR..."
    mkdir -p "$DATA_INPUT_DIR"
    
    # Перемещаем все .jpg файлы из текущей папки в целевую
    # Используем проверку, чтобы mv не выдал ошибку, если файлов .jpg вдруг нет в корне
    if ls *.jpg 1> /dev/null 2>&1; then
        mv *.jpg "$DATA_INPUT_DIR/"
        echo "Файлы успешно перемещены."
    else
        echo "Предупреждение: .jpg файлы не найдены в корне после распаковки."
    fi
fi

echo "--- ГОТОВО! Можно приступать к работе ---"
