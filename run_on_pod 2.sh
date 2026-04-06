#!/bin/bash
# Останавливать скрипт при любой ошибке
set -e

# --- ПЕРЕМЕННЫЕ ---
GDRIVE_FILE_ID="1WVDuXBvYy9nE9xKmyC7VVQHqNwNMWMFL"
ARCHIVE_NAME="input_data.rar"
RCLONE_CONFIG="/workspace/rclone_config/rclone.conf"

echo "--- 5. Архивация результатов ---"
zip -r Data_Output_$(date +%Y%m%d_%H%M%S).zip Data_Output

echo "--- 6. Выгрузка в Google Диск ---"
# Используем сохраненный конфиг rclone
rclone --config $RCLONE_CONFIG copy Data_Output_*.zip gdrive:/RunPod_Results/ --progress

echo "--- ГОТОВО! Проверьте Google Диск в папке RunPod_Results ---"