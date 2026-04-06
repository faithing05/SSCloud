set -e

GDRIVE_FILE_ID="1WVDuXBvYy9nE9xKmyC7VVQHqNwNMWMFL"
ARCHIVE_NAME="input_data.rar"
RCLONE_CONFIG="/workspace/rclone_config/rclone.conf"

echo "--- 1. installing tools ---"
apt-get update && apt-get install -y unrar zip curl
pip install gdown -q

echo "--- 2. downloading and extracting archive ---"
gdown $GDRIVE_FILE_ID -O $ARCHIVE_NAME
unrar x $ARCHIVE_NAME

echo "--- 3. preparing Data_Input folder ---"
mkdir -p Data_Input
mv *.jpg Data_Input/

echo "--- READY! ---"