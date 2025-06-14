SSCloud
Semantic Segmentation Cloud

Сборка и запуск

docker build --no-cache -t s3d .

docker run --gpus all -it -p 8888:8888 -v "${PWD}:/app/CODE" s3d
