# SSCloud - Semantic Segmentation Cloud 

## Работа с сегментацией и полуавтоматической разметкой классов с помощью CVAT_Masks

## Сборка образа

```python
docker build --no-cache -t s3d .
```
s3d - название образа

## Запуск образа
```python
docker run --gpus all -it -p 8888:8888 -v "${PWD}:/app" s3d
```
## Запуск образа с пробрасыванием путей до моделей
```python
docker run --gpus all -it --rm -p 8888:8888 -v "${PWD}:/app" s3d
```


## Docker образ CVAT
Если нет репозитория, скачайте. Откройте терминал и введите.

```python
git clone https://github.com/cvat-ai/cvat
```

Далее переходим по пути CVAT
```python
cd cvat
```

## Эта команда скачает (если нужно) свежие образы CVAT и создаст новые, чистые контейнеры. Флаг -d (detached) запускает их в фоновом режиме.
```python
docker-compose up -d
```

## Подождите пару минут, пока все сервисы запустятся. Вы можете проверить их статус командой:
```python
docker-compose ps
```

## Заходим внутрь образа и создаем пользователя
```python
docker exec -it cvat_server bash
python manager.py createsuperuser
exit
```

## Посмотреть логи и адрес сайта, где находится локальный CVAT. Обычно это http://localhost:8080/. Далее CTRL+C
```python
docker logs cvat_server -f
```

## Команда остановит все сервисы CVAT и удалит контейнеры
```python
docker-compose down
```



# TRAIN архитектуры Unet на основе собранного нам датасета

Предварительно необходимо запустить скрипт convert_labelme.py. 
Он подготавливает данные для обучения. 
Переносит исходные панорамы в train/data/imgs. 
Подготавливает размеченные панорамы для обучения, также переносит их в train/data/masks

## Запуск контейнера с Unet

    для обучения:
    ```python
    docker run --rm -it --gpus all --shm-size=8g --ulimit memlock=-1 -v "${PWD}/data:/app/data" milesial/unet
    ```

    для инфереса:
    ```python
    docker run --rm -it --gpus all -v "${PWD}/data/test/input:/workspace/unet/test_input"  -v "${PWD}/data/test/output:/workspace/unet/test_output" -v "${PWD}/checkpoints:/workspace/unet/checkpoints" ` milesial/unet
    ```

Внутри контейнера:

    Запуск обучения:
    ```python
    WANDB_MODE=offline python train.py --epochs 50 --batch-size 1 --scale 0.1 --amp --classes 68
    ```
    
    Запуск инференса:
    ```python
    python predict.py -m checkpoints/checkpoint_epoch48.pth -i test_input/1_normals.jpg -o test_output/predicted_mask.png --scale 0.1 --classes 68
    ```

В powershell:

    Для конвертации датасета в машиночитаемые маски в папке data/test/imgs и data/test/masks :
    ```python
    python _convert_labelme.py
    ```

    Для визуализации инференса
    ```python
    python _visualize_prediction.py
    ```