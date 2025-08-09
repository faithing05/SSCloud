# SSCloud - Semantic Segmentation Cloud 

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



