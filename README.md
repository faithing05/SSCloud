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

## Замените путь на ваш
```python
cd C:\Users\YourUser\Documents\Cvat
```

## Команда остановит все сервисы CVAT и удалит контейнеры
```python
docker-compose down
```

## Эта команда скачает (если нужно) свежие образы CVAT и создаст новые, чистые контейнеры. Флаг -d (detached) запускает их в фоновом режиме.
```python
docker-compose up -d
```

## Проверка: Подождите пару минут, пока все сервисы запустятся. Вы можете проверить их статус командой:
```python
docker-compose down
```

