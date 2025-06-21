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
