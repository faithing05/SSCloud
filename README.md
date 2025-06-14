# SSCloud - Semantic Segmentation Cloud 

## Сборка образа

```python
docker build --no-cache -t s3d .
```

## Запуск образа
```python
docker run --gpus all -it -p 8888:8888 -v "${PWD}:/app/CODE" s3d
```
