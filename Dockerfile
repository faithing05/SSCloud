# Шаг 1: Используем проверенный базовый образ.
FROM nvidia/cuda:11.7.1-base-ubuntu22.0

# Установка переменных окружения для автоматической установки
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Шаг 2: Устанавливаем системные зависимости.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Создаем символическую ссылку, чтобы 'python' указывал на 'python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Шаг 3: Устанавливаем рабочую директорию.
WORKDIR /workspace/SSCloud

# Шаг 4: Копируем файл с зависимостями и устанавливаем их.
COPY requirements.txt .

# Устанавливаем PyTorch и другие библиотеки. Добавляем --extra-index-url для PyTorch.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

# Шаг 5: Копируем все файлы нашего приложения в рабочую директорию.
COPY . .

# Шаг 6: Открываем порт 8000 для FastAPI-сервера.
EXPOSE 8000

# Шаг 7: Команда для запуска Uvicorn сервера.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]