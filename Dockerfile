FROM nvidia/cuda:11.7.1-base-ubuntu22.04

# Установка переменных окружения для автоматической установки
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ИЗМЕНЕНИЕ: Добавлены nodejs и npm, необходимые для сборки расширений JupyterLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    wget \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Создаем символическую ссылку, чтобы 'python' вызывал 'python3.10'
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Устанавливаем PyTorch через PIP.
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Устанавливаем остальные библиотеки через pip
RUN pip install \
    "numpy<2.0" \
    matplotlib \
    laspy \
    opencv-python-headless \
    jupyterlab \
    git+https://github.com/facebookresearch/segment-anything.git \
    "transformers==4.28.1" \
    sentencepiece \
    Pillow \
    ipywidgets

# Создаем рабочую директорию и папки для данных
WORKDIR /app

# Создаем папку для моделей
RUN mkdir -p /app/models

# Копируем рабочую директорию в /app
COPY . .

# Открываем порт 8888 для JupyterLab
EXPOSE 8888

# Команда по умолчанию для запуска контейнера
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]