// Находим все необходимые элементы на странице
const startBtn = document.getElementById('start-btn');
const filenameInput = document.getElementById('filename');
const statusDiv = document.getElementById('status');
const classifierDiv = document.getElementById('classifier');
const imageEl = document.getElementById('highlighted-image');
const counterEl = document.getElementById('mask-counter');
const classButtonsDiv = document.getElementById('class-buttons');
const skipBtn = document.getElementById('skip-btn');
const finalActionsDiv = document.getElementById('final-actions');
const visualizeBtn = document.getElementById('visualize-btn');
const visualizationContainer = document.getElementById('visualization-container');
const finalMaskImage = document.getElementById('final-mask-image');
const exportBtn = document.getElementById('export-btn');

// --- Глобальные переменные ---
const CLASS_NAMES = ["Фон", "Земля", "Человек", "Растительность", "Транспорт", "Конструкции", "Здание", "Обстановка"];
let statusInterval = null;
let currentMaskName = null;

// --- Основные функции ---

/**
 * Периодически запрашивает у сервера текущий статус и отображает его.
 */
async function pollStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        statusDiv.innerText = data.status;
    } catch (error) {
        console.error("Ошибка при опросе статуса:", error);
    }
}

/**
 * Запрашивает у бэкенда следующую маску для классификации и отображает ее.
 */
async function showNextMask() {
    statusDiv.innerText = `Загрузка следующей маски...`;
    
    try {
        const response = await fetch('/get-next-mask');
        const data = await response.json();

        if (data.done || !data.mask_data) {
            statusDiv.innerText = "Все маски классифицированы! Теперь можно визуализировать результат.";
            classifierDiv.style.display = 'none';
            finalActionsDiv.style.display = 'block';
            currentMaskName = null;
            return;
        }

        currentMaskName = data.mask_data.mask_name;
        counterEl.innerText = `Осталось масок: ${data.remaining}`;
        statusDiv.innerText = `Классифицируйте маску: ${currentMaskName}`;
        imageEl.src = `data:image/jpeg;base64,${data.mask_data.highlighted_image_b64}`;

    } catch (error) {
        statusDiv.innerText = `Ошибка при загрузке следующей маски: ${error}`;
    }
}

/**
 * Отправляет на сервер команду классифицировать текущую маску.
 */
async function classify(className) {
    if (!currentMaskName) return;

    [...classButtonsDiv.children, skipBtn].forEach(btn => btn.disabled = true);
    
    try {
        await fetch('/classify-mask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mask_name: currentMaskName, class_name: className })
        });
        await showNextMask();
    } catch (error) {
        statusDiv.innerText = `Ошибка при классификации: ${error}`;
    } finally {
        [...classButtonsDiv.children, skipBtn].forEach(btn => btn.disabled = false);
    }
}

// --- Обработчики событий ---

/**
 * Обработчик нажатия на кнопку "Начать обработку".
 */
startBtn.addEventListener('click', async () => {
    const filename = filenameInput.value;
    statusDiv.innerText = 'Запрос на запуск обработки отправлен...';
    startBtn.disabled = true;
    classifierDiv.style.display = 'none';
    finalActionsDiv.style.display = 'none';
    visualizationContainer.style.display = 'none';
    
    if (statusInterval) clearInterval(statusInterval);
    statusInterval = setInterval(pollStatus, 1500);

    try {
        const response = await fetch('/start-processing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ panorama_filename: filename })
        });

        if (!response.ok) {
            const errorData = await response.json();
            statusDiv.innerText = `Ошибка: ${errorData.detail || 'Неизвестная ошибка'}`;
            throw new Error('Server error');
        }
        
        const data = await response.json();

        if (data.total_masks > 0) {
            classifierDiv.style.display = 'block';
            await showNextMask();
        } else {
            statusDiv.innerText = 'Нет масок для классификации. Возможно, они уже были сгенерированы.';
            finalActionsDiv.style.display = 'block';
        }
    } catch (error) {
        console.error("Ошибка при запуске обработки:", error);
    } finally {
        clearInterval(statusInterval);
        startBtn.disabled = false;
    }
});

/**
 * Обработчик нажатия на кнопку "Визуализировать".
 */
visualizeBtn.addEventListener('click', async () => {
    statusDiv.innerText = 'Генерация финальной маски...';
    visualizeBtn.disabled = true;

    try {
        const response = await fetch('/visualize');
        if (!response.ok) {
            const errorText = await response.text();
            statusDiv.innerText = `Ошибка визуализации: ${errorText}`;
            return;
        }

        const data = await response.json();
        
        finalMaskImage.src = `data:image/png;base64,${data.image_b64}`;
        visualizationContainer.style.display = 'block';
        statusDiv.innerText = 'Визуализация завершена.';
    } catch (error) {
        statusDiv.innerText = `Ошибка: ${error}`;
    } finally {
        visualizeBtn.disabled = false;
    }
});

/**
 * Обработчик нажатия на кнопку "Скачать ZIP для CVAT".
 */
exportBtn.addEventListener('click', async () => {
    statusDiv.innerText = 'Создание ZIP-архива...';
    exportBtn.disabled = true;

    try {
        const response = await fetch('/export');
        
        if (!response.ok) {
            const errorText = await response.text();
            statusDiv.innerText = `Ошибка экспорта: ${errorText}`;
            throw new Error('Export failed');
        }

        // Этот код заставляет браузер скачать полученный файл
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        // Пытаемся получить имя файла из заголовков ответа
        const disposition = response.headers.get('content-disposition');
        let filename = 'archive.zip'; // Имя по умолчанию
        if (disposition && disposition.indexOf('attachment') !== -1) {
            const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
            const matches = filenameRegex.exec(disposition);
            if (matches != null && matches[1]) {
                filename = matches[1].replace(/['"]/g, '');
            }
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
        
        statusDiv.innerText = 'ZIP-архив успешно скачан.';

    } catch (error) {
        console.error("Ошибка при экспорте:", error);
    } finally {
        exportBtn.disabled = false;
    }
});


// Навешиваем обработчик на кнопку "Пропустить"
skipBtn.addEventListener('click', () => classify('Пропустить'));


// --- Инициализация при загрузке страницы ---

// Создаем кнопки для всех классов
CLASS_NAMES.forEach(name => {
    const button = document.createElement('button');
    button.innerText = name;
    button.addEventListener('click', () => classify(name));
    classButtonsDiv.appendChild(button);
});