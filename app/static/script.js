// Находим все необходимые элементы на странице
const startBtn = document.getElementById('start-btn');
const filenameInput = document.getElementById('filename');
const statusDiv = document.getElementById('status');
const classifierDiv = document.getElementById('classifier');
const originalImageEl = document.getElementById('original-image');
const maskImageEl = document.getElementById('mask-image');
const highlightedImageEl = document.getElementById('highlighted-image');
const counterEl = document.getElementById('mask-counter');
const classButtonsDiv = document.getElementById('class-buttons');
const skipBtn = document.getElementById('skip-btn');
const finalActionsDiv = document.getElementById('final-actions');
const visualizeBtn = document.getElementById('visualize-btn');
const visualizationContainer = document.getElementById('visualization-container');
const finalMaskImage = document.getElementById('final-mask-image');
const exportBtn = document.getElementById('export-btn');
const e57Select = document.getElementById('e57-select');
const e57ProcessSelectedBtn = document.getElementById('e57-process-selected-btn');
const e57ProcessAllBtn = document.getElementById('e57-process-all-btn');
const e57StatusDiv = document.getElementById('e57-status');
const panoramaSelect = document.getElementById('panorama-select');
const segmentSelectedBtn = document.getElementById('segment-selected-btn');
const segmentedPanoramaSelect = document.getElementById('segmented-panorama-select');
const startClassificationBatchBtn = document.getElementById('start-classification-batch-btn');

// --- Глобальные переменные ---
const CLASS_NAMES = ["Фон", "Земля", "Человек", "Растительность", "Транспорт", "Конструкции", "Здание", "Обстановка"];
let statusInterval = null;
let currentMaskName = null;

// --- Основные функции ---

/**
 * Загружает список .e57 файлов с сервера и заполняет выпадающий список.
 */
async function loadE57Files() {
    try {
        const response = await fetch('/get-e57-files');
        const data = await response.json();
        e57Select.innerHTML = ''; // Очищаем список
        if (data.files && data.files.length > 0) {
            data.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.innerText = file;
                e57Select.appendChild(option);
            });
        } else {
            e57StatusDiv.innerText = "В папке Vistino20241014_E57 не найдено .e57 файлов.";
        }
    } catch (error) {
        e57StatusDiv.innerText = `Ошибка загрузки списка файлов: ${error}`;
    }
}

/**
 * Загружает список JPG-панорам для сегментации.
 */
async function loadPanoramaFiles() {
    try {
        const response = await fetch('/get-panorama-files');
        const data = await response.json();
        panoramaSelect.innerHTML = '';

        if (data.files && data.files.length > 0) {
            data.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.innerText = file;
                panoramaSelect.appendChild(option);
            });
        }
    } catch (error) {
        statusDiv.innerText = `Ошибка загрузки панорам: ${error}`;
    }
}

/**
 * Загружает список доступных панорам для пакетной классификации.
 */
async function loadSegmentedPanoramas() {
    try {
        const response = await fetch('/get-segmented-panoramas');
        const data = await response.json();
        segmentedPanoramaSelect.innerHTML = '';

        if (data.files && data.files.length > 0) {
            data.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.innerText = file;
                segmentedPanoramaSelect.appendChild(option);
            });
        }
    } catch (error) {
        statusDiv.innerText = `Ошибка загрузки панорам для классификации: ${error}`;
    }
}

/**
 * Общая функция для запуска обработки E57.
 * @param {string[]} filesToProcess - Массив имен файлов для обработки.
 */
async function processE57(filesToProcess) {
    if (!filesToProcess || filesToProcess.length === 0) {
        e57StatusDiv.innerText = "Файлы для обработки не выбраны.";
        return;
    }

    e57StatusDiv.innerHTML = `Начинаем обработку ${filesToProcess.length} файла(ов)... Это может занять много времени.`;
    e57ProcessSelectedBtn.disabled = true;
    e57ProcessAllBtn.disabled = true;

    try {
        const response = await fetch('/process-e57', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Отправляем массив имен файлов
            body: JSON.stringify({ filenames: filesToProcess })
        });

        const result = await response.json();
        
        if (response.ok) {
            e57StatusDiv.innerHTML = `<strong>Обработка завершена!</strong><br><br>` + (result.logs || []).join('<br>');
            // Предлагаем пользователю имя последнего обработанного JPG
            const lastLog = result.logs[result.logs.length - 3]; // "JPG нормалей сохранён: X.jpg"
            if (lastLog && lastLog.includes('.jpg')) {
                const newJpgName = lastLog.split(': ')[1];
                filenameInput.value = newJpgName;
            }

            await loadPanoramaFiles();
        } else {
            const errorMessage = result.detail || "Неизвестная ошибка сервера.";
            e57StatusDiv.innerHTML = `<strong>Ошибка!</strong><br>${errorMessage}`;
        }
    } catch (error) {
        e57StatusDiv.innerText = `Критическая ошибка: ${error}`;
    } finally {
        e57ProcessSelectedBtn.disabled = false;
        e57ProcessAllBtn.disabled = false;
    }
}

// Обработчик для кнопки "Обработать выбранные"
e57ProcessSelectedBtn.addEventListener('click', () => {
    // Получаем все выбранные опции из select multiple
    const selectedFiles = Array.from(e57Select.selectedOptions).map(option => option.value);
    processE57(selectedFiles);
});

// Обработчик для кнопки "Обработать ВСЕ"
e57ProcessAllBtn.addEventListener('click', () => {
    // Получаем абсолютно все опции из списка
    const allFiles = Array.from(e57Select.options).map(option => option.value);
    processE57(allFiles);
});


// --- ЗАПУСК ПРИ ЗАГРУЗКЕ СТРАНИЦЫ ---
document.addEventListener('DOMContentLoaded', () => {
    loadE57Files(); // Загружаем список E57 при открытии страницы
    loadPanoramaFiles();
    loadSegmentedPanoramas();
});

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
        const panoramaLabel = data.current_panorama ? `Панорама: ${data.current_panorama}. ` : '';
        const progressLabel = data.panorama_progress
            ? `Панорама ${data.panorama_progress.current}/${data.panorama_progress.total}. `
            : '';
        counterEl.innerText = `${panoramaLabel}${progressLabel}Осталось масок: ${data.remaining}`;
        statusDiv.innerText = `Классифицируйте маску: ${currentMaskName}`;
        originalImageEl.src = `data:image/jpeg;base64,${data.mask_data.original_panorama_b64}`;
        maskImageEl.src = `data:image/png;base64,${data.mask_data.mask_image_b64}`;
        highlightedImageEl.src = `data:image/jpeg;base64,${data.mask_data.highlighted_image_b64}`;

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

        await loadSegmentedPanoramas();
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

/**
 * Обработчик пакетной сегментации выбранных панорам.
 */
segmentSelectedBtn.addEventListener('click', async () => {
    const selectedPanoramas = Array.from(panoramaSelect.selectedOptions).map(option => option.value);
    if (selectedPanoramas.length === 0) {
        statusDiv.innerText = 'Выберите хотя бы одну панораму для сегментации.';
        return;
    }

    segmentSelectedBtn.disabled = true;
    statusDiv.innerText = `Запущена сегментация ${selectedPanoramas.length} панорам...`;

    if (statusInterval) clearInterval(statusInterval);
    statusInterval = setInterval(pollStatus, 1500);

    try {
        const response = await fetch('/start-segmentation-batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ panorama_filenames: selectedPanoramas })
        });

        const data = await response.json();
        if (!response.ok) {
            statusDiv.innerText = `Ошибка пакетной сегментации: ${data.detail || 'Неизвестная ошибка'}`;
            return;
        }

        const logs = (data.logs || []).join('\n');
        statusDiv.innerText = `Пакетная сегментация завершена.\n${logs}`;
        await loadSegmentedPanoramas();
    } catch (error) {
        statusDiv.innerText = `Ошибка пакетной сегментации: ${error}`;
    } finally {
        clearInterval(statusInterval);
        segmentSelectedBtn.disabled = false;
    }
});

/**
 * Обработчик запуска пакетной классификации выбранных панорам.
 */
startClassificationBatchBtn.addEventListener('click', async () => {
    const selectedPanoramas = Array.from(segmentedPanoramaSelect.selectedOptions).map(option => option.value);
    if (selectedPanoramas.length === 0) {
        statusDiv.innerText = 'Выберите хотя бы одну панораму для классификации.';
        return;
    }

    startClassificationBatchBtn.disabled = true;
    statusDiv.innerText = 'Запуск очереди классификации...';
    classifierDiv.style.display = 'none';
    finalActionsDiv.style.display = 'none';
    visualizationContainer.style.display = 'none';

    try {
        const response = await fetch('/start-classification-batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ panorama_filenames: selectedPanoramas })
        });

        const data = await response.json();
        if (!response.ok) {
            statusDiv.innerText = `Ошибка запуска классификации: ${data.detail || 'Неизвестная ошибка'}`;
            return;
        }

        statusDiv.innerText = `Запущена классификация ${data.total_panoramas} панорам.`;
        classifierDiv.style.display = 'block';
        await showNextMask();
    } catch (error) {
        statusDiv.innerText = `Ошибка запуска классификации: ${error}`;
    } finally {
        startClassificationBatchBtn.disabled = false;
    }
});
