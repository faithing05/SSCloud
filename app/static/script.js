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
const undoBtn = document.getElementById('undo-btn');
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
const visualizationPanoramaSelect = document.getElementById('visualization-panorama-select');
const refreshVisualizationListBtn = document.getElementById('refresh-visualization-list-btn');
const refreshReviewBtn = document.getElementById('refresh-review-btn');
const reviewMaskSelect = document.getElementById('review-mask-select');
const reviewClassSelect = document.getElementById('review-class-select');
const applyReclassifyBtn = document.getElementById('apply-reclassify-btn');
const reviewHighlightedImageEl = document.getElementById('review-highlighted-image');

// --- Глобальные переменные ---
const CLASS_NAMES = ["Фон", "Земля", "Человек", "Растительность", "Транспорт", "Конструкции", "Здание", "Обстановка"];
let statusInterval = null;
let currentMaskName = null;
let reviewItems = [];


async function parseApiResponse(response) {
    const text = await response.text();
    if (!text) {
        return {};
    }

    try {
        return JSON.parse(text);
    } catch (error) {
        const preview = text.replace(/\s+/g, ' ').slice(0, 140);
        throw new Error(`Сервер вернул не JSON (HTTP ${response.status}): ${preview}`);
    }
}


function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


async function waitForSegmentationBatchCompletion() {
    while (true) {
        const response = await fetch('/segmentation-batch-status');
        const data = await parseApiResponse(response);

        if (!response.ok) {
            throw new Error(data.detail || 'Не удалось получить статус пакетной сегментации.');
        }

        const progress = data.total ? ` (${data.processed}/${data.total})` : '';
        statusDiv.innerText = `${data.status || 'Выполняется пакетная сегментация...'}${progress}`;

        if (!data.running) {
            return data;
        }

        await sleep(1500);
    }
}

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
            e57StatusDiv.innerText = "В папке Data_Input не найдено .e57 файлов.";
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
 * Загружает список панорам, готовых для визуализации финальной маски.
 */
async function loadVisualizationPanoramas() {
    try {
        const response = await fetch('/get-visualization-panoramas');
        const data = await response.json();
        const currentValue = visualizationPanoramaSelect.value;
        visualizationPanoramaSelect.innerHTML = '';

        if (data.files && data.files.length > 0) {
            data.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.innerText = file;
                visualizationPanoramaSelect.appendChild(option);
            });

            const hasCurrent = Array.from(visualizationPanoramaSelect.options).some(option => option.value === currentValue);
            visualizationPanoramaSelect.value = hasCurrent ? currentValue : data.files[0];
            if (classifierDiv.style.display !== 'block') {
                finalActionsDiv.style.display = 'block';
            }
        } else if (classifierDiv.style.display !== 'block') {
            finalActionsDiv.style.display = 'none';
        }
    } catch (error) {
        statusDiv.innerText = `Ошибка загрузки панорам для визуализации: ${error}`;
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

    const totalFiles = filesToProcess.length;
    const progressLogs = [];

    const renderE57Status = (isFinished = false) => {
        const title = isFinished
            ? `<strong>Обработка завершена!</strong><br><br>`
            : `Начинаем обработку ${totalFiles} файла(ов)... Это может занять много времени.<br><br>`;
        e57StatusDiv.innerHTML = title + progressLogs.join('<br>');
    };

    renderE57Status();
    e57ProcessSelectedBtn.disabled = true;
    e57ProcessAllBtn.disabled = true;

    try {
        for (let index = 0; index < totalFiles; index += 1) {
            const filename = filesToProcess[index];
            const response = await fetch('/process-e57', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filenames: [filename] })
            });

            const result = await response.json();
            if (!response.ok) {
                const errorMessage = result.detail || "Неизвестная ошибка сервера.";
                progressLogs.push(`Ошибка при обработке ${filename}: ${errorMessage}`);
                renderE57Status();
                continue;
            }

            const currentLogs = result.logs || [];
            progressLogs.push(`Файл ${index + 1}/${totalFiles} обработан: ${filename}`);
            if (currentLogs.length > 0) {
                progressLogs.push(...currentLogs);
            }
            renderE57Status();

            // Предлагаем пользователю имя последнего обработанного JPG
            const jpgLog = currentLogs.find(log => log.includes('JPG нормалей сохранён:'));
            if (jpgLog && jpgLog.includes('.jpg')) {
                const newJpgName = jpgLog.split(': ')[1];
                filenameInput.value = newJpgName;
            }
        }

        renderE57Status(true);
        await loadPanoramaFiles();
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
    loadVisualizationPanoramas();
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

    [...classButtonsDiv.children, skipBtn, undoBtn].forEach(btn => btn.disabled = true);
    
    try {
        const response = await fetch('/classify-mask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mask_name: currentMaskName, class_name: className })
        });

        if (!response.ok) {
            const errorData = await parseApiResponse(response);
            throw new Error(errorData.detail || 'Ошибка классификации');
        }

        await showNextMask();
        await loadReviewMasks();
    } catch (error) {
        statusDiv.innerText = `Ошибка при классификации: ${error}`;
    } finally {
        [...classButtonsDiv.children, skipBtn, undoBtn].forEach(btn => btn.disabled = false);
    }
}

async function undoLastClassification() {
    undoBtn.disabled = true;
    [...classButtonsDiv.children, skipBtn].forEach(btn => btn.disabled = true);

    try {
        const response = await fetch('/undo-last-classification', { method: 'POST' });
        const data = await parseApiResponse(response);
        if (!response.ok) {
            throw new Error(data.detail || 'Не удалось отменить действие');
        }

        statusDiv.innerText = `Отменено действие для маски: ${data.mask_name}`;
        await showNextMask();
        await loadReviewMasks();
    } catch (error) {
        statusDiv.innerText = `Ошибка отмены: ${error}`;
    } finally {
        undoBtn.disabled = false;
        [...classButtonsDiv.children, skipBtn].forEach(btn => btn.disabled = false);
    }
}

function renderReviewMaskOptions() {
    reviewMaskSelect.innerHTML = '';

    if (!reviewItems.length) {
        const option = document.createElement('option');
        option.value = '';
        option.innerText = 'Нет обработанных масок';
        reviewMaskSelect.appendChild(option);
        return;
    }

    reviewItems.forEach(item => {
        const option = document.createElement('option');
        option.value = item.mask_name;
        const classLabel = item.status === 'skipped' ? 'Пропущено' : item.class_name;
        option.innerText = `${item.mask_name} [${classLabel}]`;
        reviewMaskSelect.appendChild(option);
    });
}

async function loadReviewMasks() {
    try {
        const response = await fetch('/review-masks');
        const data = await parseApiResponse(response);
        if (!response.ok) {
            throw new Error(data.detail || 'Не удалось загрузить список масок для проверки');
        }

        reviewItems = data.items || [];
        renderReviewMaskOptions();
    } catch (error) {
        statusDiv.innerText = `Ошибка загрузки списка для проверки: ${error}`;
    }
}

async function loadReviewPreview(maskName) {
    if (!maskName) {
        reviewHighlightedImageEl.src = '';
        return;
    }

    try {
        const response = await fetch(`/mask-preview?mask_name=${encodeURIComponent(maskName)}`);
        const data = await parseApiResponse(response);
        if (!response.ok) {
            throw new Error(data.detail || 'Не удалось загрузить превью маски');
        }

        reviewHighlightedImageEl.src = `data:image/jpeg;base64,${data.mask_data.highlighted_image_b64}`;
    } catch (error) {
        statusDiv.innerText = `Ошибка загрузки превью маски: ${error}`;
    }
}

async function reclassifySelectedMask() {
    const selectedMask = reviewMaskSelect.value;
    const selectedClass = reviewClassSelect.value;

    if (!selectedMask) {
        statusDiv.innerText = 'Выберите маску в разделе проверки.';
        return;
    }

    applyReclassifyBtn.disabled = true;
    try {
        const response = await fetch('/reclassify-mask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mask_name: selectedMask, class_name: selectedClass })
        });

        const data = await parseApiResponse(response);
        if (!response.ok) {
            throw new Error(data.detail || 'Не удалось изменить класс маски');
        }

        statusDiv.innerText = data.message || `Класс маски ${selectedMask} обновлен.`;
        await loadReviewMasks();
        await loadReviewPreview(selectedMask);
    } catch (error) {
        statusDiv.innerText = `Ошибка переклассификации: ${error}`;
    } finally {
        applyReclassifyBtn.disabled = false;
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
            await loadReviewMasks();
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
    const selectedPanorama = visualizationPanoramaSelect.value;
    if (!selectedPanorama) {
        statusDiv.innerText = 'Выберите панораму для визуализации в Шаге 3.';
        return;
    }

    statusDiv.innerText = 'Генерация финальной маски...';
    visualizeBtn.disabled = true;

    try {
        const response = await fetch(`/visualize?panorama_filename=${encodeURIComponent(selectedPanorama)}`);
        if (!response.ok) {
            const errorText = await response.text();
            statusDiv.innerText = `Ошибка визуализации: ${errorText}`;
            return;
        }

        const data = await response.json();
        
        finalMaskImage.src = `data:image/png;base64,${data.image_b64}`;
        visualizationContainer.style.display = 'block';
        statusDiv.innerText = `Визуализация завершена: ${selectedPanorama}`;
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
    const selectedPanorama = visualizationPanoramaSelect.value;
    if (!selectedPanorama) {
        statusDiv.innerText = 'Выберите панораму для экспорта в Шаге 3.';
        return;
    }

    statusDiv.innerText = 'Создание ZIP-архива...';
    exportBtn.disabled = true;

    try {
        const response = await fetch(`/export?panorama_filename=${encodeURIComponent(selectedPanorama)}`);
        
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
        
        statusDiv.innerText = `ZIP-архив успешно скачан: ${selectedPanorama}`;

    } catch (error) {
        console.error("Ошибка при экспорте:", error);
    } finally {
        exportBtn.disabled = false;
    }
});


// Навешиваем обработчик на кнопку "Пропустить"
skipBtn.addEventListener('click', () => classify('Пропустить'));
undoBtn.addEventListener('click', undoLastClassification);
refreshReviewBtn.addEventListener('click', loadReviewMasks);
refreshVisualizationListBtn.addEventListener('click', loadVisualizationPanoramas);
reviewMaskSelect.addEventListener('change', event => loadReviewPreview(event.target.value));
applyReclassifyBtn.addEventListener('click', reclassifySelectedMask);


// --- Инициализация при загрузке страницы ---

// Создаем кнопки для всех классов
CLASS_NAMES.forEach(name => {
    const button = document.createElement('button');
    button.innerText = name;
    button.addEventListener('click', () => classify(name));
    classButtonsDiv.appendChild(button);

    const reviewOption = document.createElement('option');
    reviewOption.value = name;
    reviewOption.innerText = name;
    reviewClassSelect.appendChild(reviewOption);
});

const skipReviewOption = document.createElement('option');
skipReviewOption.value = 'Пропустить';
skipReviewOption.innerText = 'Пропустить';
reviewClassSelect.appendChild(skipReviewOption);

loadReviewMasks();

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

    try {
        const startResponse = await fetch('/start-segmentation-batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ panorama_filenames: selectedPanoramas })
        });

        const startData = await parseApiResponse(startResponse);
        if (!startResponse.ok) {
            statusDiv.innerText = `Ошибка пакетной сегментации: ${startData.detail || 'Неизвестная ошибка'}`;
            return;
        }

        statusDiv.innerText = startData.message || 'Пакетная сегментация запущена.';

        const result = await waitForSegmentationBatchCompletion();
        const logs = (result.logs || []).join('\n');
        const finalStatus = result.error
            ? `Пакетная сегментация завершилась с ошибкой: ${result.error}`
            : 'Пакетная сегментация завершена.';

        statusDiv.innerText = logs ? `${finalStatus}\n${logs}` : finalStatus;
        await loadSegmentedPanoramas();
        await loadVisualizationPanoramas();
    } catch (error) {
        statusDiv.innerText = `Ошибка пакетной сегментации: ${error}`;
    } finally {
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
        await loadReviewMasks();
        await loadVisualizationPanoramas();
    } catch (error) {
        statusDiv.innerText = `Ошибка запуска классификации: ${error}`;
    } finally {
        startClassificationBatchBtn.disabled = false;
    }
});
