import os
import re
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import processor
from . import sam_loader
from . import e57_processor

# --- Модели данных для запросов ---
class StartRequest(BaseModel):
    panorama_filename: str

class ClassifyRequest(BaseModel):
    mask_name: str
    class_name: str

class ProcessRequest(BaseModel):
    filenames: List[str]


class PanoramaBatchRequest(BaseModel):
    panorama_filenames: List[str]

# --- Инициализация FastAPI ---
app = FastAPI()

INPUT_DIR = "/workspace/SSCloud/Data_Input"
OUTPUT_DIR = "/workspace/SSCloud/Data_Output"
CLASS_NAMES = ["Фон", "Земля", "Человек", "Растительность", "Транспорт", "Конструкции", "Здание", "Обстановка"]

processor_instance: Optional[processor.PanoramaProcessor] = None
classification_queue: List[str] = []
classification_index = 0
segmentation_status = "Процесс не запущен."
segmentation_results: Dict[str, Dict[str, str]] = {}


def _natural_sort(values: List[str]) -> List[str]:
    def natural_sort_key(value: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", value)]

    return sorted(values, key=natural_sort_key)


def _build_processor(panorama_filename: str) -> processor.PanoramaProcessor:
    return processor.PanoramaProcessor(
        panorama_filename=panorama_filename,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        class_names=CLASS_NAMES,
    )


def _list_available_panoramas() -> List[str]:
    if not os.path.exists(INPUT_DIR):
        return []

    jpg_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg"))]
    normals = [f for f in jpg_files if f.lower().endswith("_normals.jpg")]
    if normals:
        return _natural_sort(normals)

    return _natural_sort(jpg_files)


def _list_segmented_panoramas() -> List[str]:
    if not os.path.exists(OUTPUT_DIR):
        return []

    segmented = []
    for panorama_name in _list_available_panoramas():
        panorama_base = os.path.splitext(panorama_name)[0]
        masks_dir = os.path.join(OUTPUT_DIR, panorama_base, "1_generated_masks")
        if os.path.exists(masks_dir):
            has_masks = any(filename.endswith(".png") for filename in os.listdir(masks_dir))
            if has_masks:
                segmented.append(panorama_name)

    return segmented


def _set_current_processor_for_queue() -> bool:
    global processor_instance

    if classification_index >= len(classification_queue):
        processor_instance = None
        return False

    processor_instance = _build_processor(classification_queue[classification_index])
    return True

# --- API Эндпоинты ---

@app.get("/")
def read_root():
    return {"message": "Сервер SSCloud запущен. Перейдите на /index.html для интерфейса."}

@app.get("/get-e57-files")
def get_e57_files_endpoint():
    """Возвращает список .e57 файлов, доступных для обработки."""
    files = e57_processor.get_e57_file_list()
    return {"files": files}


@app.get("/get-panorama-files")
def get_panorama_files_endpoint():
    """Возвращает список JPG-панорам для сегментации."""
    return {"files": _list_available_panoramas()}


@app.get("/get-segmented-panoramas")
def get_segmented_panoramas_endpoint():
    """Возвращает список всех доступных панорам для пакетной классификации."""
    return {"files": _list_available_panoramas()}

@app.post("/process-e57")
def process_e57_endpoint(request: ProcessRequest): # Используем новую модель
    """Запускает обработку одного или нескольких .e57 файлов."""
    full_log = []
    
    # Обрабатываем каждый файл из полученного списка
    for filename in request.filenames:
        result = e57_processor.process_e57_file(filename)
        if result.get("logs"):
            full_log.extend(result["logs"])
            full_log.append("-" * 20) # Добавляем разделитель

    # Возвращаем общий лог
    return {"status": "complete", "logs": full_log}

@app.post("/start-processing")
def start_processing(request: StartRequest):
    global classification_index
    global classification_queue
    global processor_instance
    global segmentation_status
    try:
        segmentation_status = f"Сегментация панорамы {request.panorama_filename}..."
        processor_instance = _build_processor(request.panorama_filename)
        processor_instance.generate_masks(sam_loader.MASK_GENERATOR)

        classification_queue = [request.panorama_filename]
        classification_index = 0

        masks_to_classify = processor_instance.get_mask_files_to_classify()
        segmentation_status = "Сегментация завершена."

        return {
            "message": "Генерация масок завершена.",
            "total_masks": len(masks_to_classify),
            "mask_files": masks_to_classify,
            "current_panorama": request.panorama_filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-segmentation-batch")
def start_segmentation_batch(request: PanoramaBatchRequest):
    """Запускает последовательную сегментацию выбранных панорам."""
    global segmentation_results
    global segmentation_status

    panorama_filenames = _natural_sort(list(dict.fromkeys(request.panorama_filenames)))
    if not panorama_filenames:
        raise HTTPException(status_code=400, detail="Список панорам пуст.")

    segmentation_results = {}
    batch_logs: List[str] = []

    for index, panorama_filename in enumerate(panorama_filenames, start=1):
        segmentation_status = f"Сегментация {index}/{len(panorama_filenames)}: {panorama_filename}"
        try:
            current_processor = _build_processor(panorama_filename)
            current_processor.generate_masks(sam_loader.MASK_GENERATOR)
            mask_count = len(current_processor.get_mask_files_to_classify())
            segmentation_results[panorama_filename] = {
                "status": "success",
                "total_masks": str(mask_count),
                "message": "Сегментация выполнена",
            }
            batch_logs.append(f"{panorama_filename}: успешно, масок {mask_count}")
        except Exception as e:
            segmentation_results[panorama_filename] = {
                "status": "error",
                "total_masks": "0",
                "message": str(e),
            }
            batch_logs.append(f"{panorama_filename}: ошибка - {e}")

    segmentation_status = "Пакетная сегментация завершена."
    return {
        "message": "Пакетная сегментация завершена.",
        "processed": len(panorama_filenames),
        "results": segmentation_results,
        "logs": batch_logs,
    }


@app.post("/start-classification-batch")
def start_classification_batch(request: PanoramaBatchRequest):
    """Запускает очередь классификации по выбранным сегментированным панорамам."""
    global classification_index
    global classification_queue
    global processor_instance

    selected = _natural_sort(list(dict.fromkeys(request.panorama_filenames)))
    if not selected:
        raise HTTPException(status_code=400, detail="Список панорам для классификации пуст.")

    available_panoramas = set(_list_available_panoramas())
    valid_queue = [name for name in selected if name in available_panoramas]
    if not valid_queue:
        raise HTTPException(status_code=400, detail="Нет валидных панорам для классификации.")

    classification_queue = valid_queue
    classification_index = 0
    _set_current_processor_for_queue()

    return {
        "message": "Очередь классификации запущена.",
        "total_panoramas": len(classification_queue),
        "queue": classification_queue,
    }

@app.get("/get-next-mask")
def get_next_mask():
    """Возвращает данные для следующей маски в очереди."""
    global classification_index

    if not classification_queue and not processor_instance:
        raise HTTPException(status_code=400, detail="Процесс не запущен. Вызовите /start-processing или /start-classification-batch")

    if not processor_instance and classification_queue:
        _set_current_processor_for_queue()

    while classification_index < len(classification_queue):
        if not processor_instance:
            _set_current_processor_for_queue()

        if not processor_instance:
            break

        masks = processor_instance.get_mask_files_to_classify()
        if masks:
            next_mask_name = masks[0]
            mask_data = processor_instance.get_mask_for_frontend(next_mask_name)

            return {
                "mask_data": mask_data,
                "remaining": len(masks),
                "current_panorama": classification_queue[classification_index],
                "panorama_progress": {
                    "current": classification_index + 1,
                    "total": len(classification_queue),
                },
            }

        classification_index += 1
        if classification_index < len(classification_queue):
            _set_current_processor_for_queue()

    return {"message": "Все маски классифицированы!", "done": True}

@app.post("/classify-mask")
def classify_mask_endpoint(request: ClassifyRequest):
    """Классифицирует одну маску."""
    if not processor_instance:
        raise HTTPException(status_code=400, detail="Процесс не запущен.")
        
    result = processor_instance.classify_mask(request.mask_name, request.class_name)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return {"message": f"Маска {request.mask_name} обработана."}

@app.get("/status")
def get_status():
    """Возвращает текущий статус обработки."""
    if processor_instance:
        return {"status": processor_instance.status}
    return {"status": segmentation_status}

@app.get("/visualize")
def visualize_endpoint():
    """Собирает и визуализирует финальную маску."""
    if not processor_instance:
        raise HTTPException(status_code=400, detail="Процесс не запущен.")
    
    visualization_data = processor_instance.visualize_final_mask()
    if not visualization_data:
        raise HTTPException(status_code=404, detail="Нет классифицированных масок для визуализации.")
        
    return visualization_data

@app.get("/export")
def export_endpoint():
    """Собирает финальный датасет и возвращает ZIP-архив для скачивания."""
    if not processor_instance:
        raise HTTPException(status_code=400, detail="Процесс не запущен.")
    
    try:
        # Вызываем метод, который создает и XML, и ZIP
        zip_filepath = processor_instance.create_final_dataset()
        if not zip_filepath or not os.path.exists(zip_filepath):
            raise HTTPException(status_code=404, detail="Не удалось создать или найти ZIP-архив.")
        
        # FastAPI автоматически создаст правильные заголовки, чтобы браузер скачал файл
        return FileResponse(path=zip_filepath, media_type='application/zip', filename=os.path.basename(zip_filepath))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при экспорте: {e}")

# --- Раздача статических файлов ---
# Эта строка монтирует папку 'static' в корень сайта.
# Теперь, если запросить /index.html, FastAPI отдаст файл app/static/index.html
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
