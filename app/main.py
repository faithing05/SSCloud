from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from .processor import PanoramaProcessor
from .sam_loader import MASK_GENERATOR

# --- Модели данных для запросов ---
class StartRequest(BaseModel):
    panorama_filename: str

class ClassifyRequest(BaseModel):
    mask_name: str
    class_name: str

# --- Инициализация FastAPI ---
app = FastAPI()

# Переменная для хранения нашего обработчика панорамы
# Важно: в таком виде сервер может обрабатывать только одну панораму одновременно.
# Для многопользовательского режима потребуется более сложная логика сессий.
processor_instance = None

# --- API Эндпоинты ---

@app.get("/")
def read_root():
    return {"message": "Сервер SSCloud запущен. Перейдите на /index.html для интерфейса."}

@app.post("/start-processing")
def start_processing(request: StartRequest):
    """Инициализирует процессор и запускает генерацию масок."""
    global processor_instance
    try:
        # Эти параметры можно вынести в конфиг
        processor_instance = PanoramaProcessor(
            panorama_filename=request.panorama_filename,
            input_dir="Vistino20241014_E57",
            output_dir="CVAT_Workspace",
            class_names=["Фон", "Земля", "Человек", "Растительность", "Транспорт", "Конструкции", "Здание", "Обстановка"]
        )
        processor_instance.generate_masks(MASK_GENERATOR)
        
        masks_to_classify = processor_instance.get_mask_files_to_classify()
        
        return {
            "message": "Генерация масок завершена.",
            "total_masks": len(masks_to_classify),
            "mask_files": masks_to_classify
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-next-mask")
def get_next_mask():
    """Возвращает данные для следующей маски в очереди."""
    if not processor_instance:
        raise HTTPException(status_code=400, detail="Процесс не запущен. Вызовите /start-processing")
    
    masks = processor_instance.get_mask_files_to_classify()
    if not masks:
        return {"message": "Все маски классифицированы!", "done": True}
    
    next_mask_name = masks[0]
    mask_data = processor_instance.get_mask_for_frontend(next_mask_name)
    
    return {
        "mask_data": mask_data,
        "remaining": len(masks)
    }

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
    if not processor_instance:
        return {"status": "Процесс не запущен."}
    return {"status": processor_instance.status}

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

@app.get("/get-e57-files")
def get_e57_files_endpoint():
    """Возвращает список .e57 файлов, доступных для обработки."""
    files = e57_processor.get_e57_file_list()
    return {"files": files}

@app.post("/process-e57")
def process_e57_endpoint(request: StartRequest): # Используем ту же модель StartRequest
    """Запускает обработку одного .e57 файла."""
    result = e57_processor.process_e57_file(request.panorama_filename)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

# --- Раздача статических файлов ---
# Эта строка монтирует папку 'static' в корень сайта.
# Теперь, если запросить /index.html, FastAPI отдаст файл app/static/index.html
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")