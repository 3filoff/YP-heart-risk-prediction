"""
FastAPI-сервис для предсказания риска сердечного приступа
"""

import json
import io
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from catboost import CatBoostClassifier

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Конфигурация
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# Создаем необходимые директории
STATIC_DIR.mkdir(exist_ok=True)

# Загрузка метаданных модели
try:
    FEATURES = json.loads((MODEL_DIR / "model_features.json").read_text(encoding="utf-8"))
    THRESHOLD_DATA = json.loads((MODEL_DIR / "optimal_threshold.json").read_text(encoding="utf-8"))
    THRESHOLD = float(THRESHOLD_DATA["threshold"])  # Преобразуем в float
    logger.info(f"Загружены {len(FEATURES)} признаков, порог: {THRESHOLD}")
except FileNotFoundError as e:
    logger.error(f"Ошибка загрузки метаданных модели: {e}")
    # Создаем заглушки для разработки
    FEATURES = []
    THRESHOLD = 0.5
    logger.warning("Используются заглушки данных для разработки")

# Инициализация препроцессора
try:
    from src.preprocessing.data_preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor(
        drop_leaky_features=True,
        add_missing_anamnesis_flag=True
    )
    logger.info("Препроцессор успешно инициализирован")
except Exception as e:
    logger.error(f"Ошибка инициализации препроцессора: {e}")
    preprocessor = None

# Загрузка модели
try:
    model_path = MODEL_DIR / "heart_risk_model.cbm"
    if model_path.exists():
        model = CatBoostClassifier().load_model(str(model_path))
        logger.info(f"Модель успешно загружена из {model_path}")
    else:
        logger.warning("Модель не найдена, используется заглушка")
        model = None
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    model = None

# FastAPI приложение
app = FastAPI(
    title="CardioRisk API",
    description="Сервис для предсказания риска сердечного приступа на основе медицинских данных пациентов",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Раздача статических файлов
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def numpy_to_python(obj):
    """Рекурсивно преобразует numpy типы в Python типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj

def _get_risk_level(probability: float, threshold: float = THRESHOLD) -> Dict[str, str]:
    """
    Определяет уровень риска на основе вероятности
    
    Args:
        probability: Вероятность высокого риска (0.0–1.0)
        threshold: Порог классификации
        
    Returns:
        Словарь с ключами 'level' и 'label'
    """
    if probability >= 0.7:
        return {"level": "high", "label": "Высокий риск"}
    elif probability >= (threshold - 0.15):
        return {"level": "medium", "label": "Повышенный риск"}
    else:
        return {"level": "low", "label": "Низкий риск"}

def _get_recommendations(risk_level: str, probability: float) -> str:
    """
    Генерирует рекомендации на основе уровня риска
    
    Args:
        risk_level: Уровень риска (high/medium/low)
        probability: Вероятность высокого риска
        
    Returns:
        Текст рекомендаций
    """
    if risk_level == "high":
        return "Необходима срочная консультация кардиолога, ЭКГ, ЭхоКГ, анализ на тропонин."
    elif risk_level == "medium":
        return "Рекомендуется консультация терапевта, контроль артериального давления, ЭКГ."
    else:
        return "Профилактический осмотр через 6-12 месяцев, здоровый образ жизни."

@app.get("/")
async def get_homepage():
    """
    Главная страница приложения
    Просто перенаправляем на статический index.html
    """
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/predict")
async def predict_json(
    file: UploadFile = File(..., description="CSV файл с данными пациентов")
) -> Dict[str, Any]:
    """
    Возвращает предсказания в формате JSON с детальной информацией
    """
    logger.info(f"Получен запрос на предсказание от {file.filename}")
    
    return await _process_file(file, output_format="json")

@app.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(..., description="CSV файл с данными пациентов")
) -> StreamingResponse:
    """
    Возвращает предсказания в формате CSV
    """
    logger.info(f"Получен запрос на CSV предсказание от {file.filename}")
    
    return await _process_file(file, output_format="csv")

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """
    Проверка состояния сервиса
    """
    return numpy_to_python({
        "status": "healthy" if model else "no_model",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "features_count": len(FEATURES),
        "threshold": THRESHOLD,
        "version": "2.0.0"
    })

async def _process_file(
    file: UploadFile, 
    output_format: str = "json"
) -> Dict[str, Any] | StreamingResponse:
    """
    Общая логика обработки файла
    """
    # Проверка наличия модели
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Сервис временно недоступен. Модель не загружена."
        )
    
    # Валидация файла
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400, 
            detail="Поддерживаются только CSV-файлы"
        )
    
    try:
        # Чтение файла
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Проверка наличия колонки id
        if 'id' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="В файле должна быть колонка 'id'"
            )
        
        # Предобработка данных
        df_processed = preprocessor.transform(df)
        
        # Проверка признаков
        missing_features = set(FEATURES) - set(df_processed.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Не хватает признаков: {sorted(missing_features)}"
            )
        
        # Предсказание
        proba = model.predict_proba(df_processed[FEATURES])[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        
        if output_format == "csv":
            # Формирование CSV ответа
            result_df = pd.DataFrame({
                "id": df["id"],
                "prediction": pred
            })
            
            stream = io.StringIO()
            result_df.to_csv(stream, index=False)
            stream.seek(0)
            
            return StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=submission.csv"
                }
            )
        
        else:  # JSON формат
            # Формирование детализированного ответа
            results = []
            for i in range(len(df)):
                risk_info = _get_risk_level(proba[i])
                results.append({
                    "id": int(df.iloc[i]["id"]),
                    "prediction": int(pred[i]),
                    "probability_high_risk": float(proba[i]),
                    "risk_level": risk_info["level"],
                    "risk_level_label": risk_info["label"],
                    "recommendations": _get_recommendations(risk_info["level"], proba[i])
                })
            
            response_data = {
                "predictions": results,
                "metadata": {
                    "total_patients": len(results),
                    "high_risk_count": int(sum(pred)),
                    "low_risk_count": int(len(pred) - sum(pred)),
                    "threshold": float(THRESHOLD)
                }
            }
            
            # Преобразуем все numpy типы
            return numpy_to_python(response_data)
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Файл пустой")
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Неверная кодировка файла (ожидается UTF-8)")
    
    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"{request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Time: {process_time:.3f}s"
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    
    print("━" * 75)
    print("🚀 CardioRisk Prediction System")
    print("━" * 75)
    print(f"📊 Загружено признаков : {len(FEATURES)}")
    print(f"⚖️ Порог классификации : {THRESHOLD}")
    print(f"🤖 Модель загружена    : {'Да' if model else 'Нет'}")
    print(f"🔄 Препроцессор        : {'Да' if preprocessor else 'Нет'}")
    print("━" * 75)
    print("🌐 Веб-интерфейс       : http://localhost:8000")
    print("📖 Документация API    : http://localhost:8000/api/docs")
    print("📊 Проверка состояния  : http://localhost:8000/api/health")
    print("━" * 75)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )