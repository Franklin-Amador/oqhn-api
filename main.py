import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from inference import model_session
from openaq import fetch_latest_readings_with_lags, fetch_location_info
from schemas import (
    LivePredictionResponse,
    ModelInfo,
    PredictionResult,
    SensorInput,
)
from stations import get_all_predictions, get_stations_geojson

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lazy load: el modelo se descarga en el primer request que lo necesite.
    # Esto evita el timeout de startup en Vercel serverless (10s limit).
    yield


app = FastAPI(
    title="Air Quality Classifier API",
    description=(
        "Clasificador de calidad del aire con modelo ONNX cargado desde Vercel Blob. "
        "Predice la categoría AQI 6h en el futuro a partir de lecturas de OpenAQ (Honduras)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://oqhn-frontend.vercel.app",
        "http://127.0.0.1:4321",   # dev local
        "http://localhost:4321",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Status"])
async def health():
    return {
        "status":       "ok" if model_session.is_loaded else "loading",
        "model_loaded": model_session.is_loaded,
        "model_url":    settings.model_url,
        "model_version": (model_session.metadata.get("version") if model_session.is_loaded else None),
        "n_features":    (model_session.metadata.get("n_features") if model_session.is_loaded else None),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }


# ─── Model info ───────────────────────────────────────────────────────────────

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    if not model_session.is_loaded:
        raise HTTPException(503, "Modelo aún cargando")
    meta = model_session.metadata
    return ModelInfo(
        model_name=  meta["model_name"],
        version=     meta["version"],
        task=        meta["task"],
        classes=     meta["classes"],
        features=    meta["features"],
        metrics=     meta["metrics"],
        data_source= meta["data_source"],
        created_at=  meta["created_at"],
    )


# ─── Predicción manual ────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(body: SensorInput):
    """
    Predice categoría AQI a partir de lecturas de sensores enviadas manualmente.
    El preprocessing (imputer + scaler) está embebido en el pipeline ONNX.
    """
    await model_session.load()
    if not model_session.is_loaded:
        raise HTTPException(503, "Modelo aún cargando")

    features = body.to_feature_array()
    result   = model_session.predict(features)
    return PredictionResult(**result)


# ─── Predicción en vivo desde OpenAQ ─────────────────────────────────────────

@app.get("/predict/live/{location_id}", response_model=LivePredictionResponse, tags=["Prediction"])
async def predict_live(location_id: int):
    """
    Obtiene lecturas actuales de una estación OpenAQ y predice su categoría AQI.
    """
    await model_session.load()
    if not model_session.is_loaded:
        raise HTTPException(503, "Modelo aún cargando")

    readings, loc_info = await _fetch_station_data(location_id)

    now = datetime.now(timezone.utc)
    sensor_input = SensorInput(
        pm25=             readings.get("pm25", 0.0),
        pm1=              readings.get("pm1",  0.0),
        temperature=      readings.get("temperature", 25.0),
        relativehumidity= readings.get("relativehumidity", 60.0),
        um003=            readings.get("um003", 0.0),
        hour_of_day= now.hour,
        day_of_week= now.weekday(),
        month=       now.month,
        is_weekend=  int(now.weekday() >= 5),
        is_rush_hour=int(now.hour in [7, 8, 9, 17, 18, 19]),
    )

    result = model_session.predict(sensor_input.to_feature_array(lag_values=readings))

    # Limpiar metadata interna que no queremos exponer en la respuesta
    public_readings = {k: v for k, v in readings.items() if not k.startswith("_")}

    return LivePredictionResponse(
        location_id=   location_id,
        location_name= loc_info.get("name") if loc_info else None,
        sensor_readings=public_readings,
        prediction=    PredictionResult(**result),
    )


# ─── Estaciones (catálogo + predicciones agregadas) ─────────────────────────

@app.get("/stations", tags=["Stations"])
async def stations_geojson(refresh: bool = False):
    """
    GeoJSON FeatureCollection con todas las estaciones de Honduras que tienen
    coordenadas. Cacheado en memoria 1h. `?refresh=true` fuerza refetch.
    """
    try:
        return await get_stations_geojson(force_refresh=refresh)
    except Exception as e:
        logger.error("Error fetching stations: %s", e)
        raise HTTPException(502, f"OpenAQ falló: {e}")


def _build_sensor_input(readings: dict, now) -> SensorInput:
    return SensorInput(
        pm25=             readings.get("pm25", 0.0),
        pm1=              readings.get("pm1",  0.0),
        temperature=      readings.get("temperature", 25.0),
        relativehumidity= readings.get("relativehumidity", 60.0),
        um003=            readings.get("um003", 0.0),
        hour_of_day= now.hour,
        day_of_week= now.weekday(),
        month=       now.month,
        is_weekend=  int(now.weekday() >= 5),
        is_rush_hour=int(now.hour in [7, 8, 9, 17, 18, 19]),
    )


@app.get("/stations/predictions", tags=["Stations"])
async def stations_predictions(refresh: bool = False):
    """
    Predicciones live para todas las estaciones HN. Cacheado 6h.
    Primer hit (cache miss): ~4 min. Siguientes hits: instantáneos.
    `?refresh=true` fuerza refetch.
    """
    await model_session.load()
    if not model_session.is_loaded:
        raise HTTPException(503, "Modelo aún cargando")
    return await get_all_predictions(model_session, _build_sensor_input,
                                     force_refresh=refresh)


# ─── Predicción batch ─────────────────────────────────────────────────────────

@app.post("/predict/batch", response_model=list[PredictionResult], tags=["Prediction"])
async def predict_batch(body: list[SensorInput]):
    """Predice múltiples lecturas en una sola llamada (máx. 100)."""
    await model_session.load()
    if not model_session.is_loaded:
        raise HTTPException(503, "Modelo aún cargando")
    if len(body) > 100:
        raise HTTPException(400, "Máximo 100 registros por batch")

    return [PredictionResult(**model_session.predict(item.to_feature_array())) for item in body]


# ─── Helper ───────────────────────────────────────────────────────────────────

async def _fetch_station_data(location_id: int):
    try:
        readings  = await fetch_latest_readings_with_lags(location_id)
        loc_info  = await fetch_location_info(location_id)
    except Exception as e:
        logger.warning("Error al consultar OpenAQ para location %d: %s", location_id, e)
        raise HTTPException(502, f"No se pudo obtener datos de OpenAQ: {e}")
    if not readings:
        raise HTTPException(404, f"Sin datos disponibles para location_id={location_id}")
    return readings, loc_info
