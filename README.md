# OQHN — API

FastAPI que sirve predicciones de calidad del aire (AQI) a +6 horas para Honduras, usando un modelo XGBoost exportado a ONNX y datos en tiempo real de OpenAQ v3.

## Stack

| Capa | Tecnología |
|------|-----------|
| Framework | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn |
| Modelo | ONNX Runtime (XGBoost → ONNX, opset 17) |
| Datos | [OpenAQ v3 API](https://docs.openaq.org) — red Sustenta Honduras |
| Settings | Pydantic Settings (`python-dotenv`) |
| Deploy | Vercel (Serverless) / Docker |

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del modelo cargado |
| `GET` | `/model/info` | Metadata del modelo (versión, features, clases) |
| `POST` | `/predict` | Predicción manual con JSON de sensores |
| `GET` | `/predict/live/{location_id}` | Predicción live para una estación |
| `GET` | `/stations` | GeoJSON de todas las estaciones de Honduras |
| `GET` | `/stations/predictions` | Predicciones batch (todas las estaciones, cacheado) |

### Parámetros de query

- `?refresh=true` en `/stations` y `/stations/predictions` — invalida el caché en memoria

## Desarrollo local

```bash
# 1. Crear entorno virtual
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Crear .env (ver sección de variables)
cp .env.example .env  # o crear manualmente

# 4. Levantar servidor
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

## Variables de entorno

Crear un archivo `.env` en esta carpeta (no se sube a git):

```env
BLOB_BASE_URL=https://<tu-store>.public.blob.vercel-storage.com
MODEL_BLOB_PATH=models/air_quality_classifier.onnx
METADATA_BLOB_PATH=models/model_metadata.json
OPENAQ_API_URL=https://api.openaq.org/v3
OPENAQ_API_KEY=<tu-api-key>
```

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `BLOB_BASE_URL` | Sí | URL base del Vercel Blob store donde vive el modelo |
| `MODEL_BLOB_PATH` | Sí | Path del archivo `.onnx` dentro del blob store |
| `METADATA_BLOB_PATH` | Sí | Path del `model_metadata.json` |
| `OPENAQ_API_URL` | No | Default: `https://api.openaq.org/v3` |
| `OPENAQ_API_KEY` | Recomendado | Sin key: 60 req/min. Con key: límite mayor |

## Arquitectura de caché

| Recurso | TTL | Notas |
|---------|-----|-------|
| `/stations` (catálogo) | 1 hora | Las estaciones no cambian seguido |
| `/stations/predictions` | 10 min | Coincide con el polling del frontend |

El primer hit en frío tarda ~2-3 min (rate-limit de OpenAQ con token-bucket a 5 req/s).
Los hits siguientes dentro del TTL son instantáneos.

## Rate limiting a OpenAQ

La API implementa un **token-bucket** global (`_OPENAQ_RATE_PER_SEC = 5`) que garantiza no superar 5 requests/segundo a OpenAQ independientemente de cuántas estaciones se procesen en paralelo. En caso de recibir un 429, espera 10 s y reintenta una vez.

Los `sensor_ids` se cachean en el GeoJSON de estaciones (TTL 1h) para evitar llamadas redundantes a `/locations/{id}` durante el batch de predicciones.

## Estrategia de datos por estación

```
fetch_readings_for_sensor_map(sensor_map):
  1. Intenta /sensors/{id}/hours  (25h, agr. horario)
  2. Si vacío → /sensors/{id}/measurements + resample 1h
  3. Si aún vacío → última lectura disponible de pm25 (hasta 7 días)
     → _stale=True, lags=0, predicción basada solo en lectura puntual
  4. Si nada → has_data=False (pin gris en el mapa)
```
