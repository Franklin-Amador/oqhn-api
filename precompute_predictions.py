"""
Precompute de predicciones → Vercel Blob.

Corre FUERA del request path (GitHub Action cron cada 6h), no en Vercel serverless.
Hace el trabajo pesado una sola vez por ciclo:
  1. Carga el modelo ONNX desde Blob.
  2. Trae el catálogo de estaciones de Honduras (OpenAQ).
  3. Corre el batch de predicciones para todas las estaciones (~130 requests a
     OpenAQ a 1 req/s — el token bucket de openaq.py respeta el free tier).
  4. Sube `stations.json` y `predictions.json` a Vercel Blob.

La API (main.py) luego solo LEE esos JSON desde Blob → respuestas instantáneas,
sin timeout serverless y sin martillar OpenAQ en cada visita (evita el ban).

Requiere env vars:
  OPENAQ_API_KEY         — para fetchear OpenAQ
  BLOB_READ_WRITE_TOKEN  — para subir a Vercel Blob
"""
import asyncio
import json
import os
import sys

import httpx

from config import settings
from inference import model_session
from schemas import SensorInput
from stations import get_all_predictions, get_stations_geojson

BLOB_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN", "").strip().strip("\"'")


def upload_json_to_blob(obj: dict, blob_path: str, token: str) -> dict:
    """Sube un dict como JSON a Vercel Blob (mismo patrón que el notebook)."""
    url = f"https://blob.vercel-storage.com/{blob_path}"
    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
    }
    params = {"addRandomSuffix": "false"}  # query param, sobrescribe la misma ruta
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    r = httpx.put(url, headers=headers, params=params, content=body, timeout=120)
    if r.status_code != 200:
        print(f"  Error {r.status_code}: {r.text[:300]}")
    r.raise_for_status()
    return r.json()


async def main() -> None:
    if not BLOB_TOKEN:
        sys.exit("ERROR: BLOB_READ_WRITE_TOKEN no configurado")
    if not settings.openaq_api_key:
        sys.exit("ERROR: OPENAQ_API_KEY no configurado")

    print("[1/4] Cargando modelo ONNX desde Blob...")
    await model_session.load()
    print(f"      OK — v{model_session.metadata.get('version')}")

    print("[2/4] Trayendo catálogo de estaciones (OpenAQ)...")
    stations = await get_stations_geojson(force_refresh=True)
    print(f"      {stations['metadata']['station_count']} estaciones")

    print("[3/4] Corriendo batch de predicciones (1 req/s, ~2-4 min)...")
    predictions = await get_all_predictions(
        model_session, SensorInput.from_readings, force_refresh=True
    )
    print(
        f"      {predictions['with_data']}/{predictions['station_count']} "
        f"estaciones con datos"
    )

    print("[4/4] Subiendo a Vercel Blob...")
    r1 = upload_json_to_blob(stations, settings.stations_blob_path, BLOB_TOKEN)
    print(f"      stations.json    → {r1.get('url')}")
    r2 = upload_json_to_blob(predictions, settings.predictions_blob_path, BLOB_TOKEN)
    print(f"      predictions.json → {r2.get('url')}")

    print("Listo. La API servirá estos JSON desde Blob en el próximo request.")


if __name__ == "__main__":
    asyncio.run(main())
