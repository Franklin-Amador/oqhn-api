"""
Listado de estaciones de OpenAQ (Honduras) + cache + helpers GeoJSON.

Las estaciones no cambian seguido — cacheamos el listado en memoria con TTL
para evitar fetchear el catálogo en cada request al frontend.
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import settings
from openaq import (
    PARAM_NAMES,
    _headers,
    fetch_location_data,
    fetch_readings_for_sensor_map,
)

logger = logging.getLogger(__name__)

# Honduras country_id en OpenAQ v3
HN_COUNTRY_ID = 136

# Cache simple en memoria: (timestamp, payload)
_STATIONS_CACHE: dict     = {"ts": 0.0, "data": None}
_PREDICTIONS_CACHE: dict  = {"ts": 0.0, "data": None}
_STATIONS_TTL_SEC    = 60 * 60   # 1h — el catálogo de estaciones rara vez cambia
_PREDICTIONS_TTL_SEC = 10 * 60   # 10 min — coincide con el polling del frontend


async def _fetch_honduras_locations(limit: int = 100) -> list[dict]:
    """Trae el catálogo de estaciones de Honduras desde OpenAQ."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{settings.openaq_api_url}/locations",
            headers=_headers(),
            params={"countries_id": HN_COUNTRY_ID, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json().get("results", [])


def _location_to_feature(loc: dict) -> Optional[dict]:
    """Convierte una location de OpenAQ a un Feature GeoJSON."""
    coords = loc.get("coordinates") or {}
    lat = coords.get("latitude")
    lon = coords.get("longitude")
    if lat is None or lon is None:
        return None

    sensors = [
        s.get("parameter", {}).get("name", "")
        for s in loc.get("sensors", [])
        if s.get("parameter", {}).get("name") in PARAM_NAMES
    ]

    # Guardar sensor_ids en el GeoJSON para evitar re-fetchear /locations/{id}
    # en el batch de predicciones (ahorra 56 llamadas a OpenAQ por ciclo).
    sensor_ids = {
        s.get("parameter", {}).get("name", ""): s["id"]
        for s in loc.get("sensors", [])
        if s.get("parameter", {}).get("name") in PARAM_NAMES
    }

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(lon), float(lat)],  # GeoJSON: [lon, lat]
        },
        "properties": {
            "id":         loc.get("id"),
            "name":       loc.get("name", f"Station {loc.get('id')}"),
            "locality":   loc.get("locality"),
            "country":    (loc.get("country") or {}).get("name", ""),
            "sensors":    sensors,
            "has_pm25":   "pm25" in sensors,
            "sensor_ids": sensor_ids,   # {param_name: sensor_id}
        },
    }


async def get_stations_geojson(force_refresh: bool = False) -> dict:
    """
    Devuelve un FeatureCollection GeoJSON con todas las estaciones de Honduras
    que tengan coordenadas. Cacheado 1h.
    """
    now = time.time()
    if (
        not force_refresh
        and _STATIONS_CACHE["data"] is not None
        and (now - _STATIONS_CACHE["ts"]) < _STATIONS_TTL_SEC
    ):
        return _STATIONS_CACHE["data"]

    locations = await _fetch_honduras_locations(limit=100)
    features = [f for f in (_location_to_feature(loc) for loc in locations) if f]

    payload = {
        "type":     "FeatureCollection",
        "features": features,
        "metadata": {
            "country_id":      HN_COUNTRY_ID,
            "country":         "Honduras",
            "station_count":   len(features),
            "fetched_at":      datetime.now(timezone.utc).isoformat(),
            "ttl_seconds":     _STATIONS_TTL_SEC,
        },
    }
    _STATIONS_CACHE["data"] = payload
    _STATIONS_CACHE["ts"]   = now
    return payload


async def _safe_predict_one(
    model_session,
    build_sensor_input,
    location_id: int,
    *,
    sensor_map: dict | None = None,
    loc_info:   dict | None = None,
) -> dict:
    """
    Predicción para una estación.
    Si sensor_map/loc_info vienen pre-fetcheados (desde el caché de stations),
    no se hace ninguna llamada a /locations/{id} — cero requests extra.
    """
    try:
        # Usar datos del caché o fetchear si no vienen provistos
        if sensor_map is None or loc_info is None:
            sensor_map, loc_info = await fetch_location_data(location_id)
        readings = await fetch_readings_for_sensor_map(sensor_map)

        if not readings:
            return {
                "location_id": location_id,
                "has_data":    False,
                "error":       "Sin datos disponibles",
            }

        now = datetime.now(timezone.utc)
        sensor_input = build_sensor_input(readings, now)
        result = model_session.predict(
            sensor_input.to_feature_array(lag_values=readings)
        )

        is_stale = readings.get("_stale", False)
        public_readings = {k: v for k, v in readings.items() if not k.startswith("_")}

        return {
            "location_id":    location_id,
            "location_name":  loc_info.get("name") if loc_info else None,
            "latitude":       (loc_info.get("latitude")  if loc_info else None),
            "longitude":      (loc_info.get("longitude") if loc_info else None),
            "has_data":       True,
            "stale":          is_stale,         # True = datos > 25h, lags en 0
            "stale_age_h":    readings.get("_stale_age_h"),
            "sensor_readings": public_readings,
            "prediction":     result,
        }
    except Exception as e:
        logger.warning("Predicción falló para location %d: %s", location_id, e)
        return {
            "location_id": location_id,
            "has_data":    False,
            "error":       str(e),
        }


async def get_all_predictions(model_session, build_sensor_input,
                              force_refresh: bool = False) -> dict:
    """
    Trae predicciones para todas las estaciones de Honduras (fetch en paralelo).
    Cacheado en memoria con TTL = _PREDICTIONS_TTL_SEC.

    Primer hit: lento (~3-6 min — 56 estaciones × ~7 API calls c/u).
    Hits siguientes dentro del TTL: instantáneos.

    Estaciones sin datos recientes vienen con has_data=False.
    """
    now = time.time()
    if (
        not force_refresh
        and _PREDICTIONS_CACHE["data"] is not None
        and (now - _PREDICTIONS_CACHE["ts"]) < _PREDICTIONS_TTL_SEC
    ):
        cached = dict(_PREDICTIONS_CACHE["data"])
        cached["cache_hit"]   = True
        cached["cache_age_s"] = round(now - _PREDICTIONS_CACHE["ts"], 1)
        return cached

    geojson = await get_stations_geojson()

    # Pasamos sensor_ids y loc_info directamente desde el caché de estaciones
    # → elimina 56 llamadas a /locations/{id} por ciclo de predicciones.
    station_metas = [
        {
            "location_id": f["properties"]["id"],
            "sensor_ids":  f["properties"].get("sensor_ids", {}),
            "loc_info": {
                "name":      f["properties"]["name"],
                "latitude":  f["geometry"]["coordinates"][1],
                "longitude": f["geometry"]["coordinates"][0],
            },
        }
        for f in geojson["features"]
        if f["properties"].get("has_pm25")
    ]

    # El rate-limit real está en el token-bucket de openaq.py (_bucket).
    # El semáforo aquí solo evita abrir demasiados contextos asyncio a la vez.
    sem = asyncio.Semaphore(20)

    async def _bounded(meta: dict):
        async with sem:
            return await _safe_predict_one(
                model_session, build_sensor_input,
                meta["location_id"],
                sensor_map=meta["sensor_ids"],
                loc_info=meta["loc_info"],
            )

    results = await asyncio.gather(*[_bounded(m) for m in station_metas])

    payload = {
        "fetched_at":    datetime.now(timezone.utc).isoformat(),
        "station_count": len(location_ids),
        "with_data":     sum(1 for r in results if r.get("has_data")),
        "ttl_seconds":   _PREDICTIONS_TTL_SEC,
        "cache_hit":     False,
        "cache_age_s":   0.0,
        "predictions":   results,
    }
    _PREDICTIONS_CACHE["data"] = payload
    _PREDICTIONS_CACHE["ts"]   = now
    return payload
