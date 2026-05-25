import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import pandas as pd

from config import settings

# Parámetros disponibles en Honduras (OpenAQ v3 - Sustenta Honduras)
PARAM_NAMES = ["pm25", "pm10", "pm1", "temperature", "relativehumidity", "um003"]

_LAG_COLS  = ["pm25", "pm1"]   # pm10 excluido — 100% NaN en Sustenta Honduras
_LAG_STEPS = [1, 2, 3, 6, 12, 24]
_DIFF_COLS = ["pm25", "pm1"]   # rate of change 1h


# ── Rate-limiter de token-bucket ─────────────────────────────────────────────
# Un semáforo solo limita concurrencia; no evita ráfagas que disparan el 429.
# El token-bucket garantiza que no hagamos más de N req/seg sin importar cuántas
# corutinas estén corriendo en paralelo.
#
# 4 req/seg es conservador para el plan con API-key de OpenAQ (60 req/min = 1/seg
# sin key, ≥10/seg con key). Ajustar OPENAQ_RATE_PER_SEC si tienes plan superior.
_OPENAQ_RATE_PER_SEC: float = 2  # 2 req/s — con API key OpenAQ free tier aguanta ~120 req/min


class _TokenBucket:
    def __init__(self, rate: float):
        self._interval = 1.0 / rate   # segundos mínimos entre requests
        self._last     = 0.0
        self._lock     = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now  = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


_bucket = _TokenBucket(_OPENAQ_RATE_PER_SEC)


def _headers() -> dict:
    h = {"Accept": "application/json"}
    if settings.openaq_api_key:
        h["X-API-Key"] = settings.openaq_api_key
    return h


async def _get(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
) -> httpx.Response:
    """
    GET con rate-limit real (token bucket) + 1 reintento en 429.
    El bucket garantiza máximo _OPENAQ_RATE_PER_SEC requests/segundo globales.
    """
    await _bucket.acquire()
    resp = await client.get(url, headers=_headers(), params=params)
    if resp.status_code == 429:
        # Si aún así llegó un 429, esperamos 10 s y reintentamos una vez
        await asyncio.sleep(10)
        await _bucket.acquire()
        resp = await client.get(url, headers=_headers(), params=params)
    return resp


# ─── Metadatos de estación ────────────────────────────────────────────────────

async def fetch_location_data(location_id: int) -> tuple[dict[str, int], dict | None]:
    """
    UNA sola llamada a GET /locations/{id} → sensor_map + location_info.
    Evita el doble fetch anterior (fetch_sensors_for_location + fetch_location_info).
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await _get(client, f"{settings.openaq_api_url}/locations/{location_id}")
        if resp.status_code != 200:
            return {}, None
        results = resp.json().get("results", [])
        if not results:
            return {}, None

        loc        = results[0]
        sensor_map = {
            s["parameter"]["name"]: s["id"]
            for s in loc.get("sensors", [])
            if s.get("parameter", {}).get("name") in PARAM_NAMES
        }
        coords       = loc.get("coordinates") or {}
        location_info = {
            "id":        loc.get("id"),
            "name":      loc.get("name", f"Station {location_id}"),
            "country":   (loc.get("country") or {}).get("name", ""),
            "city":      loc.get("locality", ""),
            "latitude":  coords.get("latitude"),
            "longitude": coords.get("longitude"),
        }
        return sensor_map, location_info


# Aliases retrocompatibles
async def fetch_sensors_for_location(location_id: int) -> dict[str, int]:
    sensor_map, _ = await fetch_location_data(location_id)
    return sensor_map

async def fetch_location_info(location_id: int) -> Optional[dict]:
    _, loc_info = await fetch_location_data(location_id)
    return loc_info


# ─── Series temporales de sensores ───────────────────────────────────────────

def _measurements_to_hours(raw: list[dict]) -> list[dict]:
    """Medidas individuales → bins horarios (resample 1h, media)."""
    records = []
    for r in raw:
        dt_str = (r.get("period") or {}).get("datetimeFrom", {}).get("utc") or \
                 (r.get("date") or {}).get("utc")
        val = r.get("value")
        if dt_str is not None and val is not None:
            try:
                records.append({"dt": pd.Timestamp(dt_str, tz="UTC"), "value": float(val)})
            except (ValueError, TypeError):
                pass
    if not records:
        return []
    df = (
        pd.DataFrame(records).set_index("dt").sort_index()["value"]
        .resample("h").mean().dropna()
    )
    return [
        {"period": {"datetimeFrom": {"utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ")}},
         "value": float(val)}
        for ts, val in df.items()
    ]


async def _fetch_sensor_hours(sensor_id: int, hours_back: int = 25) -> list[dict]:
    """
    Datos horarios de un sensor en la ventana indicada.
    1. Intenta /hours (agr. horario de OpenAQ).
    2. Si vacío, cae a /measurements y agrega a 1h.
    """
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(hours=hours_back)
    fmt       = "%Y-%m-%dT%H:%M:%SZ"
    window    = {"datetime_from": date_from.strftime(fmt), "datetime_to": date_to.strftime(fmt)}

    results: list[dict] = []
    async with httpx.AsyncClient(timeout=30) as client:
        page = 1
        while True:
            resp = await _get(client,
                f"{settings.openaq_api_url}/sensors/{sensor_id}/hours",
                params={**window, "limit": 100, "page": page})
            if resp.status_code != 200:
                break
            batch = resp.json().get("results", [])
            results.extend(batch)
            if len(batch) < 100:
                break
            page += 1
            if page > 5:
                break

    if results:
        return results

    # Fallback: /measurements → resamplear
    raw: list[dict] = []
    async with httpx.AsyncClient(timeout=30) as client:
        page = 1
        while True:
            resp = await _get(client,
                f"{settings.openaq_api_url}/sensors/{sensor_id}/measurements",
                params={**window, "limit": 1000, "page": page})
            if resp.status_code != 200:
                break
            batch = resp.json().get("results", [])
            raw.extend(batch)
            if len(batch) < 1000:
                break
            page += 1
            if page > 3:
                break

    return _measurements_to_hours(raw)


async def _fetch_pm25_latest(sensor_id: int) -> dict | None:
    """
    Última lectura de pm25 sin filtro de fecha (fallback stale).
    Solo para pm25 — es el único que necesitamos para colorear el pin.
    Descarta si es > 7 días.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await _get(client,
            f"{settings.openaq_api_url}/sensors/{sensor_id}/measurements",
            params={"limit": 1, "page": 1})
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if not results:
            return None

    r      = results[0]
    dt_str = (r.get("period") or {}).get("datetimeFrom", {}).get("utc") or \
             (r.get("date") or {}).get("utc")
    val    = r.get("value")
    if dt_str is None or val is None:
        return None
    try:
        ts  = pd.Timestamp(dt_str, tz="UTC")
        age = datetime.now(timezone.utc) - ts.to_pydatetime()
        if age > timedelta(days=7):
            return None
        return {"value": float(val), "age_h": age.total_seconds() / 3600}
    except Exception:
        return None


# ─── Construcción del vector de features ─────────────────────────────────────

def _series_to_features(series_map: dict[str, pd.Series]) -> dict:
    df     = pd.DataFrame(series_map).sort_index()
    latest = df.iloc[-1]
    result: dict = {p: float(latest.get(p, 0.0)) for p in PARAM_NAMES}
    result["_latest_timestamp"] = str(df.index[-1])
    result["_rows_fetched"]     = len(df)
    result["_stale"]            = False

    for col in _LAG_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        for lag in _LAG_STEPS:
            result[f"{col}_lag{lag}h"] = float(s.iloc[-lag - 1]) if len(s) > lag else 0.0

    # Diff features: rate of change 1h (current - lag1h)
    for col in _DIFF_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        current = float(s.iloc[-1])
        lag1    = float(s.iloc[-2]) if len(s) > 1 else current
        result[f"{col}_diff1h"] = round(current - lag1, 4)

    # Rolling features (pm25 y pm1; pm10 excluido)
    for col in ["pm25", "pm1"]:
        if col not in df.columns:
            continue
        s = df[col]
        result[f"{col}_roll4h_mean"]  = float(s.iloc[-4:].mean())  if len(s) >= 4  else float(s.mean())
        result[f"{col}_roll24h_mean"] = float(s.iloc[-24:].mean()) if len(s) >= 24 else float(s.mean())
    return result


# ─── API pública ──────────────────────────────────────────────────────────────

async def fetch_readings_for_sensor_map(sensor_map: dict[str, int]) -> dict:
    """
    Obtiene datos de la estación intentando 3 estrategias en orden:

    1. pm25-first check (1 req): si pm25 no tiene datos en 25h → saltar a stale.
       Evita hacer 5-10 llamadas extra por sensor para estaciones sin datos.
    2. Fetch completo: si pm25 tiene datos, traer el resto de sensores.
    3. Stale fallback: última lectura pm25 disponible (≤7 días), lags=0.

    Total de llamadas por estación:
      - Con datos recientes : ~6 (1 check + 5 sensores restantes)
      - Sin datos recientes : ~2 (1 check /hours vacío + 1 stale /measurements)
    """
    if not sensor_map:
        return {}

    pm25_id = sensor_map.get("pm25")
    if pm25_id is None:
        return {}  # sin pm25 no podemos predecir

    # ── 1. Check rápido de pm25 ───────────────────────────────────────────────
    pm25_raw = await _fetch_sensor_hours(pm25_id, hours_back=25)

    if not pm25_raw:
        # pm25 sin datos recientes → toda la estación está sin datos.
        # Saltar directo al fallback stale (ahorra 5-10 llamadas a OpenAQ).
        snap = await _fetch_pm25_latest(pm25_id)
        if snap is None:
            return {}

        pm25_val = snap["value"]
        result: dict = {p: 0.0 for p in PARAM_NAMES}
        result["pm25"]              = pm25_val
        result["_latest_timestamp"] = None
        result["_rows_fetched"]     = 0
        result["_stale"]            = True
        result["_stale_age_h"]      = round(snap["age_h"], 1)
        for col in _LAG_COLS:
            for lag in _LAG_STEPS:
                result[f"{col}_lag{lag}h"] = 0.0
        for col in _DIFF_COLS:
            result[f"{col}_diff1h"] = 0.0
        result["pm25_roll4h_mean"]  = pm25_val
        result["pm25_roll24h_mean"] = pm25_val
        result["pm1_roll4h_mean"]   = 0.0
        result["pm1_roll24h_mean"]  = 0.0
        return result

    # ── 2. pm25 tiene datos → fetch del resto de sensores ────────────────────
    pm25_records = {
        e["period"]["datetimeFrom"]["utc"]: e.get("value", float("nan"))
        for e in pm25_raw if "period" in e
    }
    series_map: dict[str, pd.Series] = {"pm25": pd.Series(pm25_records, dtype=float)}

    for param, sensor_id in sensor_map.items():
        if param == "pm25":
            continue  # ya lo tenemos
        raw = await _fetch_sensor_hours(sensor_id, hours_back=25)
        if raw:
            records = {
                e["period"]["datetimeFrom"]["utc"]: e.get("value", float("nan"))
                for e in raw if "period" in e
            }
            if records:
                series_map[param] = pd.Series(records, dtype=float)

    return _series_to_features(series_map)


async def fetch_latest_readings_with_lags(location_id: int) -> dict:
    """Wrapper de conveniencia para endpoints manuales /predict."""
    sensor_map, _ = await fetch_location_data(location_id)
    return await fetch_readings_for_sensor_map(sensor_map)
