import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
import onnxruntime as rt

from config import settings

logger = logging.getLogger(__name__)
_load_lock = asyncio.Lock()

# AQI display helpers
AQI_COLORS = {
    "Good":                    "#00E400",
    "Moderate":                "#FFFF00",
    "Unhealthy for Sensitive": "#FF7E00",
    "Unhealthy":               "#FF0000",
    "Very Unhealthy":          "#8F3F97",
}

AQI_MESSAGES = {
    "Good":                    "La calidad del aire es satisfactoria. Actividad al aire libre sin restricciones.",
    "Moderate":                "Calidad aceptable. Personas muy sensibles deben reducir actividad prolongada al aire libre.",
    "Unhealthy for Sensitive": "Grupos sensibles pueden verse afectados. Reducir actividad al aire libre.",
    "Unhealthy":               "Todos pueden comenzar a sentir efectos. Limitar actividad prolongada al aire libre.",
    "Very Unhealthy":          "Alerta de salud. Todos deben evitar actividad al aire libre.",
}


class ModelSession:
    """Singleton que mantiene el modelo ONNX cargado en memoria."""

    def __init__(self):
        self._session: Optional[rt.InferenceSession] = None
        self._metadata: Optional[dict] = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[list[str]] = None
        self._classes: Optional[list[str]] = None

    async def load(self) -> None:
        """Descarga modelo y metadata desde Vercel Blob (lazy — solo si no está cargado)."""
        if self._session is not None:
            return
        async with _load_lock:
            if self._session is not None:
                return
            logger.info("Descargando modelo ONNX desde Vercel Blob...")
            async with httpx.AsyncClient(timeout=60) as client:
                model_resp = await client.get(settings.model_url)
                model_resp.raise_for_status()

                meta_resp = await client.get(settings.metadata_url)
                meta_resp.raise_for_status()

            self._metadata = json.loads(meta_resp.content)
            self._classes  = self._metadata["classes"]

            model_bytes = model_resp.content
            self._session = rt.InferenceSession(
                model_bytes,
                providers=["CPUExecutionProvider"]
            )
            self._input_name  = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]

            logger.info(
                "Modelo cargado. Clases: %s | Features: %d",
                self._classes,
                len(self._metadata["features"])
            )

    def predict(self, features: list[float]) -> dict:
        """
        Infiere sobre un vector de features.
        Retorna clase, confianza y probabilidades por clase.
        """
        if self._session is None:
            raise RuntimeError("Modelo no cargado. Espera al startup.")

        x = np.array([features], dtype=np.float32)
        outputs = self._session.run(self._output_names, {self._input_name: x})

        # outputs[0] = label int, outputs[1] = probs array shape (1, n_classes)
        label_idx = int(outputs[0][0])
        probs     = outputs[1][0].tolist()

        category = self._classes[label_idx]
        probs_by_class = {cls: round(float(p), 4) for cls, p in zip(self._classes, probs)}

        return {
            "category":       category,
            "category_index": label_idx,
            "confidence":     round(float(max(probs)), 4),
            "probabilities":  probs_by_class,
            "aqi_color":      AQI_COLORS.get(category, "#CCCCCC"),
            "health_message": AQI_MESSAGES.get(category, ""),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            raise RuntimeError("Modelo no cargado.")
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        return self._session is not None


model_session = ModelSession()
