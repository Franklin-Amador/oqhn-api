import math
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)

# Orden exacto de ALL_FEATURES del notebook v3.1.0 (35 features):
# SENSOR_FEATURES (6) + TIME_FEATURES (5) + CYCLIC_FEATURES (6)
# + LAG_FEATURES (12) + DIFF_FEATURES (2) + ROLLING_FEATURES (4)
#
# Nota: pm10 excluido — 100% NaN en la red Sustenta Honduras.
# pm_ratio = pm25 / pm1 (en vez de pm25 / pm10)

_LAG_COLS  = ["pm25", "pm1"]
_LAG_STEPS = [1, 2, 3, 6, 12, 24]
_DIFF_COLS = ["pm25", "pm1"]


class SensorInput(BaseModel):
    # Sensores disponibles en Honduras (OpenAQ v3 - Sustenta Honduras)
    pm25: float = Field(..., ge=0, description="PM2.5 µg/m³")
    pm1: float  = Field(0.0, ge=0, description="PM1 µg/m³")
    temperature: float = Field(25.0, description="Temperatura °C")
    relativehumidity: float = Field(60.0, ge=0, le=100, description="Humedad relativa %")
    um003: float = Field(0.0, ge=0, description="Conteo de partículas um003")
    hour_of_day: int = Field(default_factory=lambda: _now().hour, ge=0, le=23)
    day_of_week: int = Field(default_factory=lambda: _now().weekday(), ge=0, le=6)
    month: int = Field(default_factory=lambda: _now().month, ge=1, le=12)
    is_weekend: int = Field(default_factory=lambda: int(_now().weekday() >= 5), ge=0, le=1)
    is_rush_hour: int = Field(default_factory=lambda: int(_now().hour in [7, 8, 9, 17, 18, 19]), ge=0, le=1)

    @property
    def pm_ratio(self) -> float:
        # pm25/pm1: proporcion particulas finas vs muy finas (pm10 no disponible en Honduras)
        return self.pm25 / (self.pm1 + 1e-6)

    def to_feature_array(self, lag_values: dict | None = None) -> list[float]:
        """
        Construye el vector de 35 features en el orden exacto de ALL_FEATURES (train.py v3.1.0).
        lag_values: dict con keys como 'pm25_lag1h', 'pm1_roll4h_mean', 'pm25_diff1h', etc.
                    Si es None (endpoint manual), lags/diffs/rolling se dejan en 0.0 y el
                    SimpleImputer del pipeline ONNX los rellena con la mediana de entrenamiento.
        """
        lags = lag_values or {}

        # 1. Sensor features (6)
        base = [
            self.pm25, self.pm1, self.temperature,
            self.relativehumidity, self.um003, self.pm_ratio
        ]

        # 2. Time features (5)
        time = [
            self.hour_of_day, self.day_of_week, self.month,
            self.is_weekend, self.is_rush_hour
        ]

        # 3. Cyclic features (6): sin/cos para hora, dia de semana, mes
        cyclic = [
            math.sin(2 * math.pi * self.hour_of_day / 24),
            math.cos(2 * math.pi * self.hour_of_day / 24),
            math.sin(2 * math.pi * self.day_of_week / 7),
            math.cos(2 * math.pi * self.day_of_week / 7),
            math.sin(2 * math.pi * self.month / 12),
            math.cos(2 * math.pi * self.month / 12),
        ]

        # 4. Lag features (12): pm25/pm1 × [1,2,3,6,12,24]h
        lag_feats = [
            lags.get(f"{col}_lag{lag}h", 0.0)
            for col in _LAG_COLS for lag in _LAG_STEPS
        ]

        # 5. Diff features (2): rate of change 1h
        #    diff1h = current - lag1h (si no hay lag, ONNX imputer usa mediana de train)
        diff_feats = [
            lags.get(f"{col}_diff1h",
                     lags.get(f"{col}_current", 0.0) - lags.get(f"{col}_lag1h", 0.0)
                     if f"{col}_lag1h" in lags else 0.0)
            for col in _DIFF_COLS
        ]

        # 6. Rolling features (4): pm25/pm1 × windows [4, 24]h
        roll_feats = [
            lags.get("pm25_roll4h_mean",  self.pm25),
            lags.get("pm25_roll24h_mean", self.pm25),
            lags.get("pm1_roll4h_mean",   self.pm1),
            lags.get("pm1_roll24h_mean",  self.pm1),
        ]

        return base + time + cyclic + lag_feats + diff_feats + roll_feats


class PredictionResult(BaseModel):
    category: str
    category_index: int
    confidence: float
    probabilities: dict[str, float]
    aqi_color: str
    health_message: str
    timestamp: str


class LivePredictionResponse(BaseModel):
    location_id: int
    location_name: Optional[str]
    sensor_readings: dict
    prediction: PredictionResult


class ModelInfo(BaseModel):
    model_name: str
    version: str
    task: str
    classes: list[str]
    features: list[str]
    metrics: dict
    data_source: str
    created_at: str
