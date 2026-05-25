from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)

# Orden exacto de ALL_FEATURES del notebook (34 features total):
# SENSOR_FEATURES (7) + TIME_FEATURES (5) + LAG_FEATURES (18) + ROLLING_FEATURES (4)
_LAG_COLS  = ["pm25", "pm10", "pm1"]
_LAG_STEPS = [1, 2, 3, 6, 12, 24]


class SensorInput(BaseModel):
    # Sensores disponibles en Honduras (OpenAQ v3 - Sustenta Honduras)
    pm25: float = Field(..., ge=0, description="PM2.5 µg/m³")
    pm10: float = Field(..., ge=0, description="PM10 µg/m³")
    pm1: float = Field(0.0, ge=0, description="PM1 µg/m³")
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
        return self.pm25 / (self.pm10 + 1e-6)

    def to_feature_array(self, lag_values: dict | None = None) -> list[float]:
        """
        Construye el vector de 34 features en el orden exacto de ALL_FEATURES.
        lag_values: dict con keys como 'pm25_lag1h', 'pm25_roll4h_mean', etc.
                    Si es None (endpoint manual), los lags se dejan en 0.0 y el
                    SimpleImputer del pipeline ONNX los rellena con la mediana.
        """
        lags = lag_values or {}

        # 7 sensor features
        base = [self.pm25, self.pm10, self.pm1, self.temperature,
                self.relativehumidity, self.um003, self.pm_ratio]
        # 5 time features
        time = [self.hour_of_day, self.day_of_week, self.month,
                self.is_weekend, self.is_rush_hour]
        # 18 lag features (pm25/pm10/pm1 × [1,2,3,6,12,24]h)
        lag_feats = [lags.get(f"{col}_lag{lag}h", 0.0)
                     for col in _LAG_COLS for lag in _LAG_STEPS]
        # 4 rolling features
        roll_feats = [
            lags.get("pm25_roll4h_mean",  self.pm25),
            lags.get("pm25_roll24h_mean", self.pm25),
            lags.get("pm10_roll4h_mean",  self.pm10),
            lags.get("pm10_roll24h_mean", self.pm10),
        ]
        return base + time + lag_feats + roll_feats


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
