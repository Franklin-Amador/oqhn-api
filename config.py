from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    blob_base_url: str = "https://bdtpq2w9jokgb3pf.public.blob.vercel-storage.com"
    model_blob_path: str = "models/air_quality_classifier.onnx"
    metadata_blob_path: str = "models/model_metadata.json"
    # Predicciones precalculadas por el job precompute (GitHub Action cada 6h).
    # La API solo LEE estos JSON desde Blob — no toca OpenAQ en el request path.
    predictions_blob_path: str = "models/predictions.json"
    stations_blob_path: str = "models/stations.json"
    openaq_api_url: str = "https://api.openaq.org/v3"
    openaq_api_key: str = ""

    @property
    def model_url(self) -> str:
        return f"{self.blob_base_url}/{self.model_blob_path}"

    @property
    def metadata_url(self) -> str:
        return f"{self.blob_base_url}/{self.metadata_blob_path}"

    @property
    def predictions_url(self) -> str:
        return f"{self.blob_base_url}/{self.predictions_blob_path}"

    @property
    def stations_url(self) -> str:
        return f"{self.blob_base_url}/{self.stations_blob_path}"

    class Config:
        env_file = ".env"


settings = Settings()
