import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # PV_ENV_FILE="" disables .env loading (tests must not pick up a developer's local settings).
    model_config = SettingsConfigDict(
        env_prefix="PV_", env_file=os.environ.get("PV_ENV_FILE", ".env") or None, extra="ignore"
    )

    # PV_DATABASE_URL wins; otherwise the DATABASE_URL / POSTGRES_URL injected by the Neon integration.
    database_url: str = Field(
        default=f"sqlite:///{_BACKEND_DIR.as_posix()}/patentvision.db",
        validation_alias=AliasChoices("PV_DATABASE_URL", "DATABASE_URL", "POSTGRES_URL"),
    )
    auto_create_tables: bool = True

    # local: files under storage_root. blob: Vercel Blob (private store). auto: blob when Blob credentials exist.
    storage_backend: str = "auto"
    storage_root: Path = _BACKEND_DIR / "storage"

    # Public URL prefix the API is mounted under (Vercel routes /api/* to this service). Stripped if present.
    api_prefix: str = "/api"

    secret_key: str = "dev-insecure-secret-change-me"
    access_token_minutes: int = 60 * 24

    # Vercel Functions accept request bodies up to 4.5 MB; the web app shrinks larger photos before upload.
    max_upload_bytes: int = 4 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    processing_max_side: int = 1600

    background_remover: str = "auto"  # auto | rembg | classical
    segmenter: str = "classical"  # classical | sam
    sam_checkpoint: str | None = None
    sam_model_type: str = "vit_b"

    # False (default, per PRD FR-09) = strict outer outline. True keeps large enclosed openings (e.g. a mug
    # handle) as inner contours; a light label on a light background can then be mistaken for an opening.
    keep_openings: bool = False
    drawing_long_side: int = 1400
    drawing_line_width: float = 4.0
    drawing_detail_level: str = "standard"  # standard (restored previous best default) | clean | detailed
    edge_low_threshold: float | None = None
    edge_high_threshold: float | None = None
    smoothing_strength: float | None = None
    min_contour_length: float | None = None
    texture_suppression: bool = False

    rate_limit_per_minute: int = 60
    max_concurrent_jobs: int = 2
    # A job still Queued/Processing after this long lost its worker (crash, timeout) and is shown as Failed.
    stale_job_minutes: int = 10

    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cors_origin_regex: str | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> list[str]:
        """Accept either a JSON list or a comma-separated string of origins."""
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(v).strip().rstrip("/") for v in parsed if str(v).strip()]
            return [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]
        if isinstance(value, (list, tuple)):
            return [str(v).strip().rstrip("/") for v in value if str(v).strip()]
        raise ValueError("PV_CORS_ORIGINS must be a JSON list or comma-separated URLs")

    @property
    def on_vercel(self) -> bool:
        return bool(os.environ.get("VERCEL"))

    @property
    def use_blob(self) -> bool:
        if self.storage_backend == "blob":
            return True
        if self.storage_backend == "local":
            return False
        return bool(
            os.environ.get("BLOB_READ_WRITE_TOKEN")
            or os.environ.get("VERCEL_BLOB_READ_WRITE_TOKEN")
            or os.environ.get("BLOB_STORE_ID")
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
