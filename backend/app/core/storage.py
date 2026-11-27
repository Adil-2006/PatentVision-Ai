"""File storage addressed by relative keys (never absolute paths or public URLs).

LocalStorage keeps files under a storage root (development, tests, single-server deployments).
BlobStorage keeps them in a private Vercel Blob store (serverless deployments); every read goes
through the API so ownership checks always apply.
SupabaseStorage keeps files in a Supabase Storage bucket (Render + Supabase deployments).
"""
import shutil
from pathlib import Path, PurePosixPath
from typing import Protocol

from app.core.config import get_settings


def check_key(key: str) -> str:
    parts = PurePosixPath(key.replace("\\", "/")).parts
    if not parts or key.startswith(("/", "\\")) or ".." in parts or ":" in key:
        raise ValueError("storage key escapes the storage root")
    return "/".join(parts)


class StorageBackend(Protocol):
    def save_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...
    def read_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def delete_tree(self, prefix: str) -> None: ...


class LocalStorage:
    def __init__(self, root: Path | None = None) -> None:
        self._root_override = root

    @property
    def root(self) -> Path:
        return Path(self._root_override or get_settings().storage_root).resolve()

    def path(self, key: str) -> Path:
        target = (self.root / check_key(key)).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("storage key escapes the storage root")
        return target

    def save_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        return key

    def read_bytes(self, key: str) -> bytes:
        p = self.path(key)
        if not p.is_file():
            clean = check_key(key)
            fallbacks = [
                self.root.parent / "backend" / "storage" / clean,
                self.root.parent / "storage" / clean,
                Path("backend/storage") / clean,
                Path("storage") / clean,
            ]
            for fb in fallbacks:
                if fb.is_file():
                    return fb.read_bytes()
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        p = self.path(key)
        if p.is_file() and p.stat().st_size > 0:
            return True
        clean = check_key(key)
        fallbacks = [
            self.root.parent / "backend" / "storage" / clean,
            self.root.parent / "storage" / clean,
            Path("backend/storage") / clean,
            Path("storage") / clean,
        ]
        for fb in fallbacks:
            if fb.is_file() and fb.stat().st_size > 0:
                return True
        return False

    def delete(self, key: str) -> None:
        p = self.path(key)
        if p.is_file():
            p.unlink()

    def delete_tree(self, prefix: str) -> None:
        p = self.path(prefix)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


class SupabaseStorage:
    """Supabase Storage backend for Render deployments.

    Files are stored in a private bucket (``patentvision-files`` by default).
    Credentials come from ``PV_SUPABASE_URL`` and ``PV_SUPABASE_SERVICE_KEY``.
    The service-role key is required so the backend can read and write private objects.
    """

    def __init__(self) -> None:
        self._client = None
        self._client_creds: tuple[str, str] | None = None

    def _get_client(self):
        settings = get_settings()
        if not settings.supabase_url or not settings.supabase_service_key:
            raise RuntimeError(
                "PV_SUPABASE_URL and PV_SUPABASE_SERVICE_KEY must be set to use Supabase storage"
            )
        creds = (settings.supabase_url.rstrip("/"), settings.supabase_service_key)
        if self._client is not None and self._client_creds == creds:
            return self._client
        from supabase import create_client  # type: ignore[import]

        self._client = create_client(creds[0], creds[1])
        self._client_creds = creds
        return self._client

    def _bucket(self):
        return self._get_client().storage.from_(get_settings().supabase_bucket)

    def save_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        clean = check_key(key)
        try:
            # upsert=True overwrites existing objects
            self._bucket().upload(clean, data, {"content-type": content_type, "upsert": "true"})
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg or "duplicate" in msg or "23505" in msg:
                self._bucket().update(clean, data, {"content-type": content_type})
            else:
                raise
        return key

    def read_bytes(self, key: str) -> bytes:
        clean = check_key(key)
        try:
            result = self._bucket().download(clean)
            return bytes(result)
        except Exception as exc:
            status = getattr(exc, "status", None)
            msg = str(exc).lower()
            if str(status) == "404" or "not found" in msg or "not_found" in msg or "404" in msg:
                raise FileNotFoundError(key) from exc
            raise

    def exists(self, key: str) -> bool:
        clean = check_key(key)
        bucket = self._bucket()
        # 1. Try info() first if available (direct object check)
        if hasattr(bucket, "info"):
            try:
                bucket.info(clean)
                return True
            except Exception as exc:
                msg = str(exc).lower()
                status = getattr(exc, "status", None)
                if str(status) == "404" or "not found" in msg or "not_found" in msg or "404" in msg:
                    return False
        # 2. Try exists()
        if hasattr(bucket, "exists"):
            try:
                return bool(bucket.exists(clean))
            except Exception:
                pass
        # 3. Fallback: list the folder
        try:
            parent = "/".join(clean.split("/")[:-1])
            name = clean.split("/")[-1]
            items = bucket.list(parent, {"search": name, "limit": 100})
            return any(item.get("name") == name for item in (items or []))
        except Exception as exc:
            log.warning("SupabaseStorage.exists error for key %s: %s", key, exc)
            return False

    def delete(self, key: str) -> None:
        clean = check_key(key)
        try:
            self._bucket().remove([clean])
        except Exception:
            pass  # best-effort

    def _collect_files(self, bucket, folder: str) -> list[str]:
        paths: list[str] = []
        offset = 0
        while True:
            items = bucket.list(folder, {"limit": 100, "offset": offset}) or []
            for item in items:
                name = item.get("name")
                if not name:
                    continue
                full = f"{folder}/{name}" if folder else name
                # Supabase storage list() returns id=None for folder entries.
                if item.get("id") is None and item.get("metadata") is None:
                    paths.extend(self._collect_files(bucket, full))
                else:
                    paths.append(full)
            if len(items) < 100:
                break
            offset += 100
        return paths

    def delete_tree(self, prefix: str) -> None:
        clean = check_key(prefix).rstrip("/")
        try:
            bucket = self._bucket()
            paths = self._collect_files(bucket, clean)
            for i in range(0, len(paths), 100):
                bucket.remove(paths[i : i + 100])
        except Exception:
            pass  # best-effort


class BlobStorage:
    """Private Vercel Blob store. Credentials come from the environment (OIDC + BLOB_STORE_ID, or token)."""

    def save_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        from vercel.blob.sync import put_blob

        put_blob(check_key(key), data, access="private", content_type=content_type, allow_overwrite=True)
        return key

    def read_bytes(self, key: str) -> bytes:
        from vercel.blob.sync import get_blob

        try:
            return get_blob(check_key(key), access="private").body
        except Exception as exc:
            if type(exc).__name__ == "BlobNotFoundError":
                raise FileNotFoundError(key) from exc
            raise

    def exists(self, key: str) -> bool:
        from vercel.blob.sync import head_blob

        try:
            head_blob(check_key(key))
            return True
        except Exception as exc:
            if type(exc).__name__ == "BlobNotFoundError":
                return False
            raise

    def delete(self, key: str) -> None:
        from vercel.blob.sync import delete_blob

        delete_blob(check_key(key))

    def delete_tree(self, prefix: str) -> None:
        from vercel.blob.sync import delete_blob, query_blobs

        names = [b.pathname for b in query_blobs(prefix=check_key(prefix).rstrip("/") + "/")]
        for i in range(0, len(names), 100):
            delete_blob(names[i : i + 100])


class _Storage:
    """Picks the backend from settings on each call, so tests and config changes take effect."""

    _local = LocalStorage()
    _blob = BlobStorage()
    _supabase = SupabaseStorage()

    @property
    def backend(self) -> StorageBackend:
        settings = get_settings()
        if settings.use_supabase:
            return self._supabase
        if settings.use_blob:
            return self._blob
        return self._local

    @property
    def root(self) -> Path:  # local backend only (tests, dev tooling)
        return self._local.root

    def save_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        return self.backend.save_bytes(key, data, content_type)

    def read_bytes(self, key: str) -> bytes:
        return self.backend.read_bytes(key)

    def exists(self, key: str) -> bool:
        return self.backend.exists(key)

    def delete(self, key: str) -> None:
        self.backend.delete(key)

    def delete_tree(self, prefix: str) -> None:
        self.backend.delete_tree(prefix)

    def path(self, key: str) -> Path:  # local backend only
        return self._local.path(key)


storage = _Storage()
