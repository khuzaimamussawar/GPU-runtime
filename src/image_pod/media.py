from __future__ import annotations

import hashlib
import math
import os
import shutil
import time
import urllib.parse
from pathlib import Path
from typing import Any, Iterable


IMAGE_ROOT = Path(os.environ.get("SCENEBUILDER_IMAGE_ROOT", "/opt/scenebuilder-image"))
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"
LORA_CACHE = Path(os.environ.get("IMAGE_LORA_CACHE_DIR", str(IMAGE_ROOT / "cache" / "loras")))
KREA2_BAKED_LORA_DIR = Path(os.environ.get("KREA2_BAKED_LORA_DIR", "/opt/scenebuilder-models/krea2/loras"))
STYLE_REFERENCE_PREFIXES = ("projects/", "temp/", "style/", "styles/", "Style/", "images/")
LORA_PREFIXES = ("models/lora/",)


class ImageMediaError(RuntimeError):
    pass


def safe_name(value: Any) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return cleaned.strip("._") or "asset"


def r2_path_part(value: Any) -> str:
    """Match SceneBuilder's Worker-side R2 filename sanitization exactly."""
    text = str(value or "unknown")
    return "".join(ch if (ch.isascii() and ch.isalnum()) or ch in {"-", "_", "."} else "_" for ch in text)[:120]


def canonical_krea2_lora_key(lora_id: Any, file_name: Any) -> str:
    """Return the one accepted R2 location for a user-supplied Krea 2 LoRA."""
    return f"models/lora/krea2/{r2_path_part(lora_id)}/{r2_path_part(file_name)}"


def _r2_client():
    bucket = os.environ.get("R2_BUCKET_NAME")
    endpoint = os.environ.get("R2_ENDPOINT")
    access_key = os.environ.get("R2_ACCESS_KEY")
    secret_key = os.environ.get("R2_SECRET_KEY")
    if not all([bucket, endpoint, access_key, secret_key]):
        return None, None

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.environ.get("R2_REGION", "auto"),
    )
    return client, bucket


def _media_ref(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("http://") or text.startswith("https://"):
            return {"url": text}
        return {"objectKey": text.lstrip("/")}
    if isinstance(value, dict):
        return dict(value)
    raise ImageMediaError(f"unsupported media reference: {type(value).__name__}")


def _object_key_from_url(value: Any, allowed_prefixes: tuple[str, ...]) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    direct = text.lstrip("/")
    if direct.startswith(allowed_prefixes):
        return direct
    try:
        parsed = urllib.parse.urlparse(text)
    except Exception:
        return None
    key = urllib.parse.unquote(parsed.path or "").lstrip("/")
    return key if key.startswith(allowed_prefixes) else None


def _download_ref_to_path(
    value: Any,
    target: Path,
    *,
    allowed_prefixes: tuple[str, ...],
    max_bytes: int,
) -> Path:
    ref = _media_ref(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = str(ref.get("url") or "").strip() or None
    explicit_key = str(ref.get("objectKey") or ref.get("key") or "").strip().lstrip("/")
    object_key = explicit_key or _object_key_from_url(url, allowed_prefixes)
    if not object_key or not object_key.startswith(allowed_prefixes):
        raise ImageMediaError("media reference must resolve to a trusted SceneBuilder R2 object")

    client, bucket = _r2_client()
    if client is None or not bucket:
        raise ImageMediaError(
            "R2 download requires R2_BUCKET_NAME, R2_ENDPOINT, R2_ACCESS_KEY and R2_SECRET_KEY"
        )

    tmp = target.with_suffix(target.suffix + ".part")
    tmp.unlink(missing_ok=True)
    try:
        metadata = client.head_object(Bucket=bucket, Key=object_key)
        content_length = int(metadata.get("ContentLength") or 0)
        if content_length <= 0:
            raise ImageMediaError(f"R2 object is empty: {object_key}")
        if content_length > max_bytes:
            raise ImageMediaError(
                f"R2 object exceeds runtime limit: {content_length} > {max_bytes} bytes ({object_key})"
            )
        client.download_file(bucket, object_key, str(tmp))
        actual = tmp.stat().st_size
        if actual != content_length:
            raise ImageMediaError(
                f"R2 download size mismatch: expected {content_length}, got {actual} ({object_key})"
            )
        os.replace(tmp, target)
        return target
    except ImageMediaError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise ImageMediaError(f"failed trusted R2 download for {object_key}: {exc}") from exc


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _cache_limits() -> tuple[int, int]:
    max_bytes = int(float(os.environ.get("IMAGE_LORA_CACHE_MAX_GB", "4")) * 1024**3)
    min_free_bytes = int(float(os.environ.get("IMAGE_LORA_CACHE_MIN_FREE_GB", "3")) * 1024**3)
    return max(0, max_bytes), max(0, min_free_bytes)


def _evict_lora_cache(*, protect: set[Path] | None = None) -> None:
    protect = {path.resolve() for path in (protect or set())}
    LORA_CACHE.mkdir(parents=True, exist_ok=True)
    max_bytes, min_free_bytes = _cache_limits()

    files = [
        path
        for path in LORA_CACHE.glob("*.safetensors")
        if path.is_file() and path.resolve() not in protect
    ]
    files.sort(key=lambda path: path.stat().st_atime)

    def cache_bytes() -> int:
        return sum(path.stat().st_size for path in LORA_CACHE.glob("*.safetensors") if path.is_file())

    while files:
        current = cache_bytes()
        free = shutil.disk_usage(LORA_CACHE).free
        if (not max_bytes or current <= max_bytes) and free >= min_free_bytes:
            break
        victim = files.pop(0)
        victim.unlink(missing_ok=True)


def materialize_user_loras(values: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    raw_values = [dict(value) for value in (values or [])]
    if len(raw_values) > 3:
        raise ImageMediaError("at most 3 user LoRAs are supported")

    protected: set[Path] = set()
    touched_lora_cache = False

    for item in raw_values:
        raw_lora_id = str(item.get("loraId") or item.get("lora_id") or "").strip()
        if not raw_lora_id:
            raise ImageMediaError("LoRA loraId is required")
        lora_id = safe_name(raw_lora_id)
        storage_source = str(item.get("storageSource") or item.get("storage_source") or "r2").strip().lower()
        if storage_source not in {"r2", "baked"}:
            raise ImageMediaError(f"LoRA {raw_lora_id} has invalid storageSource")

        expected_size_raw = item.get("fileSizeBytes", item.get("file_size_bytes"))
        if expected_size_raw in {None, ""}:
            raise ImageMediaError(f"LoRA {raw_lora_id} requires trusted fileSizeBytes")
        try:
            expected_size = int(expected_size_raw)
        except (TypeError, ValueError) as exc:
            raise ImageMediaError(f"LoRA {raw_lora_id} has invalid fileSizeBytes") from exc
        if expected_size <= 0:
            raise ImageMediaError(f"LoRA {raw_lora_id} has invalid fileSizeBytes")

        expected_sha = str(item.get("sha256") or "").strip().lower()
        if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
            raise ImageMediaError(f"LoRA {raw_lora_id} requires a valid trusted SHA256")

        min_strength = item.get("minStrength", item.get("min_strength"))
        max_strength = item.get("maxStrength", item.get("max_strength"))
        if min_strength is None or max_strength is None:
            raise ImageMediaError(f"LoRA {raw_lora_id} requires trusted min/max strength")

        if storage_source == "baked":
            file_name = safe_name(str(item.get("fileName") or item.get("file_name") or ""))
            if not file_name.endswith(".safetensors"):
                raise ImageMediaError(f"LoRA {raw_lora_id} has invalid baked fileName")
            baked_root = KREA2_BAKED_LORA_DIR.resolve()
            target = (baked_root / file_name).resolve()
            if baked_root not in target.parents:
                raise ImageMediaError(f"LoRA {raw_lora_id} baked path escapes model root")
            trusted_baked_path = str(item.get("bakedPath") or item.get("baked_path") or "").strip()
            if trusted_baked_path and Path(trusted_baked_path).resolve() != target:
                raise ImageMediaError(f"LoRA {raw_lora_id} baked path does not match fileName")
            if not target.is_file():
                raise ImageMediaError(f"LoRA {raw_lora_id} baked file is missing")
            if target.stat().st_size != expected_size:
                raise ImageMediaError(f"LoRA size mismatch for {raw_lora_id}")
            if _hash_file(target) != expected_sha:
                raise ImageMediaError(f"LoRA SHA256 mismatch for {raw_lora_id}")
        else:
            LORA_CACHE.mkdir(parents=True, exist_ok=True)
            touched_lora_cache = True
            object_key = str(item.get("objectKey") or item.get("r2ObjectKey") or "").strip().lstrip("/")
            file_name = r2_path_part(str(item.get("fileName") or item.get("file_name") or object_key.split("/")[-1]))
            expected_key = canonical_krea2_lora_key(raw_lora_id, file_name)
            if not file_name.endswith(".safetensors") or object_key != expected_key:
                raise ImageMediaError(
                    f"LoRA {raw_lora_id} must use its canonical Krea 2 R2 object key"
                )

            target = LORA_CACHE / f"{lora_id}.safetensors"
            valid = target.exists()
            if valid and target.stat().st_size != expected_size:
                valid = False
            if valid and _hash_file(target) != expected_sha:
                valid = False
            if not valid:
                target.unlink(missing_ok=True)
                _download_ref_to_path(
                    {"objectKey": object_key},
                    target,
                    allowed_prefixes=LORA_PREFIXES,
                    max_bytes=expected_size,
                )
                if target.stat().st_size != expected_size:
                    target.unlink(missing_ok=True)
                    raise ImageMediaError(f"LoRA size mismatch for {raw_lora_id}")
                if _hash_file(target) != expected_sha:
                    target.unlink(missing_ok=True)
                    raise ImageMediaError(f"LoRA SHA256 mismatch for {raw_lora_id}")

            now = time.time()
            os.utime(target, (now, target.stat().st_mtime))
            protected.add(target)
        resolved.append(
            {
                "loraId": raw_lora_id,
                "fileName": target.name,
                "strength": item.get("strength"),
                "minStrength": min_strength,
                "maxStrength": max_strength,
            }
        )

    if touched_lora_cache:
        _evict_lora_cache(protect=protected)
    return resolved


def _image_suffix(value: Any) -> str:
    ref = _media_ref(value)
    source = str(ref.get("objectKey") or ref.get("key") or ref.get("url") or "")
    suffix = Path(urllib.parse.urlparse(source).path).suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else ".img"


def stage_style_references(job_id: str, values: Iterable[Any] | None) -> tuple[list[str], Path]:
    refs = list(values or [])
    if not 1 <= len(refs) <= 10:
        raise ImageMediaError("style-reference mode requires 1-10 images")

    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise ImageMediaError("style reference preprocessing requires Pillow") from exc

    job_dir = COMFY_INPUT / "scenebuilder" / safe_name(job_id) / "style_refs"
    job_dir.mkdir(parents=True, exist_ok=True)
    max_dim = max(256, int(os.environ.get("KREA2_REFERENCE_MAX_DIM", "4096")))
    max_bytes = max(1024 * 1024, int(os.environ.get("KREA2_REFERENCE_MAX_BYTES", str(25 * 1024 * 1024))))
    filenames: list[str] = []

    for index, ref in enumerate(refs, start=1):
        source = job_dir / f"source_{index:02d}{_image_suffix(ref)}"
        _download_ref_to_path(
            ref,
            source,
            allowed_prefixes=STYLE_REFERENCE_PREFIXES,
            max_bytes=max_bytes,
        )
        output = job_dir / f"reference_{index:02d}.png"
        try:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                width, height = image.size
                longest = max(width, height)
                if longest > max_dim:
                    scale = max_dim / longest
                    target = (max(1, round(width * scale)), max(1, round(height * scale)))
                    image = image.resize(target, Image.Resampling.LANCZOS)
                image.save(output, "PNG", optimize=True)
        except Exception as exc:
            raise ImageMediaError(f"unable to preprocess style reference {index}: {exc}") from exc
        finally:
            source.unlink(missing_ok=True)

        filenames.append(str(output.relative_to(COMFY_INPUT)).replace("\\", "/"))

    return filenames, job_dir


def _resolve_comfy_output(image: dict[str, Any]) -> Path:
    filename = safe_name(image.get("filename"))
    subfolder = str(image.get("subfolder") or "").strip().replace("\\", "/").strip("/")
    candidate = (COMFY_OUTPUT / subfolder / filename).resolve()
    root = COMFY_OUTPUT.resolve()
    if candidate != root and root not in candidate.parents:
        raise ImageMediaError("Comfy output escaped output directory")
    if not candidate.is_file():
        raise ImageMediaError(f"Comfy output file not found: {candidate}")
    return candidate


def finalize_image_outputs(
    *,
    job_id: str,
    project_id: str | None,
    outputs: Iterable[dict[str, Any]],
    settings: dict[str, Any],
    output_prefix: str | None = None,
) -> dict[str, Any]:
    images = list(outputs)
    if not images:
        raise ImageMediaError("no Comfy image output to finalize")

    try:
        from PIL import Image
    except Exception as exc:
        raise ImageMediaError("image finalization requires Pillow") from exc

    source = _resolve_comfy_output(images[0])
    final_dir = IMAGE_ROOT / "tmp" / safe_name(job_id)
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "image.png"

    output_w_raw = settings.get("outputWidth")
    output_h_raw = settings.get("outputHeight")
    output_w = int(output_w_raw) if output_w_raw not in {None, ""} else None
    output_h = int(output_h_raw) if output_h_raw not in {None, ""} else None

    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if output_w and output_h and image.size != (output_w, output_h):
            source_ratio = image.width / image.height
            target_ratio = output_w / output_h
            if not math.isclose(source_ratio, target_ratio, rel_tol=0, abs_tol=1e-5):
                raise ImageMediaError(
                    f"refusing non-proportional final resize {image.size} -> {(output_w, output_h)}"
                )
            image = image.resize((output_w, output_h), Image.Resampling.LANCZOS)
        image.save(final_path, "PNG", optimize=True)

    thumbnail_status = "completed"
    thumbnail: dict[str, Any] | None = None
    thumbnail_error: str | None = None
    thumb_path = final_dir / "thumbnail.jpg"
    try:
        with Image.open(final_path) as opened:
            thumb = opened.convert("RGB")
            if thumb.width > 700:
                height = max(1, round(thumb.height * (700 / thumb.width)))
                thumb = thumb.resize((700, height), Image.Resampling.LANCZOS)
            thumb.save(thumb_path, "JPEG", quality=80, optimize=True)
        thumbnail = {
            "fileName": thumb_path.name,
            "contentType": "image/jpeg",
            "sizeBytes": thumb_path.stat().st_size,
        }
    except Exception as exc:
        thumbnail_status = "failed"
        thumbnail_error = str(exc)

    try:
        source.unlink(missing_ok=True)
    except Exception:
        pass

    return {
        # SceneBuilder, not the GPU pod, stores generated output in R2. The
        # pod keeps this artifact only until the Worker fetches it securely.
        "full": {
            "fileName": final_path.name,
            "contentType": "image/png",
            "sizeBytes": final_path.stat().st_size,
        },
        "thumbnail": thumbnail,
        "thumbnailStatus": thumbnail_status,
        "thumbnailError": thumbnail_error,
        "width": output_w,
        "height": output_h,
    }


def finalized_output_path(job_id: str, file_name: str) -> Path | None:
    if file_name not in {"image.png", "thumbnail.jpg"}:
        return None
    candidate = (IMAGE_ROOT / "tmp" / safe_name(job_id) / file_name).resolve()
    root = (IMAGE_ROOT / "tmp").resolve()
    if root not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def cleanup_finalized_outputs(job_id: str) -> None:
    try:
        path = (IMAGE_ROOT / "tmp" / safe_name(job_id)).resolve()
        root = (IMAGE_ROOT / "tmp").resolve()
        if root in path.parents:
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def cleanup_job_inputs(path: Path | None) -> None:
    if path is None:
        return
    try:
        resolved = path.resolve()
        root = COMFY_INPUT.resolve()
        if resolved != root and root in resolved.parents:
            shutil.rmtree(resolved, ignore_errors=True)
    except Exception:
        pass
