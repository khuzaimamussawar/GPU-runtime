from __future__ import annotations
import os
from pathlib import Path
import boto3


def _client():
    endpoint = os.environ.get('R2_ENDPOINT', '').strip()
    access = os.environ.get('R2_ACCESS_KEY', '').strip()
    secret = os.environ.get('R2_SECRET_KEY', '').strip()
    if not endpoint or not access or not secret:
        raise RuntimeError('R2_ENDPOINT/R2_ACCESS_KEY/R2_SECRET_KEY are required')
    return boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=access, aws_secret_access_key=secret, region_name=os.environ.get('R2_REGION', 'auto'))


def upload_file(path: str | Path, object_key: str, content_type: str) -> dict:
    bucket = os.environ.get('R2_BUCKET_NAME', 'scene-builder-images')
    file_path = Path(path)
    size = file_path.stat().st_size
    _client().upload_file(str(file_path), bucket, object_key, ExtraArgs={'ContentType': content_type})
    base = os.environ.get('R2_PUBLIC_URL', '').rstrip('/')
    return {'objectKey': object_key, 'url': f'{base}/{object_key}' if base else object_key, 'sizeBytes': size, 'contentType': content_type}
