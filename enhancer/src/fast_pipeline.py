from __future__ import annotations
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from PIL import Image
from .r2_store import upload_file

REAL_ESRGAN_ROOT = Path(os.environ.get('REAL_ESRGAN_ROOT', '/opt/Real-ESRGAN'))


def _download(url: str, path: Path):
    request = urllib.request.Request(url, headers={'User-Agent': 'SceneBuilder-Enhancer/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response, open(path, 'wb') as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _run(cmd: list[str], cancel_event, progress):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        while proc.poll() is None:
            if cancel_event.is_set():
                proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill()
                raise RuntimeError('cancelled')
            line = proc.stdout.readline() if proc.stdout else ''
            if line:
                print(f'[FAST] {line.rstrip()}', flush=True)
            progress('upscaling', 45)
            time.sleep(0.05)
        if proc.returncode != 0:
            rest = proc.stdout.read() if proc.stdout else ''
            raise RuntimeError(f'Real-ESRGAN exited {proc.returncode}: {rest[-3000:]}')
    finally:
        if proc.stdout: proc.stdout.close()


def _target_long_side(target_resolution: str) -> int:
    return 3840 if str(target_resolution).upper() == '4K' else 2560


def run_image_upscale(job: dict, cancel_event, progress) -> dict:
    source = str((job.get('input') or {}).get('url') or '').strip()
    output_key = str((job.get('output') or {}).get('objectKey') or '').strip()
    settings = job.get('settings') or {}
    if not source.startswith(('http://', 'https://')):
        raise ValueError('image_upscale input.url must be HTTP(S)')
    if not output_key:
        raise ValueError('output.objectKey is required')
    model = str(job.get('modelFamily') or 'realesr-general-x4v3')
    if model not in {'realesr-general-x4v3', 'realesr-animevideov3', 'RealESRGAN_x4plus'}:
        raise ValueError(f'unsupported FAST image model: {model}')

    with tempfile.TemporaryDirectory(prefix='sb-enhancer-') as tmp:
        root = Path(tmp); input_path = root / 'input.png'; output_dir = root / 'out'; output_dir.mkdir()
        progress('download', 5); _download(source, input_path)
        if cancel_event.is_set(): raise RuntimeError('cancelled')
        scale = float(settings.get('scale') or 4)
        scale = min(4.0, max(1.0, scale))
        cmd = ['python3', str(REAL_ESRGAN_ROOT / 'inference_realesrgan.py'), '-i', str(input_path), '-o', str(output_dir), '-n', model, '-s', str(scale), '--suffix', 'enhanced', '--ext', 'png', '--gpu-id', '0']
        if bool(settings.get('faceEnhance')): cmd.append('--face_enhance')
        tile = int(settings.get('tile') or os.environ.get('REAL_ESRGAN_TILE', '0'))
        if tile > 0: cmd.extend(['--tile', str(tile)])
        progress('upscaling', 15); _run(cmd, cancel_event, progress)
        candidates = sorted(output_dir.glob('input_enhanced.*'))
        if not candidates: raise RuntimeError('Real-ESRGAN produced no output')
        enhanced = candidates[0]
        progress('resize', 78)
        with Image.open(enhanced) as img:
            img = img.convert('RGB')
            target = _target_long_side(settings.get('targetResolution', '2K'))
            long_side = max(img.size)
            if long_side > target:
                ratio = target / long_side
                img = img.resize((max(1, round(img.width * ratio)), max(1, round(img.height * ratio))), Image.Resampling.LANCZOS)
            final_path = root / 'final.png'; img.save(final_path, format='PNG', optimize=False)
        if cancel_event.is_set(): raise RuntimeError('cancelled')
        progress('upload', 90); stored = upload_file(final_path, output_key, 'image/png')
        progress('done', 100)
        return {**stored, 'runtime': 'scenebuilder-enhancer-fast', 'modelFamily': model, 'targetResolution': settings.get('targetResolution', '2K')}
