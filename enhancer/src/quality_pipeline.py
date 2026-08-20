from __future__ import annotations
import importlib.util
import os
import tempfile
import urllib.request
from pathlib import Path
from .r2_store import upload_file

FLASH_ROOT = Path(os.environ.get('FLASHVSR_ROOT', '/opt/FlashVSR'))
SCRIPT = FLASH_ROOT / 'examples/WanVSR/infer_flashvsr_v1.1_tiny_long_video.py'
_PIPE = None
_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        os.chdir(str(FLASH_ROOT / 'examples/WanVSR'))
        spec = importlib.util.spec_from_file_location('flashvsr_runtime', SCRIPT)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        _MODULE = module
    return _MODULE


def _pipe():
    global _PIPE
    if _PIPE is None: _PIPE = _module().init_pipeline()
    return _PIPE


def _download(url: str, path: Path):
    req = urllib.request.Request(url, headers={'User-Agent': 'SceneBuilder-Enhancer/1.0'})
    with urllib.request.urlopen(req, timeout=120) as response, open(path, 'wb') as out:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk: break
            out.write(chunk)


def run_video_upscale(job: dict, cancel_event, progress) -> dict:
    source = str((job.get('input') or {}).get('url') or '').strip()
    output_key = str((job.get('output') or {}).get('objectKey') or '').strip()
    if not source.startswith(('http://', 'https://')): raise ValueError('video_upscale input.url must be HTTP(S)')
    if not output_key: raise ValueError('output.objectKey is required')
    if cancel_event.is_set(): raise RuntimeError('cancelled')
    import torch
    mod = _module()
    with tempfile.TemporaryDirectory(prefix='sb-flashvsr-') as tmp:
        root = Path(tmp); input_path = root / 'input.mp4'; out_path = root / 'output.mp4'
        progress('download', 5); _download(source, input_path)
        if cancel_event.is_set(): raise RuntimeError('cancelled')
        progress('preparing_model', 15); pipe = _pipe()
        LQ, th, tw, F, fps = mod.prepare_input_tensor(str(input_path), scale=4.0, dtype=torch.bfloat16, device='cuda')
        if cancel_event.is_set(): raise RuntimeError('cancelled')
        progress('upscaling', 30)
        video = pipe(prompt='', negative_prompt='', cfg_scale=1.0, num_inference_steps=1, seed=0, LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True, topk_ratio=2.0*768*1280/(th*tw), kv_ratio=3.0, local_range=11, color_fix=True)
        if cancel_event.is_set(): raise RuntimeError('cancelled')
        progress('encoding', 82); frames = mod.tensor2video(video); mod.save_video(frames, str(out_path), fps=fps, quality=5)
        progress('upload', 94); stored = upload_file(out_path, output_key, 'video/mp4'); progress('done', 100)
        return {**stored, 'runtime': 'scenebuilder-enhancer-quality', 'modelFamily': 'flashvsr-v1.1', 'scale': 4}
