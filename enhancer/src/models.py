from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REAL_ESRGAN_ROOT = Path(os.environ.get("REAL_ESRGAN_ROOT", "/opt/Real-ESRGAN"))
RIFE_ROOT = Path(os.environ.get("RIFE_ROOT", "/opt/Practical-RIFE"))
RIFE_MODEL_DIR = Path(os.environ.get("RIFE_MODEL_DIR", "/opt/scenebuilder-models/rife-4.9"))

_ESRGAN: dict[str, Any] = {}
_RIFE: Any | None = None
_RIFE_PADDING_LOGGED: set[tuple[int, int, int, int]] = set()


def require_cuda_tensor(value: Any, name: str = "tensor") -> None:
    import torch
    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        raise RuntimeError(f"GPU_ONLY_INVARIANT:{name}_NOT_CUDA")


def _esrgan_architecture(model_name: str):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    if model_name == "RealESRGAN_x4plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    if model_name == "RealESRGAN_x4plus_anime_6B":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    if model_name == "realesr-animevideov3":
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
    if model_name == "realesr-general-x4v3":
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
    raise ValueError(f"Unsupported Real-ESRGAN model: {model_name}")


def esrgan(model_name: str):
    """Persistent full-frame FP16 Real-ESRGAN adapter. Tiling is forbidden."""
    import torch
    from realesrgan import RealESRGANer

    if model_name in _ESRGAN:
        return _ESRGAN[model_name]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    weights = REAL_ESRGAN_ROOT / "weights" / f"{model_name}.pth"
    if not weights.is_file():
        raise RuntimeError(f"MODEL_LOAD_FAILED:{weights}")
    model_path: str | list[str] = str(weights)
    dni_weight = None
    if model_name == "realesr-general-x4v3":
        wdn = REAL_ESRGAN_ROOT / "weights" / "realesr-general-wdn-x4v3.pth"
        if wdn.is_file():
            # Fixed balanced general-video denoise policy for V1.
            model_path = [str(weights), str(wdn)]
            dni_weight = [0.5, 0.5]
    runner = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=dni_weight,
        model=_esrgan_architecture(model_name),
        tile=0,
        tile_pad=0,
        pre_pad=0,
        half=True,
        device=torch.device("cuda:0"),
    )
    if not next(runner.model.parameters()).is_cuda:
        raise RuntimeError("GPU_ONLY_INVARIANT:ESRGAN_MODEL_NOT_CUDA")
    _ESRGAN[model_name] = runner
    return runner


def upscale_bgr(frame: np.ndarray, model_name: str, outscale: float = 4.0) -> np.ndarray:
    output, _ = esrgan(model_name).enhance(frame, outscale=float(outscale))
    return output


def _install_rife_train_log_alias(module_path: Path) -> None:
    """Make Hugging Face's flat RIFE 4.9 snapshot importable without mutating it.

    The pinned Practical-RIFE checkout provides the repository-level ``model``
    package, while the downloaded RIFE 4.9 snapshot stores ``RIFE_HDv3.py`` and
    ``IFNet_HDv3.py`` side-by-side. Its runtime wrapper still imports
    ``train_log.IFNet_HDv3``. Expose the flat snapshot directory as that package
    only when loading the snapshot copy, so the native fallback uses the exact
    files that accompany the downloaded checkpoint.
    """
    if module_path.parent != RIFE_MODEL_DIR:
        return
    ifnet_path = RIFE_MODEL_DIR / "IFNet_HDv3.py"
    if not ifnet_path.is_file():
        raise RuntimeError(f"RIFE_RUNTIME_FAILED:{ifnet_path} not found")
    package = sys.modules.get("train_log")
    if package is None:
        package = types.ModuleType("train_log")
        package.__package__ = "train_log"
        package.__path__ = [str(RIFE_MODEL_DIR)]
        sys.modules["train_log"] = package


def _load_rife_module():
    candidates = [
        RIFE_MODEL_DIR / "RIFE_HDv3.py",
        RIFE_MODEL_DIR / "train_log" / "RIFE_HDv3.py",
        RIFE_ROOT / "train_log" / "RIFE_HDv3.py",
    ]
    module_path = next((path for path in candidates if path.is_file()), None)
    if module_path is None:
        raise RuntimeError("RIFE_RUNTIME_FAILED:RIFE_HDv3.py not found")
    spec = importlib.util.spec_from_file_location("scenebuilder_rife_hd", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("RIFE_RUNTIME_FAILED:module spec")

    # Practical-RIFE's train_log/RIFE_HDv3.py imports siblings through the
    # repository-level `model` package. Adding only train_log makes
    # `from model...` fail at runtime with ModuleNotFoundError. Keep both the
    # configured Practical-RIFE root and the module directory importable.
    import_roots = [RIFE_ROOT, module_path.parent]
    if module_path.parent.name == "train_log":
        import_roots.append(module_path.parent.parent)
    for root in import_roots:
        value = str(root)
        if value and value not in sys.path:
            sys.path.insert(0, value)

    _install_rife_train_log_alias(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rife():
    global _RIFE
    import torch
    if _RIFE is not None:
        return _RIFE
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    print(f"[enhancer rife] native_load_start model_dir={RIFE_MODEL_DIR}", flush=True)
    module = _load_rife_module()
    model = module.Model()
    model.load_model(str(RIFE_MODEL_DIR), -1)
    model.eval()
    model.device()
    version = float(getattr(model, "version", 0.0) or 0.0)
    if version < 3.9:
        raise RuntimeError(f"RIFE_RUNTIME_FAILED:checkpoint lacks arbitrary timestep support version={version}")
    _RIFE = model
    print(f"[enhancer rife] native_load_ready version={version}", flush=True)
    return model


def _rgb_to_rife_tensor(frame_rgb: np.ndarray):
    import torch
    tensor = torch.from_numpy(np.ascontiguousarray(frame_rgb)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device="cuda:0", dtype=torch.float16).div_(255.0)
    require_cuda_tensor(tensor, "rife_input")
    return tensor


def _rife_padded_shape(width: int, height: int, scale: float) -> tuple[int, int]:
    # Practical-RIFE pads to tmp=max(32, int(32/scale)) before IFNet. 2160 is
    # not divisible by 32, which otherwise produces internal 2176-vs-2160
    # concatenation failures in the native fallback at 4K.
    safe_scale = max(float(scale), 1e-6)
    block = max(32, int(32 / safe_scale))
    padded_w = ((int(width) - 1) // block + 1) * block
    padded_h = ((int(height) - 1) // block + 1) * block
    return padded_w, padded_h


def interpolate_rife(frame0_rgb: np.ndarray, frame1_rgb: np.ndarray, timestep: float, scale: float = 1.0) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    if not 0.0 < float(timestep) < 1.0:
        raise ValueError("RIFE timestep must be between 0 and 1")
    if frame0_rgb.shape != frame1_rgb.shape:
        raise RuntimeError(f"RIFE_RUNTIME_FAILED:frame shape mismatch {frame0_rgb.shape} != {frame1_rgb.shape}")
    i0 = _rgb_to_rife_tensor(frame0_rgb)
    i1 = _rgb_to_rife_tensor(frame1_rgb)
    height, width = int(i0.shape[-2]), int(i0.shape[-1])
    padded_w, padded_h = _rife_padded_shape(width, height, float(scale))
    if padded_w != width or padded_h != height:
        padding = (0, padded_w - width, 0, padded_h - height)
        i0 = F.pad(i0, padding)
        i1 = F.pad(i1, padding)
        key = (width, height, padded_w, padded_h)
        if key not in _RIFE_PADDING_LOGGED:
            _RIFE_PADDING_LOGGED.add(key)
            print(f"[enhancer rife] native_pad source={width}x{height} padded={padded_w}x{padded_h}", flush=True)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        result = rife().inference(i0, i1, float(timestep), float(scale))
    require_cuda_tensor(result, "rife_output")
    result = result[..., :height, :width]
    result = result[0].clamp(0, 1).float().permute(1, 2, 0).cpu().numpy()
    return np.rint(result * 255.0).astype(np.uint8)


def qualify_rife_timesteps() -> dict[str, float]:
    """Boot/release smoke for non-.5 timesteps required by SceneBuilder."""
    import torch
    model = rife()
    left = torch.zeros((1, 3, 64, 64), device="cuda:0", dtype=torch.float16)
    right = torch.ones((1, 3, 64, 64), device="cuda:0", dtype=torch.float16)
    outputs: dict[str, float] = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for timestep in (0.2, 1 / 3, 0.4, 0.583, 0.729, 0.8):
            out = model.inference(left, right, float(timestep), 1.0)
            require_cuda_tensor(out, f"rife_{timestep}")
            if not torch.isfinite(out).all():
                raise RuntimeError(f"RIFE_RUNTIME_FAILED:nonfinite timestep={timestep}")
            outputs[f"{timestep:.6f}"] = float(out.float().mean().item())
    return outputs
