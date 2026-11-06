"""Narrow ZeroGPU boundary: CPU arrays in, CPU cosine scores out.

Official ZeroGPU guidance: import spaces before torch; initialize/move the model
at module scope, and decorate only the actual CUDA inference. Outside ZeroGPU this
module does not import torch at all, so local previews use the ONNX CPU backend.
"""
from __future__ import annotations

import os

import numpy as np
import spaces

MODEL_ID = "facebook/dinov2-small"
_model = None
_model_error: str | None = None


def enabled() -> bool:
    return bool(os.getenv("SPACES_ZERO_GPU")) or os.getenv("ENGLISHAWY_GPU_BACKEND") == "torch"


def ready() -> bool:
    return enabled() and _model is not None


def status() -> str:
    if ready():
        return "ZeroGPU · DINOv2 Small"
    if enabled():
        return f"GPU initialization failed: {_model_error or 'model unavailable'}"
    return "CPU · DINOv2 ONNX"


def _load_model():
    # Do not override HF_HOME: preload_from_hub populates the default HF cache.
    import torch
    from transformers import AutoModel

    torch.set_num_threads(int(os.getenv("TORCH_CPU_THREADS", "2")))
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
    return model.to("cuda").eval()


if enabled():
    try:
        _model = _load_model()
    except Exception as exc:
        # Surface the failure in the UI; explicit Auto requests must fail clearly.
        _model_error = f"{type(exc).__name__}: {exc}"[:500]


@spaces.GPU(duration=30)
def similarities(reference: np.ndarray, candidates: np.ndarray) -> list[float]:
    """reference: [1,3,224,224], candidates: [N,3,224,224], both normalized on CPU."""
    if _model is None:
        raise RuntimeError(f"ZeroGPU model is unavailable: {_model_error or 'not initialized'}")
    import torch

    reference = np.asarray(reference, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    if reference.shape != (1, 3, 224, 224) or candidates.ndim != 4 or candidates.shape[1:] != (3, 224, 224):
        raise ValueError("Invalid DINOv2 input shapes")
    if candidates.shape[0] > 64:
        raise ValueError("At most 64 GPU candidates per request")
    with torch.inference_mode():
        ref_tensor = torch.from_numpy(np.ascontiguousarray(reference)).to("cuda")
        ref = _model(ref_tensor).last_hidden_state[:, 0]
        ref = torch.nn.functional.normalize(ref, dim=-1)
        scores = []
        for start in range(0, len(candidates), 8):
            batch = torch.from_numpy(np.ascontiguousarray(candidates[start:start + 8])).to("cuda")
            features = torch.nn.functional.normalize(_model(batch).last_hidden_state[:, 0], dim=-1)
            scores.extend((features @ ref.T).squeeze(-1).detach().cpu().tolist())
    return [float(x) for x in scores]
