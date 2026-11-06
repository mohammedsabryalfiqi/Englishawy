import contextlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import zero_gpu


class FakeTensor:
    def __init__(self, array): self.array = np.asarray(array, dtype=np.float32)
    @property
    def shape(self): return self.array.shape
    @property
    def T(self): return FakeTensor(self.array.T)
    def __len__(self): return len(self.array)
    def __getitem__(self, key): return FakeTensor(self.array[key])
    def __matmul__(self, other): return FakeTensor(self.array @ other.array)
    def to(self, device): return self
    def cpu(self): return self
    def detach(self): return self
    def numpy(self): return self.array
    def squeeze(self, dim): return FakeTensor(np.squeeze(self.array, axis=dim))
    def tolist(self): return self.array.tolist()


@pytest.fixture
def fake_torch(monkeypatch):
    module = ModuleType("torch")
    module.from_numpy = FakeTensor
    module.tensor = FakeTensor
    module.inference_mode = contextlib.nullcontext
    module.nn = SimpleNamespace(functional=SimpleNamespace(
        normalize=lambda tensor, dim: FakeTensor(tensor.array / np.maximum(np.linalg.norm(tensor.array, axis=dim, keepdims=True), 1e-12))))
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_local_cpu_does_not_initialize_torch(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    monkeypatch.delenv("ENGLISHAWY_GPU_BACKEND", raising=False)
    assert not zero_gpu.enabled()
    assert not zero_gpu.ready()
    assert "CPU" in zero_gpu.status()


def test_gpu_boundary_returns_cpu_floats_with_mocked_model(monkeypatch, fake_torch):
    calls = []
    class Model:
        def __call__(self, tensor):
            calls.append(tuple(tensor.shape))
            value = tensor.numpy().reshape(tensor.shape[0], -1)[:, :3]
            return SimpleNamespace(last_hidden_state=fake_torch.tensor(value[:, None, :]))
    monkeypatch.setattr(zero_gpu, "_model", Model())
    reference = np.random.default_rng(1).normal(size=(1, 3, 224, 224)).astype(np.float32)
    candidates = np.random.default_rng(2).normal(size=(10, 3, 224, 224)).astype(np.float32)
    scores = zero_gpu.similarities(reference, candidates)
    assert len(scores) == 10
    assert all(isinstance(x, float) and -1.001 <= x <= 1.001 for x in scores)
    assert calls[0] == (1, 3, 224, 224)
    assert calls[1:] == [(8, 3, 224, 224), (2, 3, 224, 224)]
