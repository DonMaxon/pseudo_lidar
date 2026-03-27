"""
depth_estimator.py

Обёртка над Depth Anything 3 Metric (DA3METRIC-LARGE) для предсказания
метрической карты глубины по одному изображению.

Установка:
    git clone https://github.com/ByteDance-Seed/depth-anything-3
    cd depth-anything-3
    pip install -e .
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


_HF_MODEL_ID = "depth-anything/DA3METRIC-LARGE"


def _patch_swiglu_fused(model: "torch.nn.Module") -> None:
    """
    Заменяет xFormers SwiGLUFFNFused.forward на стандартный PyTorch forward.
    xFormers fused kernel напрямую использует raw weight тензоры и несовместим с INT8.
    После патча w12/w3 (nn.Linear) работают через стандартный F.linear.
    """
    import torch.nn.functional as _F
    from torch import Tensor as _Tensor

    def _std_swiglu_fwd(self, x: _Tensor) -> _Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(_F.silu(x1) * x2)

    for module in model.modules():
        if type(module).__name__ == "SwiGLUFFNFused":
            import types
            module.forward = types.MethodType(_std_swiglu_fwd, module)


def _quantize_linear_to_int8(model: "torch.nn.Module") -> "torch.nn.Module":
    """
    1. Заменяет xFormers SwiGLUFFNFused.forward на стандартный (совместимый с INT8).
    2. Рекурсивно заменяет nn.Linear на bitsandbytes Linear8bitLt (INT8, has_fp16_weights=False).
       Уменьшает память линейных слоёв в 4x (float32 → INT8).
    Требует: pip install bitsandbytes
    """
    try:
        from bitsandbytes.nn import Linear8bitLt
    except ImportError:
        raise ImportError("Установите bitsandbytes: pip install bitsandbytes")

    _patch_swiglu_fused(model)

    def _replace(m: "torch.nn.Module") -> None:
        for name, child in list(m.named_children()):
            if isinstance(child, torch.nn.Linear) and not isinstance(child, Linear8bitLt):
                q = Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
                # Copy data into Int8Params (NOT q.weight = child.weight)
                q.weight.data = child.weight.data
                if child.bias is not None:
                    q.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
                setattr(m, name, q)
            else:
                _replace(child)

    _replace(model)
    return model


class DepthEstimator:
    """
    Обёртка над DepthAnything3 для инференса метрической глубины.

    Args:
        model_id: HuggingFace model ID или путь к локальной директории модели.
        device:   'cuda', 'cpu' или 'auto'.
        max_depth: Клиппинг предсказанной глубины (метры).
    """

    def __init__(
        self,
        model_id: str = _HF_MODEL_ID,
        device: str = "auto",
        max_depth: float = 80.0,
    ):
        self.max_depth = max_depth

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Загрузка DA3METRIC-LARGE из '{model_id}' на {self.device} ...")
        self._load_model(model_id)
        print("  Модель загружена.")

    def _load_model(self, model_id: str) -> None:
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise ImportError(
                "Установите depth-anything-3:\n"
                "  git clone https://github.com/ByteDance-Seed/depth-anything-3\n"
                "  cd depth-anything-3 && pip install -e ."
            ) from e

        self._model = DepthAnything3.from_pretrained(model_id)
        if self.device.type == "cuda":
            free_vram, _ = torch.cuda.mem_get_info(self.device)
            model_bytes = sum(
                p.numel() * p.element_size() for p in self._model.parameters()
            )
            if model_bytes < int(free_vram * 0.90):
                # Model fits in VRAM — move directly
                self._model = self._model.to(self.device)
            else:
                # Model too large for VRAM — apply INT8 quantization via bitsandbytes.
                # Linear layer weights: float32 → INT8 (4x smaller), rest stays float32.
                print(
                    f"  Модель ({model_bytes / 1024**3:.2f} GB) не влезает в VRAM "
                    f"({free_vram / 1024**3:.2f} GB свободно). "
                    f"Применяется INT8-квантизация (bitsandbytes)..."
                )
                self._model = _quantize_linear_to_int8(self._model)
                self._model = self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray,
                intrinsic: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Предсказать метрическую карту глубины.

        Args:
            image:     (H, W, 3) uint8 RGB.
            intrinsic: (3, 3) float64 — nuScenes intrinsics в оригинальном разрешении.
                       Если передан, модель использует его для более точного depth.

        Returns:
            depth: (H, W) float32 в метрах, обрезано до max_depth.
        """
        depth, _ = self._run(image, intrinsic)
        return depth

    def predict_with_intrinsics(
        self, image: np.ndarray,
        intrinsic: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Предсказать карту глубины, опционально передав известные intrinsics в модель.

        Args:
            image:     (H, W, 3) uint8 RGB.
            intrinsic: (3, 3) float64 — nuScenes intrinsics в оригинальном разрешении.

        Returns:
            depth:     (H, W) float32 в метрах.
            intrinsic: None (модель DA3METRIC не предсказывает intrinsics).
        """
        return self._run(image, intrinsic)

    def _run(
        self, image: np.ndarray,
        intrinsic: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # DA3 принимает intrinsics формата (N, 3, 3) в оригинальном разрешении
        ixts = np.array([intrinsic], dtype=np.float32) if intrinsic is not None else None

        with torch.no_grad():
            prediction = self._model.inference([image], intrinsics=ixts)

        # depth: [N, H, W] → [H, W]
        depth = prediction.depth[0]
        if isinstance(depth, torch.Tensor):
            depth = depth.float().cpu().numpy()
        depth = np.clip(depth, 0.0, self.max_depth).astype(np.float32)

        return depth, None

    # ------------------------------------------------------------------
    # Выравнивание масштаба по разреженному LiDAR (опционально)
    # ------------------------------------------------------------------

    @staticmethod
    def align_scale_shift(
        depth_pred: np.ndarray,
        depth_sparse: np.ndarray,
        min_valid: int = 100,
    ) -> np.ndarray:
        """
        Выровнять предсказанную глубину по разреженному GT LiDAR
        методом наименьших квадратов: depth_aligned = scale * depth_pred + shift.

        Args:
            depth_pred:   (H, W) предсказанная глубина.
            depth_sparse: (H, W) разреженная GT-глубина (0 там, где нет точек).
            min_valid:    Минимальное число валидных LiDAR-пикселей.

        Returns:
            depth_aligned: (H, W) float32.
        """
        valid = depth_sparse > 0.1
        if valid.sum() < min_valid:
            return depth_pred

        p = depth_pred[valid]
        g = depth_sparse[valid]

        A = np.stack([p, np.ones_like(p)], axis=1)
        result, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
        scale = float(np.clip(result[0], 0.1, 10.0))
        shift = float(result[1])

        return (scale * depth_pred + shift).clip(0.0).astype(np.float32)
