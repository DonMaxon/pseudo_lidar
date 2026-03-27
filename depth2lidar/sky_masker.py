"""
sky_masker.py

Маскирование пикселей неба и/или растительности на RGB-изображении
с помощью SegFormer, обученного на ADE20K
(nvidia/segformer-b0-finetuned-ade-512-512).

Применение: обнулять пиксели в depth map перед back-projection,
чтобы не генерировать шумные точки от ненадёжных предсказаний глубины.

Установка:
    pip install transformers
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ADE20K class IDs
_ADE20K_SKY        = 2
_ADE20K_VEGETATION = [4, 9, 17]   # tree, grass, plant

_DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"


class SemanticMasker:
    """
    Маскирует заданные семантические классы в depth map перед back-projection.

    Модель запускается один раз на изображение и строит объединённую маску
    для всех выбранных классов.

    Args:
        mask_sky:        Маскировать небо (ADE20K class 2).
        mask_vegetation: Маскировать деревья, траву, кустарники
                         (ADE20K classes 4=tree, 9=grass, 17=plant).
        model_id:        HuggingFace model ID или путь к локальной директории.
        device:          'cuda', 'cpu' или 'auto'.
    """

    def __init__(
        self,
        mask_sky: bool = True,
        mask_vegetation: bool = True,
        model_id: str = _DEFAULT_MODEL,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._labels: list[int] = []
        if mask_sky:
            self._labels.append(_ADE20K_SKY)
        if mask_vegetation:
            self._labels.extend(_ADE20K_VEGETATION)

        if not self._labels:
            raise ValueError("Нужно включить хотя бы один класс для маскирования.")

        label_names = []
        if mask_sky:
            label_names.append("sky")
        if mask_vegetation:
            label_names.append("vegetation")
        print(
            f"Загрузка SemanticMasker (SegFormer ADE20K) из '{model_id}' "
            f"на {self.device} [маски: {', '.join(label_names)}] ..."
        )
        self._load_model(model_id)
        print("  SemanticMasker загружен.")

    def _load_model(self, model_id: str) -> None:
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        except ImportError as e:
            raise ImportError(
                "Установите transformers: pip install transformers"
            ) from e

        self._processor = SegformerImageProcessor.from_pretrained(model_id)
        self._model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self._model = self._model.to(self.device)
        self._model.eval()

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Предсказать объединённую маску для всех выбранных классов.

        Args:
            image: (H, W, 3) uint8 RGB.

        Returns:
            mask: (H, W) bool — True там, где нужно обнулить depth.
        """
        H, W = image.shape[:2]
        pil_img = Image.fromarray(image)

        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, num_classes, H/4, W/4)

        logits_up = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        mask = np.zeros((H, W), dtype=bool)
        for label in self._labels:
            mask |= (pred == label)
        return mask

    def apply_to_depth(
        self,
        depth_map: np.ndarray,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Обнулить замаскированные пиксели в depth map.

        Args:
            depth_map: (H, W) float32.
            image:     (H, W, 3) uint8 RGB (может быть другого разрешения).

        Returns:
            depth_masked: (H, W) float32 с обнулёнными пикселями.
            mask:         (H, W) bool — итоговая маска в разрешении depth_map.
        """
        mask_img = self.predict_mask(image)

        dH, dW = depth_map.shape[:2]
        iH, iW = mask_img.shape[:2]
        if (dH, dW) != (iH, iW):
            mask_t = torch.from_numpy(mask_img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=(dH, dW), mode="nearest")
            mask = mask_t.squeeze().numpy().astype(bool)
        else:
            mask = mask_img

        depth_masked = depth_map.copy()
        depth_masked[mask] = 0.0
        return depth_masked, mask


# Обратная совместимость
SkyMasker = SemanticMasker
