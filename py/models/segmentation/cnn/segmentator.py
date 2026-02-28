import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
import numpy as np
from collections import Counter
from PIL import Image

import mxnet as mx
from mxnet import image as mx_image
import gluoncv
from gluoncv.utils.viz import get_color_pallete

from gluoncv.data import ADE20KSegmentation, CitySegmentation

_MODEL_REGISTRY: Dict[tuple, str] = {
    ("ade20k",     "deeplab"): "deeplab_resnet101_ade",
    ("ade20k",     "pspnet"):  "psp_resnet101_ade",
    ("cityscapes", "deeplab"): "deeplab_resnet101_citys",
    ("cityscapes", "pspnet"):  "psp_resnet101_citys",
}

_NUM_CLASSES: Dict[str, int] = {
    "ade20k":     150,
    "cityscapes":  19,
}

_CLASSES_NAMES: Dict[str, str] = {
    "ade20k":     [ x.split(',')[0].replace(" ", "_") for x in list(ADE20KSegmentation.CLASSES)],
    "cityscapes":  [ "_".join(x.split()) for x in list(CitySegmentation.CLASSES)],
}

SUPPORTED_DATASETS  = ("ade20k", "cityscapes")
SUPPORTED_BACKBONES = ("deeplab", "pspnet")


class ConvMaskClassifier:
    """
    Semantic segmentation using GluonCV pretrained models.

    Parameters
    ----------
    dataset : str
        Target label vocabulary: ``'ade20k'`` (150 classes) or
        ``'cityscapes'`` (19 classes).
    backbone : str
        Model architecture: ``'deeplab'`` or ``'pspnet'``.
    ctx : mxnet.Context, optional
        Inference context.  Defaults to GPU 0 when available, CPU otherwise.
    """

    def __init__(self,
                    dataset:  str = "ade20k",
                    backbone: str = "deeplab",
                    load_model: bool = True,
                    ctx: Optional["mx.Context"] = None,
                ):
                
        dataset  = dataset.lower()
        backbone = backbone.lower()

        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset must be one of {SUPPORTED_DATASETS}, got '{dataset}'."
            )
        if "deeplab" not in backbone.lower() and "psp" not in backbone.lower():
            raise ValueError(
                f"backbone must be one of {SUPPORTED_BACKBONES}, got '{backbone}'."
            )
        
        if "deeplab" in backbone.lower():
            self.backbone = SUPPORTED_BACKBONES[0]
        elif "psp" in backbone.lower():
            self.backbone = SUPPORTED_BACKBONES[1]
        else:
            raise ValueError("No model selected")

        self.dataset     = dataset if dataset.lower() != "cityscapes" else "citys"
        self.class_names = _CLASSES_NAMES[dataset]
        self.num_classes = _NUM_CLASSES[dataset]

        # Context
        if ctx is None:
            self.ctx = (
                mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
            )
        else:
            self.ctx = ctx

        # Load pretrained model
        model_name = _MODEL_REGISTRY[(dataset, self.backbone)]
        if load_model:
            print(f"[Segmentator] Loading '{model_name}' …")
            self.model = gluoncv.model_zoo.get_model(model_name, pretrained=True)
            print(f"[Segmentator] Ready  — dataset={dataset}, backbone={backbone}, "
                  f"ctx={self.ctx}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def get_model(self):
        return self.model
    
    def _load_and_transform(self, 
                            image_path: Union[str, Path], 
                            transforms_list: Optional[Callable] = None,
                            ) -> "mx.nd.NDArray":
        """Read an image from disk and apply the segmentation test transform."""
        image_path = str(image_path)
        img = mx_image.imread(image_path)  # MXNet NDArray, HWC, uint8

        if transforms_list:
            # Convert MXNet NDArray → PIL Image (what torchvision expects)
            pil_img = Image.fromarray(img.asnumpy().astype(np.uint8))
            tensor  = transforms_list(pil_img)        # torch.Tensor (3, H, W)
            return mx.nd.array(tensor.numpy())        # MXNet NDArray (3, H, W)

        return img

    def _infer(self, img_tensor: "mx.nd.NDArray") -> np.ndarray:
        """Run forward pass and return the argmax label map (H × W)."""
        # Ensure tensor is 4-D: (1, C, H, W)
        if img_tensor.ndim == 3:
            img_tensor = mx.nd.expand_dims(img_tensor, axis=0)
        output  = self.model.predict(img_tensor)           # (1, C, H, W)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1))  # (H, W)
        return predict.asnumpy().astype(np.int32)

    def _get_color_dict(self):
        # Create a dummy single-pixel mask per class and extract its color
        dummy = np.zeros((1, 1), dtype=np.uint8)
        img = get_color_pallete(dummy, dataset=self.dataset)
        palette = img.getpalette()  # Returns flat list [R, G, B, R, G, B, ...]
        
        # Convert flat list to dict {class_idx: (R, G, B)}
        color_dict = {
            self.class_names[i-1]: tuple(palette[i*3 : i*3+3])
            for i in range(len(palette) // 3)
        }
        
        color_list = [(self.class_names.index(k)+1, k, v) if k!="bicyle" else (self.class_names.index(k)+1, "bicycle", v) for (k, v) in color_dict.items()]
        color_list.insert( 0, ( 0, "background", (0, 0, 0)) )
        color_list.sort(key=lambda x: x[0])
        return color_list

    def _get_objects_ratio(self, masks, factor=1.0):
    
        color_list = self._get_color_dict()
    
        total_pixels = masks.shape[0]*masks.shape[1]
        flattened = [str(element) for row in masks for element in row]
        count_dict = Counter(flattened)
        count_dict = { color_list[int(k)][1]: v/total_pixels*factor  for k, v  in count_dict.items() }
        return count_dict


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, 
                image_path: Union[str, Path], 
                transforms_list: Optional[Callable] = None,
                ) -> np.ndarray:
        """
        Run segmentation on a single image.

        Parameters
        ----------
        image_path : str or Path

        Returns
        -------
        np.ndarray
            Integer label map of shape ``(H, W)`` where each value is a
            class index in ``[0, num_classes)``.
        """
        img    = self._load_and_transform(image_path, transforms_list)
        return self._infer(img)

    def extract_masks(self,
                       image_path: Union[str, Path],
                       return_overlay: bool = True,
                       alpha: float = 0.6,
                       transforms_list: Optional[Callable] = None,
                      ) -> np.ndarray:
        """
        Compute per-class pixel-coverage features for a single image.

        Each element ``features[c]`` is the fraction of pixels assigned to

        Parameters
        ----------
        image_path : str or Path
            Divide counts by total number of pixels.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(num_classes,)``.
        """
        label_map = self.predict(image_path, transforms_list)
        palette_img = get_color_pallete(label_map, self.dataset)
        
        if return_overlay:
            palette_rgb = palette_img.convert("RGB")
            image = Image.open(image_path).convert("RGB")
            image_overlay = Image.blend(image, palette_rgb, alpha=alpha)
            return label_map+1, palette_img, image_overlay
        
        return label_map+1, palette_img

    def visualize(self,
                  image_path: Union[str, Path],
                  save_path:  Optional[Union[str, Path]] = None,
                  transforms_list: Optional[Callable] = None,
                 ) -> "PIL.Image.Image":
        """
        Produce a colour-coded segmentation mask.

        Parameters
        ----------
        image_path : str or Path
        save_path  : str or Path, optional
            If provided, the palette image is saved to this path.

        Returns
        -------
        PIL.Image.Image
            Palette image with dataset colour coding.
        """
        predict     = self.predict(image_path, transforms_list)
        palette_img = get_color_pallete(predict, self.dataset)

        if save_path is not None:
            palette_img.save(str(save_path))
            print(f"[Segmentator] Mask saved → {save_path}")

        return palette_img

