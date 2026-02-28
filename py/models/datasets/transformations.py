from torchvision import transforms

MODEL_SIZE = {
              # VGG
              "vgg16":           (224, 224),
              "vgg19":           (224, 224),
              # ResNet
              "resnet18":        (224, 224),
              "resnet34":        (224, 224),
              "resnet50":        (224, 224),
              "resnet101":       (224, 224),
              "resnet152":       (224, 224),
              # EfficientNet
              "efficientnet_b0": (224, 224),
              "efficientnet_b1": (240, 240),
              "efficientnet_b2": (260, 260),
              "efficientnet_b3": (300, 300),
              "efficientnet_b4": (380, 380),
              "efficientnet_b5": (456, 456),
              "efficientnet_b6": (528, 528),
              "efficientnet_b7": (600, 600),
              # Vision Transformers
              "vit_b16":         (384, 384),
              "vit_b32":         (224, 224),
              "vit_l16":         (512, 512),
              "vit_l32":         (224, 224),
              "vit_h14":         (518, 518),
              # DenseNet
              "densenet121":     (224, 224),
              "densenet169":     (224, 224),
              "densenet201":     (224, 224),
              # MobileNet
              "mobilenet_v2":    (224, 224),
              "mobilenet_v3s":   (224, 224),
              "mobilenet_v3l":   (224, 224),
              # ConvNeXt
              "convnext_tiny":   (224, 224),
              "convnext_small":  (224, 224),
              "convnext_base":   (224, 224),
              "convnext_large":  (224, 224),
              # Inception / Xception
              "inception_v3":    (299, 299),
              "xception":        (299, 299),
}

class ImageTransforms():
    def __init__(self):
        pass

    def _get_size(self, image_size=None, model_name=None):
        """Resolves the image size from either image_size or model_name."""
        if image_size is not None:
            return (image_size, image_size)
        elif model_name is not None:
            return MODEL_SIZE[model_name]
        return None  # No resizing will be applied

    def _build_augmentations(self, size, **kwargs):
        """Builds augmentation list dynamically from kwargs.
        
        Args:
            image_size (int): Resize images to this square size. Default None (no resizing)
            horizontal_flip (bool): Default True
            vertical_flip (bool): Default False
            rotation (int): Max rotation degrees. Default None (disabled)
            color_jitter (bool): Default False
            color_jitter_params (dict): Default {brightness:0.2, contrast:0.2, saturation:0.2, hue:0.1}
            random_crop (bool): RandomResizedCrop instead of Resize. Default False
            grayscale (bool): Grayscale (3ch). Default False
        """
        horizontal_flip     = kwargs.get('horizontal_flip', True)
        vertical_flip       = kwargs.get('vertical_flip', False)
        rotation            = kwargs.get('rotation', None)
        color_jitter        = kwargs.get('color_jitter', False)
        color_jitter_params = kwargs.get('color_jitter_params', {
            'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1
        })
        random_crop         = kwargs.get('random_crop', False)
        grayscale           = kwargs.get('grayscale', False)

        aug = []

        if size is not None:
            if random_crop:
                aug.append(transforms.RandomResizedCrop(size, scale=(0.8, 1.0)))
            else:
                aug.append(transforms.Resize(size))

        if horizontal_flip:
            aug.append(transforms.RandomHorizontalFlip())
        if vertical_flip:
            aug.append(transforms.RandomVerticalFlip())
        if rotation is not None:
            aug.append(transforms.RandomRotation(rotation))
        if color_jitter:
            aug.append(transforms.ColorJitter(**color_jitter_params))
        if grayscale:
            aug.append(transforms.Grayscale(num_output_channels=3))

        return aug

    def get(self, image_size=None, model_name=None, type_transform=None, **kwargs):
        
        if image_size is not None and model_name is not None:
            raise ValueError(f"Invalid combination: image_size: {image_size} and model_name: {model_name}, both cannot be not None.")

        size = self._get_size(image_size, model_name)

        if type_transform is None:
            # Return dict with both train and val
            return {
                    "train": {
                        "image_transform": self._train_transforms(size, **kwargs),
                    },
                    "val": {
                        "image_transform": self._val_transforms(size),
                    }
                }
            
        elif type_transform == "train":
            return self._train_transforms(size, **kwargs)
            
        else:
            return self._val_transforms(size)

    def _val_transforms(self, size):
        t = []
        if size is not None:
            t.append(transforms.Resize(size))
        t.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transforms.Compose(t)

    def _train_transforms(self, size, **kwargs):
        aug = self._build_augmentations(size, **kwargs)
        return transforms.Compose([
                          *aug,
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                      ])
