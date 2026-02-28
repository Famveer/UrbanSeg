from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import joblib
import torch.nn.functional as F

class SVISLoader(Dataset):
    def __init__(self, dataset, seg_num_classes=20, image_transform=None, generation_transform=None, mask_transform=None):
        self.image_transform = image_transform
        self.generation_transform = generation_transform
        self.image_paths = dataset["image_path"].tolist()

        self.targets = dataset["target"].tolist()
        
        self.mask_transform = mask_transform
        if mask_transform:
            self.image_masks = dataset["mask_path"].tolist()
            self.seg_num_classes = seg_num_classes
        
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            A single sample (image, label) where the label can be inferred from the filename or other metadata.
        """
        
        return_dict = {}
        
        # Read real images
        if self.image_transform:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image = self.image_transform(image)
            return_dict["images"] = image

        # Read latent images
        if self.generation_transform:
            latent_images = Image.open(self.image_paths[idx]).convert("RGB")
            latent_images = self.generation_transform(latent_images)
            return_dict["latent_images"] = latent_images

        # Read masks
        if self.mask_transform:
            masks = joblib.load(self.image_masks[idx]).astype(np.uint8)
            masks = Image.fromarray(masks)
            masks = self.mask_transform(masks).squeeze(0).long()        # → (H, W) int64
            # Convert mask to one-hot encoding
            mask_onehot = F.one_hot(masks, num_classes=self.seg_num_classes)  # → (H, W, num_classes)
            mask_onehot = mask_onehot.permute(2, 0, 1).float()
            return_dict["masks"] = masks,
            return_dict["masks_onehot"] = mask_onehot

        return_dict["targets"] = self.targets[idx]

        return return_dict
