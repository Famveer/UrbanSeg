from torch.utils.data import Dataset
from PIL import Image

class StreetViewDataset(Dataset):
    """
    Dataset for street view images with segmentation masks and beauty scores.
    
    Expected structure:
    - images: RGB images (.jpg, .png)
    - masks: Segmentation masks (.png) with class indices per pixel
    - scores.csv: Columns ['image_name', 'score']
    """
    def __init__(self, dataset, seg_num_classes, transform=None):
        self.image_paths = dataset["image_path"].tolist()
        self.image_masks = dataset["mask_path"].tolist()
        self.target = dataset["target"].tolist()
        self.label = dataset["label"].tolist()
        
        self.img_size = img_size
        self.seg_num_classes = seg_num_classes
        
        print(f"Loaded {len(self.image_paths)} images")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
          # Load image
          image = Image.open(self.image_paths[idx]).convert("RGB")
          image = image.resize((self.img_size, self.img_size), Image.LANCZOS)
          image = np.array(image) / 127.5 - 1  # Normalize to [-1, 1]
          image = torch.FloatTensor(image).permute(2, 0, 1)
        
        # Load mask
        mask = joblib.load(self.image_masks[idx]).astype(np.uint8)
        mask = Image.fromarray(mask)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask).astype(np.int64)
        
        # Convert mask to one-hot encoding
        mask_onehot = np.zeros((self.seg_num_classes, self.img_size, self.img_size), dtype=np.float32)
        for c in range(self.seg_num_classes):
            mask_onehot[c][mask == c] = 1.0
        mask_onehot = torch.FloatTensor(mask_onehot)
        
        # Get score
        target = torch.FloatTensor([self.target[idx]])
        label = self.label[idx]
        
        return {
                'image': image,
                'mask': mask_onehot,
                'target': target,
                'label': label
            }
