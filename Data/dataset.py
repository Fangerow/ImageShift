import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations import PadIfNeeded, Compose, Normalize, LongestMaxSize
from albumentations.pytorch.transforms import ToTensorV2
from config import cfg

transform = Compose([
    LongestMaxSize(max_size=cfg['image_size']),
    PadIfNeeded(min_height=cfg['image_size'], min_width=cfg['image_size'],
                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    ToTensorV2(),
])


class PairedImageDataset(Dataset):
    def __init__(self, image_pairs, transform=None):
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_name1, img_name2 = self.image_pairs[idx]

        image1 = cv2.imread(img_name1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = image1.astype(np.float32) / 255.0

        image2 = cv2.imread(img_name2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = image2.astype(np.float32) / 255.0

        if self.transform:
            image1 = self.transform(image=image1)["image"]
            image2 = self.transform(image=image2)["image"]

        return image1, image2
