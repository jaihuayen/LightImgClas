import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LightImgDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.imag_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.imag_dir,self.img_labels.iloc[idx,'Path'])
        img = Image.open(img_path)
        label = self.img_labels.iloc[idx,'Label']
        if self.transform:
            img = self.transform(img)
        return img, label