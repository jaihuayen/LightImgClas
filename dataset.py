import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile

from config import get_config

args = get_config()

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LightImgDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.img_labels.loc[idx,'Path'])
        img = (
            Image.open(img_path)
            .convert('RGB')
        )
        label = (
            self.img_labels.loc[idx,'LabelName']
            .split(",")
        )
        label = list(map(int,label))

        if self.transform:
            img = self.transform(img)

        ids_tensor = torch.LongTensor(np.array(label))
        label_onehot = torch.FloatTensor(args.num_classes)
        label_onehot.zero_()

        label_onehot.scatter_(0, ids_tensor, 1)

        return img, label_onehot.squeeze()