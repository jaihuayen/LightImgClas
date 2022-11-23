import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from dataset import LightImgDataset
from torch.utils.data.dataset import DataLoader

from config import get_config

class LightImgModel(pl.LightningModule):
    def __init__(self):
        super(LightImgModel, self).__init__()
        self.save_hyperparameters()

        # Parameter Settings
        self.args = get_config()

        self.num_classes = self.args.num_classes
        self.class_mapper = {'resnet50': 2048, 'mobilenet_v3_large': 960}
        self.model_mapper = {'resnet50': models.resnet50, 
                             'mobilenet_v3_large': models.mobilenet_v3_large}

        self.model = self.model_mapper[self.args.modelname](pretrained=self.args.pretrain)

        if self.modelname == 'resnet50':
            self.model.fc = nn.Linear(self.class_mapper[self.modelname], self.num_classes)

        elif self.modelname == 'mobilenet_v3_large':
            self.model.classifier[1] = nn.Linear(self.class_mapper[self.modelname],
                                                    len(self.num_classes))
        else:
            print('Does not support fine-tune {} architecture yet!'.format(self.modelname))

        if not self.args.train_all_layers:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

        # Image transformations
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), 
                                                                  (0.229, 0.224, 0.225))
                                            ])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def setup(self):
        self.train_dataset = LightImgDataset(annotation_file=self.args.train,
                                             img_dir=self.args.data,
                                             transform=self.transform)

        self.val_dataset = LightImgDataset(annotation_file=self.args.val,
                                           img_dir=self.args.data,
                                           transform=self.transform)

        self.test_dataset = LightImgDataset(annotation_file=self.args.test,
                                            img_dir=self.args.data,
                                            transform=self.transform)


    def train_dataloader(self):
        return DataLoader(dataset = self.train_dataset, 
                          batch_size = self.args.train_batch_size,
                          num_workers = self.args.workers,
                          pin_memory = True)

    def val_dataloader(self):
        return DataLoader(dataset = self.val_dataset, 
                          batch_size = self.args.val_batch_size,
                          num_workers = self.args.workers,
                          pin_memory = True)

    def test_dataloader(self):
        return DataLoader(dataset = self.test_dataset, 
                          batch_size = self.args.test_batch_size,
                          num_workers = self.args.workers,
                          pin_memory = True)