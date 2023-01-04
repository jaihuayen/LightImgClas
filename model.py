import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from dataset import LightImgDataset
from torch.utils.data import DataLoader

from torchmetrics.classification import MultilabelFBetaScore
from torch.nn import MultiLabelSoftMarginLoss

from torch.utils.data.sampler import WeightedRandomSampler

from config import get_config
from misc import *

class LightImgModel(pl.LightningModule):
    def __init__(self):
        super(LightImgModel, self).__init__()
        self.save_hyperparameters()

        # Parameter Settings
        self.args = get_config()

        self.num_classes = self.args.num_classes
        self.modelname = self.args.modelname
        self.class_mapper = {'resnet50': 2048, 'mobilenet_v3_large': 960}

        if self.modelname == 'resnet50':
            self.model = models.resnet50(pretrained=self.args.pretrain)
            self.model.fc = nn.Linear(self.class_mapper[self.modelname], self.num_classes)

        elif self.modelname == 'mobilenet_v3_large':
            self.model = models.mobilenet_v3_large(pretrained=self.args.pretrain)
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

        # Metrics calculation

        self.metric_f1_1 = MultilabelFBetaScore(beta=1.0, num_labels=self.num_classes, 
                                               threshold=0.1, weighted='weighted')
        self.metric_f1_2 = MultilabelFBetaScore(beta=1.0, num_labels=self.num_classes, 
                                               threshold=0.2, weighted='weighted')
        self.metric_f1_3 = MultilabelFBetaScore(beta=1.0, num_labels=self.num_classes, 
                                               threshold=0.3, weighted='weighted')
        self.metric_f1_4 = MultilabelFBetaScore(beta=1.0, num_labels=self.num_classes, 
                                               threshold=0.4, weighted='weighted')
        self.metric_f1 = MultilabelFBetaScore(beta=1.0, num_labels=self.num_classes, 
                                             threshold=0.5, weighted='weighted')

        self.loss_fn = MultiLabelSoftMarginLoss()

        self.weight = compute_sampler_weight(self.args.train, 'LabelName')[0]
        self.sampler = WeightedRandomSampler(weights=torch.from_numpy(np.array(self.weight)),
                                             num_samples=len(self.weight),
                                             replacement=True)

    def forward(self, x):
        self.model.eval()
        f = self.model(x)
        f = f.view(f.size(0), -1)
        return torch.sigmoid(f)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        train_f1_1 = self.metric_f1_1(y_hat, y)
        train_f1_2 = self.metric_f1_2(y_hat, y)
        train_f1_3 = self.metric_f1_3(y_hat, y)
        train_f1_4 = self.metric_f1_4(y_hat, y)
        train_f1 = self.metric_f1(y_hat, y)
        self.log("loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_f1", train_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_1", train_f1_1, on_epoch=True, logger=True)
        self.log("train_f1_2", train_f1_2, on_epoch=True, logger=True)
        self.log("train_f1_3", train_f1_3, on_epoch=True, logger=True)
        self.log("train_f1_4", train_f1_4, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        valid_f1_1 = self.metric_f1_1(y_hat, y)
        valid_f1_2 = self.metric_f1_2(y_hat, y)
        valid_f1_3 = self.metric_f1_3(y_hat, y)
        valid_f1_4 = self.metric_f1_4(y_hat, y)
        valid_f1 = self.metric_f1(y_hat, y)
        self.log("valid_f1", valid_f1, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_f1_1", valid_f1_1, on_epoch=True, logger=True)
        self.log("valid_f1_2", valid_f1_2, on_epoch=True, logger=True)
        self.log("valid_f1_3", valid_f1_3, on_epoch=True, logger=True)
        self.log("valid_f1_4", valid_f1_4, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_f1_1 = self.metric_f1_1(y_hat, y)
        test_f1_2 = self.metric_f1_2(y_hat, y)
        test_f1_3 = self.metric_f1_3(y_hat, y)
        test_f1_4 = self.metric_f1_4(y_hat, y)
        test_f1 = self.metric_f1(y_hat, y)
        self.log("test_f1", test_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_1", test_f1_1, on_epoch=True, logger=True)
        self.log("test_f1_2", test_f1_2, on_epoch=True, logger=True)
        self.log("test_f1_3", test_f1_3, on_epoch=True, logger=True)
        self.log("test_f1_4", test_f1_4, on_epoch=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def setup(self, stage=None):
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
                          sampler = self.sampler,
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