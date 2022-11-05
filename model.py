import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class LightImgModel(pl.LightningModule):
    def __init__(self, modelname, num_classes, pretrain, train_all_layers):
        super(LightImgModel, self).__init__()

        self.modelname = modelname
        self.num_classes = num_classes
        self.pretrain = pretrain
        self.train_all_layers = train_all_layers
        self.class_mapper = {'resnet50': 2048, 'mobilenet_v3_large': 960}
        self.model_mapper = {'resnet50': models.resnet50, 
                             'mobilenet_v3_large': models.mobilenet_v3_large}

        self.model = self.class_mapper[self.modelname](pretrained=self.pretrain)

        if self.modelname == 'resnet50':
            self.model.fc = nn.Linear(self.class_mapper[self.modelname], self.num_classes)

        elif self.modelname == 'mobilenet_v3_large':
            self.model.classifier[1] = nn.Linear(self.class_mapper[self.modelname],
                                                    len(self.num_classes))
        else:
            print('Does not support fine-tune {} architecture yet!'.format(self.modelname))

        if not self.train_all_layers:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)