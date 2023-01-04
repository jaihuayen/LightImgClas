import os
from config import get_config

args = get_config()

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from model import LightImgModel
import pytorch_lightning.callbacks as plc

def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='loss_epoch',
        mode='min',
        patience=5,
        min_delta=0.001
    ))
    callbacks.append(plc.ModelCheckpoint(
        monitor='loss_epoch',
        filename='best-{epoch:02d}-{val_f1:.3f}',
        save_top_k=1,
        mode='min',
        save_last=True
    ))
    callbacks.append(plc.TQDMProgressBar(
        refresh_rate=1
    ))
    return callbacks

tensor_logger = TensorBoardLogger(os.path.join(args.c, "tb_logs"), name="tensor_model")
csv_logger = CSVLogger(os.path.join(args.c, "csv_logs"), name="csv_model")

model = LightImgModel()

if __name__ == "__main__":
    trainer = Trainer(
        accelerator="gpu", 
        devices=-1,
        max_epochs=args.num_epochs,
        callbacks=load_callbacks(),
        logger=[tensor_logger, csv_logger],
        strategy="ddp_find_unused_parameters_false",
        default_root_dir=args.c,
        log_every_n_steps=1
    )
    trainer.fit(model)
