import os
from config import get_config

args = get_config()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

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
from config import get_config

args = get_config()

tensor_logger = TensorBoardLogger(os.path.join(args.c, "tb_logs"), name="tensor_model")
csv_logger = CSVLogger(os.path.join(args.c, "csv_logs"), name="csv_model")

model = LightImgModel()
trainer = Trainer(
    accelerator="gpu", 
    devices=-1,
    max_epochs=3,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=[tensor_logger, csv_logger],
)
trainer.fit(model)