import pytorch_lightning as pl
import torchmetrics
import torch
from torch.nn import functional as F
from loguru import logger
from rich import inspect
from rich import print as rprint
from rich.console import Console
from rich.pretty import pprint
from tqdm import tqdm
from timebudget import timebudget

NUM_CLASSES = 101
console = Console()
class LightModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # console.rule('Training Step')
        x, y = batch
        # pprint(x.shape)
        # pprint(y.shape)
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        
        self.train_acc(preds, y)
        
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True,logger=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True,  logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        
        self.val_acc(preds, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, logger=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

