import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from dataset import Caltech101DataModule
from model import LightModule
from vgg16 import VGG16
from loguru import logger
from rich import inspect
from rich import print as rprint
from rich.console import Console
from rich.pretty import pprint
from tqdm import tqdm
from timebudget import timebudget

console = Console()

# create data module
caltech101_dm = Caltech101DataModule(batch_size=8)

# create torch model
torch_vgg16 = VGG16(num_classes=101)

# create pytorch lightning model (wrapper) by passing torch model
model = LightModule(torch_vgg16)

# create trainer
NUM_EPOCHS = 10
# callbacks = [ModelCheckpoint(
#     save_top_k=1, mode='max', monitor="val_acc")]  # save top 1 model 

logger = CSVLogger(save_dir='logs/', name='vgg16')

trainer = pl.Trainer(gpus=1, max_epochs=NUM_EPOCHS, accelerator='auto', devices='auto', logger=logger)

console.rule("Training started")
with timebudget("Training"):
    # train and validate
    trainer.fit(model, caltech101_dm)

console.rule("Plotting results")



    



