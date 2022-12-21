import pytorch_lightning as pl
from dataset import Caltech101DataModule
from model import LightModule
from vgg16 import VGG16

# create data module
caltech101_dm = Caltech101DataModule(batch_size=32)

# create torch model
torch_vgg16 = VGG16(num_classes=101)

# create pytorch lightning model (wrapper) by passing torch model
model = LightModule(torch_vgg16)

# create trainer
NUM_EPOCHS = 10
trainer = pl.Trainer(gpus=1, max_epochs=NUM_EPOCHS, accelerator='auto')

# train and validate
trainer.fit(model, caltech101_dm)



    



