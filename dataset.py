import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Caltech101
from torchvision import transforms

from loguru import logger
from rich import inspect
from rich import print as rprint
from rich.console import Console
from rich.pretty import pprint
from tqdm import tqdm
from timebudget import timebudget
import cv2
import matplotlib.pyplot as plt

console = Console()
class Caltech101DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir='.') -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # this is a hack to convert grayscale images to rgb
        # since caltech101 has grayscale images
        # we need to convert them to rgb for normalization(3 channels) std/mean
        grayscale_to_rgb= transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            grayscale_to_rgb,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # download data
    def prepare_data(self):
        Caltech101(self.hparams.data_dir, download=True, transform=self.transforms)
        
        
    # split data
    def setup(self, stage: str = None) -> None:
        self.caltech101 = Caltech101(self.hparams.data_dir, transform=self.transforms)
        
        train_ratio = 0.8
        train_size = int(train_ratio * len(self.caltech101))
        val_size = len(self.caltech101) - train_size
        
        self.train_data, self.val_data = random_split(self.caltech101, [train_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, shuffle=False)

def main():
    data_module = Caltech101DataModule(batch_size=32)
    data_module.prepare_data()
    data_module.setup()
    pprint(data_module.caltech101.categories)
    shape_set = set()
    for i in tqdm(range(len(data_module.caltech101))):
        img, label = data_module.caltech101[i]
        shape_set.add(img.shape)
    pprint(shape_set)
    
    

if __name__ == "__main__":
    main()
    
    



