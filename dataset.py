import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import Caltech101
from torchvision import transforms

from rich.console import Console
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from halib.filetype.csvfile import fn_display_df as showdf


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
    
    def stratified_split(self, dataset, test_size=0.1):
        dataset_indices = list(range(len(dataset)))
        targets = None
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        if not targets:
            # if targets is not available, we need to iterate over the dataset
            targets = []
            for i in tqdm(range(len(dataset))):
                img, label = dataset[i]
                targets.append(label)
                
        
        train_indices, val_indices, _, _ = train_test_split(dataset_indices, targets, stratify=targets, test_size=test_size, random_state=42)
        
        # generate Subset of dataset with indices
        train_data = Subset(dataset, train_indices)
        test_data = Subset(dataset, val_indices)
        
        return train_data, test_data
        
    # split data, do not use `random_split` since it is not stratified split
    def setup(self, stage: str = None) -> None:
        self.caltech101 = Caltech101(self.hparams.data_dir, transform=self.transforms)
        
        test_size = 0.1
        self.train_data, self.val_data = self.stratified_split(self.caltech101, test_size=test_size)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, shuffle=False)
    
    
    def plot_dataloader(self, data_loader, name="train"):
        counter = Counter()
        categories = self.caltech101.categories
        for batch in data_loader:
            _, label_idxs = batch
            labels = []
            for label_idx in label_idxs:
                labels.append(categories[label_idx])
            counter.update(labels)
                        
        counter = dict(sorted(counter.items(), key=lambda x: x[1]))
        plt.figure(figsize=(15, 10))
        y_pos = range(len(counter.keys()))
        y_counts = counter.values()
        plt.barh(y_pos, y_counts)
        plt.yticks(y_pos, labels=counter.keys())
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"plot/{name}_dist.png")
        return counter
        
        
    def plot_distribution(self):
        train_counter =self.plot_dataloader(self.train_dataloader(), name="train")
        val_counter = self.plot_dataloader(self.val_dataloader(), name="val")
        
        df_train = pd.DataFrame.from_dict(train_counter, orient='index', columns=['train'])
        df_val = pd.DataFrame.from_dict(val_counter, orient='index', columns=['val'])
        
        df = pd.concat([df_train, df_val], axis=1)
        df.sort_index(inplace=True)
        showdf(df)
        
        
def main():
    data_module = Caltech101DataModule(batch_size=32)
    data_module.prepare_data()
    data_module.setup()
    data_module.plot_distribution()
    # pprint(data_module.caltech101.categories)
    # shape_set = set()
    # for i in tqdm(range(len(data_module.caltech101))):
    #     img, label = data_module.caltech101[i]
    #     shape_set.add(img.shape)
    # pprint(shape_set)
    
    

if __name__ == "__main__":
    main()
    
    



