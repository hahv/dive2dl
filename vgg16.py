import torch.nn as nn
from argparse import ArgumentParser
from torchinfo import summary

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same')
        self.do_pooling = pooling
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
    def forward(self, x):
        out = self.conv1(x)
        out = nn.ReLU()(out)
        if self.do_pooling:
            out = self.pool(out)
        return out
   


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=2, pooling=True):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(repeat):
            layers.append(BasicBlock(in_channels, out_channels, pooling=(i == repeat - 1)))
            in_channels = out_channels
            
        self.nn_layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.nn_layers(x)
        
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        
        self.block1 = VGGBlock(3, 64, repeat=2)
        self.block2 = VGGBlock(64, 128, repeat=2)
        self.block3 = VGGBlock(128, 256, repeat=3)
        self.block4 = VGGBlock(256, 512, repeat=3)
        self.block5 = VGGBlock(512, 512, repeat=3)        
        self.features = nn.Sequential(self.block1, self.block2, self.block3, self.block4, self.block5, nn.Flatten())
        
        self.classifier = nn.Sequential( nn.Linear(7*7*512, 4096), nn.Linear(4096, 4096), nn.Linear(4096, 1000), nn.Linear(1000, num_classes))
        
        
    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits

def main():
    model = VGG16(101)
    batch_size = 1
    summary(model, input_size=(batch_size, 3, 224, 224), device='cpu', depth=6)
    
if __name__ == "__main__":
    main()