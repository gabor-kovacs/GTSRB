"""
create a VGG style net with batchnorm, maxpool, dropout at fc layers
should be small enough to run 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

n_classes = 43



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def block(in_features, out_features):
            return nn.Sequential(
                nn.Conv2d(in_features,out_features,kernel_size=3),      # 64x64
                nn.Hardswish(),
                nn.BatchNorm2d(out_features),
                nn.Conv2d(out_features,out_features,kernel_size=3),
                nn.Hardswish(),
                nn.BatchNorm2d(out_features),
                nn.MaxPool2d(kernel_size=2)
            )

        self.block1 = block(3,16)
        self.block2 = block(16,32)
        self.block3 = block(32,64)
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128,kernel_size=3),
            nn.Hardswish(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        # no softmax for nn.CrossEntropyLoss
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=128),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=n_classes)
        )

    def forward(self, x):
        x =  self.block1(x)
        x =  self.block2(x)
        x =  self.block3(x)
        x =  self.block4(x)
        x =  self.fc(x)
        return x


if __name__ == "__main__":
  model = Net()
  model.eval()
  input = torch.rand(1,3,64,64)
  output = model(input)
  print(output)
  summary(model, input_size=(1, 3, 64, 64))