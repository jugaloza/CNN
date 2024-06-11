
import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,16,3,1,1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.classifier = nn.Linear(3*3*64,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.batchnorm1(self.conv1(x))),2) # B, 2, 14,14
         
        x = F.max_pool2d(F.relu(self.batchnorm2(self.conv2(x))),2) # B, 4, 7,7
        
        x = F.max_pool2d(F.relu(self.batchnorm3(self.conv3(x))),2) # B, 8, 3,3
        
        x = self.classifier(torch.flatten(x,start_dim=1))
        return x
