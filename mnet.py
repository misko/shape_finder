import torch.nn as nn
from groupnorm import *

class MNET(nn.Module):
    def __init__(self):
        # conv
        f=32
	super(MNET, self).__init__()
        self.conv1_1 = nn.Conv2d(1, f, 11, padding=5)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(f) # GroupNorm(f,f)
        self.conv1_2 = nn.Conv2d(f, f, 11, padding=5)
        self.relu1_2 = nn.ReLU(inplace=True)
        #self.norm2 = GroupNorm(f,f)
        self.norm2 = nn.BatchNorm2d(f) # GroupNorm(f,f)
        self.conv1_3 = nn.Conv2d(f, f, 11, padding=5)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.norm3 = nn.BatchNorm2d(f) # GroupNorm(f,f)
        self.conv1_4 = nn.Conv2d(f, f, 5, padding=2)
        self.relu1_4 = nn.ReLU(inplace=True)
        self.norm4 = nn.BatchNorm2d(f) # GroupNorm(f,f)
        self.conv1_5 = nn.Conv2d(f, f, 5, padding=2)
        self.relu1_5 = nn.ReLU(inplace=True)
        self.norm5 = nn.BatchNorm2d(f) # GroupNorm(f,f)
        self.conv1_6 = nn.Conv2d(f, f, 5, padding=2)
        self.relu1_6 = nn.ReLU(inplace=True)
        self.norm6 = nn.BatchNorm2d(f) # GroupNorm(f,f)
        self.conv1_7 = nn.Conv2d(f, f, 5, padding=2)
        self.relu1_7 = nn.ReLU(inplace=True)
        self.norm7 = nn.BatchNorm2d(f) # GroupNorm(f,f)
        self.conv1_8 = nn.Conv2d(f, 2, 5, padding=2)
        self.norm8 = nn.BatchNorm2d(f) # GroupNorm(f,f)

    def forward(self, x):
        h = x
        #h = self.relu1_1(self.norm1(self.conv1_1(h)))
        h = self.relu1_1(self.conv1_1(h))
        #h = self.relu1_2(self.norm2(self.conv1_2(h)))
        h = self.relu1_2(self.conv1_2(h))
        #h = self.relu1_3(self.norm3(self.conv1_3(h)))
        h = self.relu1_3(self.conv1_3(h))
        #h = self.relu1_4(self.norm4(self.conv1_4(h)))
        #h = self.relu1_4(self.conv1_4(h))
        #h = self.relu1_5(self.norm5(self.conv1_5(h)))
        #h = self.relu1_5(self.conv1_5(h))
        #h = self.relu1_6(self.norm6(self.conv1_6(h)))
        #h = self.relu1_6(self.conv1_6(h))
        #h = self.relu1_7(self.norm7(self.conv1_7(h)))
        #h = self.relu1_7(self.conv1_7(h))
	h = self.conv1_8(h)
        return h

