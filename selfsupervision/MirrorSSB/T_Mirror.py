import torch
import random


a = torch.arange(36).view(1,4,3,3).float()
T,C,H,W = a.size()
print(a)

b = torch.tensor([]).float()
for i in range(C):
    b = torch.cat([b,a[:,C-i-1]],dim=0)

b = b.view(1,4,3,3)
print(b.shape)