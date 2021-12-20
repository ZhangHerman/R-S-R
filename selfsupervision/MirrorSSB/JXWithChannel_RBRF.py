import torch
import random


a = torch.arange(36).view(4,3,3)

C,H,W = a.size()

#image:[BT,C,H,W]

image_5D = create5Dimages(image)#[BT,C,H,W]->[B,T,C,H,W]



def getLabel():#随机B随机F
    B,T,C,H,W = image_5D.size()
    for b in range(B):
        label_B = [random.randint(0,1) for _ in range(B)]
    
    




def getLabel():
    T = 16
    for t in range(T):
     #得到每个B内T帧的label
        label_T = [random.randint(0,1) for _ in range(T)]
    return torch.tensor(label_T).float().cuda()