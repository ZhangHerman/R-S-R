import torch
import random

def create5images(images):
    # images : 4D tensor with shape [BT,C,H,W]
    T = 16
    B = images.size(0) // T
    C = images.size(1)
    H = images.size(2)
    W = images.size(3)
    image = torch.tensor([]).cuda()
    for b in range(B):
        bimage = images[b * T:(b + 1) * T, :, :, :].view(1, T, C, H, W)
        image = torch.cat([image, bimage], dim=0)
    return image  # [B,T,C,H,W]

def reverse(images):
    #image[BT,C,H,W]
    image = create5images(images) #[B,T,C,H,W]
    B,T,C,H,W = image.size()
    label = [random.randint(0,1) for _ in range(B)]
    #print(label)
    labels_r = torch.tensor(label).repeat(T).view(T,B).cuda()
    #print(torch.tensor(label).repeat(T).view(T,B),'..........')
    labels = torch.tensor([]).cuda()
    for b in range(B):
        labels = torch.cat([labels,labels_r[:,b].float()])
    result = torch.tensor([]).cuda()
    #print(labels)
    #print(labels.shape)
    #1 -> 逆序 0 -> 正序
    for b in range(B):
        image_B = image[b] #[T,C,H,W]
        if label[b]==0:
            result = torch.cat([result,image_B],dim=0)
        elif label[b]==1:#当前B的T帧进行逆序处理
            image_T = torch.tensor([]).cuda()
            for t in range(T):
                image_T = torch.cat([image_T,image_B[T-t-1].view(1,C,H,W)])
            result = torch.cat([result,image_T],dim = 0)
    labels = labels.long()
    return result,labels

#x = torch.arange(0,64.*1.*3.*3.).view(64,1,3,3).cuda()
#y = reverse(x)
