import torch
import torch.nn as nn
import random
#首先将4D图片转换为5D图片
def create5images(images):
    # images : 4D tensor with shape [BT,C,H,W]
    T = 8
    B = images.size(0)//T
    C = images.size(1)
    H = images.size(2)
    W = images.size(3)
    image = torch.tensor([]).cuda()
    for b in range(B):
        bimage = images[b*T:(b+1)*T,:,:,:].view(1,T,C,H,W)
        image = torch.cat([image,bimage],dim = 0)
    return image #[B,T,C,H,W]

def RBRC(images):
    '''images:[BT,C,H,W]'''
    image = create5images(images) #[B,T,C,H,W,]
    B,T,C,H,W = image.size()
    #生成Batchsize的label
    #[0,1,1,0]
    labelb = [random.randint(0,1) for _ in range(B)]
    #list -> tensor
    label_B = torch.tensor(labelb).cuda()
    '''构建伪标签'''

    labels = torch.tensor([]).long().cuda()
    #0正序  1逆序
    for l in range(B):
        if(label_B[l]==0):
            labels = torch.cat([labels,torch.zeros(8).long().cuda()])
        elif(label_B[l]==1):
            labels = torch.cat([labels,torch.ones(8).long().cuda()])
    #labels.shape = [64]
    #保存临时结果
    result = torch.tensor([]).cuda()

    for b in range(B):
        #生成随机通道 [0,2047]，用于该B的T帧
        cha = random.randint(0,C-1)
        #取出该B下的通道cha中的T帧
        images_B = image[b,:,cha,:,:].view(1,T,1,H,W) #[1,T,1,H,W]
        #print(images_B.shape,'........')
        #保存该B下，通道cha中T帧的逆序结果
        images_T = torch.tensor([]).cuda()
        #如果batch标签等于0，则不进行逆序处理，直接将这选定第C个通道的T帧,cat到结果tensor里
        if(label_B[b]==0):
            result = torch.cat([result,images_B],dim = 0)
        #如果batch标签等于1，则进行逆序处理，先将整个选定第C个通道的T帧进行逆序处理，然后再cat到结果tensor里
        #逆序处理
        elif(label_B[b]==1):
            for t in range(T):#[0,...,15]
                images_T = torch.cat([images_T,images_B[:,T-t-1,:,:,:].view(1,1,H,W)],dim=0) #[T,1.H.W]
            images_T = images_T.view(1,T,1,H,W) #[1,T,1,H,W]
            result = torch.cat([result,images_T],dim = 0) #[B,T,1,H,W]
    #5D image -> 4D image #最终的结果
    results = torch.cat([result[0],result[1],result[2],result[3]],dim=0) #[BT,1,H,W]
    return results,labels #[BT,1,H,W] [BT]