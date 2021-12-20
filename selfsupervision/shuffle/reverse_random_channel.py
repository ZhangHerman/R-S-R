import torch
import torch.nn as nn
import random

#首先将4D图片转换为5D图片
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

def reverse_random_channels(image):
    #image [BT,C,H,W]
    '''4D image -> 5D image'''
    images = create5images(image) #[B,T,C,H,W]
    B,T,C,H,W = images.size() #4,16,2048,7,7
    #B_label = [random.randint(0,1) for _ in range(B)]

    '''
    print(len(B_label))
    print(type(B))
    print(B_label)
    '''
    '''0不动  1动'''

    #反传的是1/K个channels的feature map
    result_ = torch.tensor([]).cuda()
    result = torch.tensor([]).cuda()
    for b in range(B):
        #每个B所取到的channels是不同的，所以反传时，对于每个B，反传的feature map所来的channels是不同的
        cha = random.randint(0,C-1) #取出该B所要用的Channels #[4,16,2048,7,7]
        #print("cha=",cha)
        #该B所对应的T帧的第C个通道
        images_B = images[b,:,cha,:,:].view(1,T,1,H,W)
        #用于保存该B下，T帧的逆序结果
        images_T = torch.tensor([]).cuda()
        for t in range(T): #[0,...,15]
            images_T = torch.cat([images_T,images_B[:,T-t-1,:,:,:].view(1,1,H,W)],dim=0)
        images_T = images_T.view(1,T,1,H,W)
        #print(images_T.shape) #[1,16,1,7,7]
        result_ = torch.cat([result_,images_T])
    
    result = torch.cat([result_[0],result_[1],result_[2],result_[3]],dim=0)
    
    labels = torch.ones(64).long().cuda()

    return result,labels
        

        
        

    








'''
images = torch.arange(0,64.*2048.*7.*7.).view(64,2048,7,7).cuda()
reverse_random_channels(images)
'''