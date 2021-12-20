import torch
import random


def create5Dimages(images):
    # images : 4D tensor with shape [BT,C,H,W]
    T = 16
    B = images.size(0) // T
    C = images.size(1)
    H = images.size(2)
    W = images.size(3)
    image = torch.tensor([]).cuda()
    for b in range(B):
        bimage = images[b * T:(b + 1) * T, :, :, :].float().view(1, T, C, H, W).cuda()
        image = torch.cat([image, bimage], dim=0)
    return image  # [B,T,C,H,W]

def create4Dimages(images):
    # images : 5D tensor with shape [B,T,C,H,W]
    B,T,C,H,W = images.size()
    image = torch.tensor([]).cuda()
    for b in range(B):
        image = torch.cat([image,images[b]],dim=0)
    return image


def TmirrorZong(image):
    #image[C,H,W] W->L
    C,H,L = image.size()
    index = list(range(0,L,1))
    index.reverse()
    return  image[:,:,index]

def TmirrorHeng(image):
    #image[C,H,W] 
    C,H,W = image.size()
    index = list(range(0,H,1))
    index.reverse()
    return image[:,index]




def mirror(images,labels):
    #images[T,C,H,W]
    #labels[T]
    # 0:Zong   1:横 
    T,C,H,W = images.size()
    result = torch.Tensor().cuda()
    for l in range(len(labels)):
        if(labels[l]==0):
            image = TmirrorZong(images[l]).view(1,C,H,W)
            #image = mirror3DZong(images[l]).view(1,C,H,W)
            result = torch.cat([result,image],0)
        elif(labels[l]==1):
            image = TmirrorHeng(images[l]).view(1,C,H,W)
            #image = mirror3DHeng(images[l]).view(1,C,H,W)
            result = torch.cat([result,image],0)
    return result 
        
def getLabel():
    T = 16
    for t in range(T):
     #得到每个B内T帧的label
        label_T = [random.randint(0,1) for _ in range(T)]
    return torch.tensor(label_T).float().cuda()

#核心代码，结构1：无通道变换的直接翻转
def Mirror_Self_Supervision(images):
    #images [BT,C,H,W]
    Bt,C,H,W = images.size()
    T = 16
    B = Bt//T
    label_domain = (0,1)
    #保存最终的结果image
    image = create5Dimages(images) #[B,T,C,H,W]
    mirrorImage = torch.Tensor().cuda()
    mirrorLabel = torch.Tensor().cuda()
    for b in range(B):
        label_T = getLabel() #[T]
        mirror_image_T = mirror(image[b],label_T)  # image[b]:[T,C,H,W]   label_T:[T]
        mirrorLabel = torch.cat([mirrorLabel,label_T],0)
        mirrorImage = torch.cat([mirrorImage,mirror_image_T.view(1,T,C,H,W)],0)
    #5D->4D
    mirrorImage = create4Dimages(mirrorImage)
    return mirrorImage,mirrorLabel

