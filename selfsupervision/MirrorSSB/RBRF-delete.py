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
    B, T, C, H, W = images.size()
    image = torch.tensor([]).cuda()
    for b in range(B):
        image = torch.cat([image, images[b]], dim=0)
    return image


def TmirrorZong(image):
    # image[C,H,W] W->L
    C, H, L = image.size()
    index = list(range(0, L, 1))
    index.reverse()
    return image[:, :, index]


def TmirrorHeng(image):
    # image[C,H,W]
    C, H, W = image.size()
    index = list(range(0, H, 1))
    index.reverse()
    return image[:, index]


'''def getLabel():
    T = 16 #采用16帧的设定
    for t in range(T):
     #得到每个B内T帧的label
        label_T = [random.randint(0,1) for _ in range(T)]
    return torch.tensor(label_T).float().cuda()'''


def RBRF(images, labels):  # images[BT,C,H,W]
    image = create5Dimages(images)  # image[B,T,C,H,W]
    
    B, T, C, H, W = image.size()
    # 生成Batch级标签
    labelb = [random.randint(0, 1) for _ in range(B)]
    # list->tensor
    label_B = torch.tensor(labelb).cuda()
    '''构建1级伪标签'''
    labels = torch.tensor([]).long().cuda()
    for l in range(B):
        if (label_B[l] == 0):  # 当前batch没有被选中，所以该batch内的所有帧的标签均为0
            labels = torch.cat([labels, torch.zeros(16).long().cuda()])
        elif (label_B[l] == 1):  # 当前batch被选中，所以该batch内的所有帧的标签暂定为1,表示将进入RF候选阶段
            labels = torch.cat([labels, torch.ones(16).long().cuda()])

    # 至此,1级标签创建完毕,该1级标签被表示为labels

    result = torch.tensor([]).cuda()
    labels = torch.tensor([]).long().cuda()

    for b in range(B):
        if (label_B[b] == 0):
            result = torch.cat([result, image[l, :, :, :, :]], dim=0).view(1, T, C, H, W)
            labels = torch.cat([labels, torch.zeros(16).long().cuda()])
        if (label_B[b] == 1):
            labelt = [random.randint(0, 1) for _ in range(T)]
        label_t = torch.tensor(labelt).cuda()
        # 此时需要先创建2级标签,然后用2级标签去修改1级标签
        # label_t = torch.tensor(labelt).long().cuda()
        for t in range(T):
            if (label_t[t] == 0):  # 选中的B而未选中的Frame进行Z+H变换
                # 计划是先更改相应的B的相应的label然后再具体的对result中的feature map变换
                label_t[t] = 1

            elif (label_t[t] == 1):  # 选中的B并且选中的Frame进行Z+H+C变换
                # 计划是先更改相应的B的相应的label然后再具体的对result中的feature map变换
                label_t[t] = 2

        labels = torch.cat([labels, label_t])


'''
def mirror(images,labels):
    #images[T,C,H,W]
    #labels[T]
    # 0:Zong   1:横 
    T,C,H,W = images.size()
    result = torch.Tensor().cuda()
    for l in range(len(labels)):
        if(labels[l]==0):
            image = TmirrorZong(images[l]).view(1,C,H,W)
            image_c = torch.tensor([]).cuda()
            index = list(range(0,C,1))
            index.reverse()
            image_c = image[:,index]


            #image = mirror3DZong(images[l]).view(1,C,H,W)
            result = torch.cat([result,image_c],0)
        elif(labels[l]==1):
            image = TmirrorHeng(images[l]).view(1,C,H,W)
            #image = mirror3DHeng(images[l]).view(1,C,H,W)

            image_c = torch.tensor([]).cuda()
            index = list(range(0,C,1))
            index.reverse()
            image_c = image[:,index]

            result = torch.cat([result,image_c],0)

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
    return mirrorImage,mirrorLabel'''

'''
a = torch.arange(36).view(4,3,3).float()
C,H,W = a.size()
print(a)

b = torch.tensor([]).float()
for i in range(C):
    b = torch.cat([b,a[C-i-1]],dim=0)

b = b.view(4,3,3)
print(b)
'''