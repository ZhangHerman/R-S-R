import torch
import random

def create5Dimages(images):#[BT,C,H,W]->[B,T,C,H,W]
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


def create4Dimages(images):#[B,T,C,H,W]->[BT,C,H,W]
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


def RBRF(images):#image->[BT,C,H,W]
    image = create5Dimages(images)#image->[B,T,C,H,W]
    B,T,C,H,W = image.size()
    #生成batch级标签-1-level标签
    labelb = [random.randint(0, 1) for _ in range(B)]
    label_B = torch.tensor(labelb).cuda()
    #创建总标签
    labels = torch.tensor([]).long().cuda()
    #创建总的结果集
    mirrorImage = torch.tensor([]).cuda()
    
    
    for l in range(B):
        #创建保存每个B内T帧标签的集合
        label_1_t = torch.tensor([]).long().cuda()
        if(label_B[l]==0):
            #当前batch没有被选中，此时，该batch内的T帧不做任何的变动，并且每一帧的伪标签为0
            #首先直接给最终的标签集合cat T个0
            labels = torch.cat([labels,torch.zeros(16).long().cuda()])
            #将该batch内的T帧原封不动的直接cat到结果集中
            
            mirrorImage = torch.cat([mirrorImage,image[l].view(1,T,C,H,W)],dim=0) #[1,T,C,H,W]
        #########################################################################
        elif(label_B[l]==1):
            #当前batch被选中，此时，在该batch内部随机挑选frame
            #选中的batch的1-level标签
            #创建保存每个B内结果的集合
            result_B = torch.tensor([]).cuda()
            #生成2-level标签
            #生成T个2级标签
            labelt_2 = [random.randint(0, 1) for _ in range(T)] 
            label_t_2 = torch.tensor(labelt_2).cuda()

            #遍历label_t_2
            for t in range(T):
                if(label_t_2[t]==0):
                    #未选中的F,此时只需要进行不带通道变换的轴旋转变换,具体是Z还是H，需要3级标签来进行具体的实现
                    #3-level标签
                    labelt_3 = [random.randint(0, 1) for _ in range(1)]
                    label_t_3 = torch.tensor(labelt_3).cuda()
                    if(label_t_3[0]==0):
                        #Z
                        #首先直接修改1级标签
                        ll = [1]
                        l_l = torch.tensor(ll).cuda() #list->tensor
                        label_1_t = torch.cat([label_1_t,l_l])
                        #进行变换
                        result_t = TmirrorZong(image[l][t]) #image[l][t]->[C,H,W]
                        result_B = torch.cat([result_B,result_t.view(1,C,H,W).cuda()])

                    elif(label_t_3[0]==1):
                        #H
                        ll = [2]
                        l_l = torch.tensor(ll).cuda()
                        label_1_t = torch.cat([label_1_t,l_l])
                        #进行变换
                        result_t = TmirrorHeng(image[l][t]) #image[l][t]->[C,H,W]
                        result_B = torch.cat([result_B,result_t.view(1,C,H,W).cuda()])
                elif(label_t_2[t]==1):
                    #选中的batch，
                    #3-level标签
                    labelt_3 = [random.randint(0, 1) for _ in range(1)]
                    label_t_3 = torch.tensor(labelt_3).cuda()
                    if(label_t_3[0]==0):
                        #Z
                        #首先直接修改1级标签
                        ll = [3]
                        l_l = torch.tensor(ll).cuda()
                        label_1_t = torch.cat([label_1_t,l_l])
                        #进行变换
                        tt = TmirrorZong(image[l][t]).view(1,C,H,W)
                        result_t = torch.tensor([]).cuda()
                        index = list(range(0, C, 1))
                        index.reverse()
                        result_t = tt[:, index]
                        result_B = torch.cat([result_B,result_t.view(1,C,H,W).cuda()])
                    elif(label_t_3[0]==1):
                        #Z
                        ll = [4]
                        l_l = torch.tensor(ll).cuda()
                        label_1_t = torch.cat([label_1_t,l_l])
                        #进行变换
                        tt = TmirrorHeng(image[l][t]).view(1,C,H,W)
                        result_t = torch.tensor([]).cuda()
                        index = list(range(0, C, 1))
                        index.reverse()
                        result_t = tt[:, index]
                        result_B = torch.cat([result_B,result_t.view(1,C,H,W).cuda()])
            mirrorImage = torch.cat([mirrorImage, result_B.view(1, T, C, H, W)],dim=0)

        labels = torch.cat([labels,label_1_t])#[BT]

        

    mirrorImage = create4Dimages(mirrorImage) #[BT,C,H,W]

    return mirrorImage,labels
