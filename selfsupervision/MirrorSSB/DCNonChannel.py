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

#处理2D图片-纵轴
def mirror2DZong(image):
    #image[H,L]
    image = image.float().cuda()
    result = torch.Tensor().cuda()
    H,L = image.size()
    for i in range(L):
        #print(image[:,L-i-1])
        result = torch.cat([result,image[:,L-i-1]],0)
    return result.view(L,H).t() 

#处理2D图片-横轴
def mirror2DHeng(image):
    #image(H,L)
    image = image.float().cuda()
    result = torch.Tensor().cuda()
    H,L = image.size()
    for j in range(H):
        #print(image[H-j-1,:])
        result = torch.cat([result,image[H-j-1,:]],0)
    return result.view(H,L)
#处理3D图片-纵轴
def mirror3DZong(image):
    #image[C,H,L]
    image = image.float().cuda()
    result = torch.Tensor().cuda()
    C,H,L = image.size()
    for i in range(C):
        result = torch.cat([result,mirror2DZong(image[i,:,:])],0)
    return result.view(C,H,L)
#处理3D图片-横轴
def mirror3DHeng(image):
    #image[C,H,L]
    image = image.float().cuda()
    result = torch.Tensor().cuda()
    C,H,L = image.size()
    for j in range(C):
        result = torch.cat([result,mirror2DHeng(image[j,:,:])],0)
    return result.view(C,H,L)

def mirror(images,labels):
    #images[T,C,H,W]
    #labels[T]
    # 0:Zong   1:横 
    T,C,H,W = images.size()
    result = torch.Tensor().cuda()
    for l in range(len(labels)):
        if(labels[l]==0):
            image = mirror3DZong(images[l]).view(1,C,H,W)
            result = torch.cat([result,image],0)
        elif(labels[l]==1):
            image = mirror3DHeng(images[l]).view(1,C,H,W)
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
    






    


'''x = torch.arange(0,4*3*6).view(4,3,6)
print('原始的x=\n',x)'''

'''#测试处理2D图片——纵轴
print(mirror2DZong(x))'''

'''#测试处理2D图片——横轴
print(mirror2DHeng(x))'''

'''#测试处理3D图片——纵轴
print(mirror3DZong(x))'''

'''#测试处理3D图片——横轴
print(mirror3DHeng(x))'''

'''T = 16
label_domain = (0,1)
label_T = []
print(label_T)
for i in range(T):
    label_T = [random.randint(0,1) for _ in range(T)]
print(label_T)

images = torch.arange(0,16*3*3*3).view(16,3,3,3)
print(mirror(images,label_T))'''



'''x = torch.arange(0,4*16*2048*7*7).float().view(64,2048,7,7).cuda()
y = create5Dimages(x)
B,T,C,H,W = y.size()
print(B,T,C,H,W)'''

#image = torch.arange(0,64*3*3*3).view(4,16,3,3,3)
'''mirrorLabel = torch.Tensor()
label_T = getLabel()
for i in range(4):
    mirrorLabel = torch.cat([mirrorLabel,label_T],0)
print(mirrorLabel)
print(mirrorLabel.shape)'''


#主要测试代码
'''x = torch.arange(0,4*16*3*3*3).view(4*16,3,3,3)
mirrorImage,mirrorLabel = Mirror_Self_Supervision(x)
print('原始输入图片=\n',x)
print('mirrorImage=\n',mirrorImage)
print(mirrorImage.shape)
print('mirrorLabel=\n',mirrorLabel)
print(mirrorLabel.shape)'''
