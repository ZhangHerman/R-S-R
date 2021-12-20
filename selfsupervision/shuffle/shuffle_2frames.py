import torch
import random

def create5images(images):
    # images : 4D tensor with shape [BT,C,H,W]
    T = 16
    B = images.size(0)//T
    C = images.size(1)
    H = images.size(2)
    W = images.size(3)
    image = torch.tensor([]).cuda()
    for b in range(B):
        bimage = images[b*T:(b+1)*T,:,:,:].view(1,T,C,H,W)
        image = torch.cat([image,bimage],dim = 0)
    return image #[B,T,C,H,W]

def randomly_create2frames_labels():
    label = torch.IntTensor(random.sample(range(0,16),2))
    if label[0].item() > label[1].item():
        label = torch.cat([label[1].view(1,1),label[0].view(1,1)],dim=0).view(2)
    return label #[2]

def create_shuffle_images(images,label): #要求label为正序 [3,5]
    #images:[T,C,H,W]
    T,C,H,W = images.size()
    label1 = label[0].item() #3
    label2 = label[1].item() #5
    shuffle_imageT = torch.tensor([]).cuda()
    for t in range(T):
        if t!=label1 and t!= label2 :
            shuffle_imageT = torch.cat([shuffle_imageT,images[t].view(1,C,H,W)],dim = 0)
        elif t==label1:
            shuffle_imageT = torch.cat([shuffle_imageT,images[label2].view(1,C,H,W)],dim = 0)
        elif t == label2:
            shuffle_imageT = torch.cat([shuffle_imageT,images[label1].view(1,C,H,W)],dim = 0)
    
    shuffle_imageT = shuffle_imageT
    return shuffle_imageT # shuffled image with shape [T,C,H,W]

#if T = 16
def create_shuffled_label(T,label):#[2]
    shuffled_label = torch.tensor([])
    label1 = label[0].item()
    label2 = label[1].item()
    #0是正序  1是非正序
    l = [0 for i in range(T)]
    l[label1] = 1
    l[label2] = 1
    shuffled_label = torch.tensor(l).cuda()
    return shuffled_label #[16]

#上面的label不是最终要计算loss的label,最终计算loss的label
#只有两个类别，0和1,0表示正序，1表示非正序
def create_shuffle_images_with_batch(images):
    #images : 4D [BT,C,H,W]
    image = create5images(images).cuda() #[BT,C,H,W] -> [B,T,C,H,W]
    B,T,C,H,W = image.size()
    shuffled_images = torch.tensor([]).cuda()
    shuffled_labels = torch.tensor([]).cuda()
    for b in range(B):
        image_T = image[b] #取出一个B的frame [T,C,H,W]
        label = randomly_create2frames_labels()#正序label with shape [2]
        shuffled_label = create_shuffled_label(T,label).float()
        image_T = create_shuffle_images(image_T,label)
        #[BT,C,H,W]
        shuffled_images = torch.cat([shuffled_images,image_T],dim = 0)
        shuffled_labels = torch.cat([shuffled_labels,shuffled_label])
    shuffled_labels = shuffled_labels.long()
    return shuffled_images,shuffled_labels

'''
整体测试
x = torch.arange(0,64.*2048.*3.*3.).view(64,2048,3,3)
shuffled_images,shuffled_labels = create_shuffle_images_with_batch(x)
print(shuffled_images.shape)
print(shuffled_labels)
'''

'''
#测试shuffle是否正确，测试通过
#将randomly_create2frames_labels()中的range(0,16)变为(0,4)
#因为此时[4,1,3,3]中，T=4
x = torch.arange(0,4.*1.*3.*3.).view(4,1,3,3)
labels = randomly_create2frames_labels()
print(labels)
shuffle_imageT = create_shuffle_images(x,labels)
print(shuffle_imageT)
'''

'''
x = torch.arange(0,16).view(4,4)
l = randomly_create2frames_labels()
a,b = create_shuffle_images(x,l)
print(a,b)
'''