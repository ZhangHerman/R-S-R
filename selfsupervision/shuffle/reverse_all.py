import torch


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

def reverse_all(images):
    image = create5images(images)
    #image [B,T,C,H,W]
    B,T,C,H,W = image.size()
    result = torch.tensor([]).cuda()
    # 1->逆序  0->正序
    labels = torch.ones(B*T).long().cuda()
    for b in range(B):
        image_B = image[b] #[T,C,H,W]
        torch_T = torch.tensor([]).cuda()
        for t in range(T):
            torch_T = torch.cat([torch_T,image_B[T-t-1].view(1,C,H,W)],dim = 0)
        #torch_T -> [T,C,H,W]
        result = torch.cat([result,torch_T],dim = 0) #[BT,C,H,W]
    return result,labels


'''x = torch.arange(0,64.*1.*3.*3.).view(64,1,3,3).cuda()
y,labels = reverse_all(x)
print(y)
print(labels)
print(labels.shape)'''





