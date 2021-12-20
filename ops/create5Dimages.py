import torch
def create5Dimages(images):
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