import torch

def create4Dimages(images):
    # images : 5D tensor with shape [B,T,C,H,W]
    B,T,C,H,W = images.size()
    image = torch.tensor([]).cuda()
    for b in range(B):
        image = torch.cat([image,images[b]],dim=0)
    return image

