import torch
import torch.nn as nn

def reverse_channel_T(images):
    '''images.shape = [1,16,1,7,7],输入是切出来的某个通道'''
    T = 16
    images_T = torch.tensor([]).cuda()
    for t in range(T):
        images_T = torch.cat([images_T,images[:,T-t-1,:,:,:].view(1,1,7,7)],dim=0)
    images_T = images_T.view(1,16,1,7,7)
    
    return images_T
