# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from ops.base_module import *
import torch.nn.functional as F
import random

#import selfsupervision.rotation.utils as ut
#import selfsupervision.shuffle.shuffle_2frames as shu
#import selfsupervision.shuffle.reverse as rev
#import selfsupervision.shuffle.reverse as rev
#import selfsupervision.shuffle.reverse_random_channel as rev
'''import selfsupervision.shuffle.reverse_random_batch_channel as revbc'''

#mport ops.Attention as attention
'''import ops.YSAttention as attention'''
#import ops.Attention_Batchsize8 as attention

'''import selfsupervision.MirrorSSB.DCNonChannel as DCnC'''

#Before RBRF Mirror
#import selfsupervision.MirrorSSB.JXNonChannel as JXnC
#import selfsupervision.MirrorSSB.JXWithChannel as JXWC

#RBRF
#import selfsupervision.MirrorSSB.RBRF as rbrf
import selfsupervision.MirrorSSB.RBRF_2 as rbrf
import selfsupervision.shuffle.reverse_random_batch_channel as rbrc
#import ops.DomainAttention_K as domainattention
#import ops.DomainAttention_K as domainattention
import ops.Strongly_constrained_self_attention as sattention
#import ops.Multi_head_self_attention as mhattention

class TDN_Net(nn.Module):

    def __init__(self,resnet_model,resnet_model1,apha,belta):
        super(TDN_Net, self).__init__()

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)
        
        # implement conv1_5 and inflate weight 
        self.conv1_temp = list(resnet_model1.children())[0]
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * 4,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5 = nn.Sequential(nn.Conv2d(12,64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv1_5[0].weight.data = new_kernels

        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnext_layer1 =nn.Sequential(*list(resnet_model1.children())[4])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.avgpool_multi = nn.AvgPool2d(7, stride=1)

        self.avg_diff = nn.AvgPool2d(kernel_size=2,stride=2)
        self.fc = list(resnet_model.children())[8]
        
        self.apha = apha
        self.belta = belta

        self.fc_shuffle_rbrf = nn.Linear(2048,2048,bias = True)
        self.fc_shuffle_rbrc = nn.Linear(1,2048,bias = True)

        #self.fc_multi = nn.Linear(2048,2048,bias = True)


        self.avgpool_rbrf = nn.AvgPool2d(7,stride=1)
        self.avgpool_rbrc = nn.AvgPool2d(7,stride=1)
        #self.domainattention = domainattention.Self_Attn(2048,4,16,7,7,256)
        '''self.Attention = attention.Self_Attn(2048,4,16,7,7,128)'''
        #self.Attention = attention.Self_Attn(2048,8,16,7,7,256)
        
        self.scsattention = sattention.SCSAttention(2048,4,8,7,7,256)
        
        #self.multi_head_attention = mhattention.Self_Attn(2048,4,16,7,7,256)



    def forward(self, x):
        x1, x2, x3, x4, x5 = x[:,0:3,:,:], x[:,3:6,:,:], x[:,6:9,:,:], x[:,9:12,:,:], x[:,12:15,:,:]
        x_c5 = self.conv1_5(self.avg_diff(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],1).view(-1,12,x2.size()[2],x2.size()[3])))
        x_diff = self.maxpool_diff(1.0/1.0*x_c5)  
        temp_out_diff1 = x_diff 
        x_diff = self.resnext_layer1(x_diff)
        x = self.conv1(x3)
        x = self.bn1(x)
        x = self.relu(x)
        #fusion layer1
        x = self.maxpool(x)
        temp_out_diff1 = F.interpolate(temp_out_diff1, x.size()[2:])
        x = self.apha*x + self.belta*temp_out_diff1
        #fusion layer2
        x = self.layer1_bak(x)
        x_diff = F.interpolate(x_diff, x.size()[2:])
        x = self.apha*x + self.belta*x_diff
        x = self.layer2_bak(x)
        x = self.layer3_bak(x)     
        x = self.layer4_bak(x) #[64,2048,7,7]

        '''images,labels = revbc.reverse_B_channel(x)'''
        '''images,labels = DCnC.Mirror_Self_Supervision(x)'''

        #images,labels = JXnC.Mirror_Self_Supervision(x)
        #images,labels = JXWC.Mirror_Self_Supervision(x)
        images_rbrf,labels_rbrf = rbrf.RBRF(x)
        images_rbrc,labels_rbrc = rbrc.RBRC(x) #[64,1,7,7]
        
        
        
        labels_rbrf = labels_rbrf.long()
        labels_rbrc = labels_rbrc.long()

        x = self.scsattention(x,images_rbrf,images_rbrc)

        ##########################################################


        images_rbrf = self.avgpool_rbrf(images_rbrf) #[64,1,1,1]
        images_rbrc = self.avgpool_rbrc(images_rbrc) #[64,1,1,1]


        x = self.avgpool(x)
        x = x.view(x.size(0),-1)


        y = images_rbrf.view(32,2048) #[64,1]
        z = images_rbrc.view(32,1)
        
 
 


        x = self.fc(x) #[64,2048]
        #mx = self.fc(mx)
        #print(mx.shape,'fc')[64,2048]
        

        y = self.fc_shuffle_rbrf(y)#[64,2048]
        z = self.fc_shuffle_rbrc(z)

        
        return x,y,labels_rbrf,z,labels_rbrc

def tdn_net(base_model=None,num_segments=8,pretrained=True, **kwargs):
    if("50" in base_model):
        resnet_model = fbresnet50(num_segments, pretrained)
        resnet_model1 = fbresnet50(num_segments, pretrained)
    else:
        resnet_model = fbresnet101(num_segments, pretrained)
        resnet_model1 = fbresnet101(num_segments, pretrained)

    if(num_segments is 8):
        model = TDN_Net(resnet_model,resnet_model1,apha=0.5,belta=0.5)
    else:
        model = TDN_Net(resnet_model,resnet_model1,apha=0.75,belta=0.25)
    return model

