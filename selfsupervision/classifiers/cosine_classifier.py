import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.insert(0, r'/home/zhangyongkang/TDNWRot/')
import selfsupervision.rotation.utils as ut
import selfsupervision.rotation_loss as rotloss


def cosine_fully_connected_layer(
    x_in,weight,scale=None,bias = None,normalize_x =True,normalize_w=True
):

    # x_in: a 2D tensor with shape [batch_size,num_features_in]
    # weight: a 2D tensor with shape [num_features_in,num_features_out]
    # scale: (optional) a scalar value
    # bias: (optional) a 1D tensor with shape [num_features_out]

    assert x_in.dim() == 2
    assert weight.dim() == 2
    assert x_in.size(1) == weight.size(0)

    if normalize_x:
        x_in = F.normalize(x_in,p=2,dim=1,eps=1e-12)
    if normalize_w:
        weight = F.normalize(weight,p=2,dim = 0,eps = 1e-12)
    
    x_out = torch.mm(x_in,weight) #矩阵相乘  x_in × weight = [batch_size,num_features_in] × [num_features_in,num_features_out] = [batch_size,num_features_out]
    
    if scale is not None:
        x_out = x_out*scale.view(1,-1) #给x_out乘上相应的scale参数
    
    if bias is not None:
        x_out = x_out + bias.view(1,-1) #给x_out加上相应的bias
    
    return x_out




class CosineClassifier(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        scale=20.0,
        learn_scale=False,
        bias=False,
        normalize_x=True,
        normalize_w=True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.normalize_x = normalize_x
        self.normalize_w = normalize_w

        weight = torch.FloatTensor(num_classes,num_channels).normal_(
            0.0,np.sqrt(2.0/num_channels)
        )

        

        self.weight = nn.Parameter(weight,requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_classes).fill_(0.0)
            self.bias = nn.Parameter(bias,requires_grad=True)
        else:
            self.bias = None
        scale_cls = torch.FloatTensor(1).fill_(scale)  #tensor([20.])
        self.scale_cls = nn.Parameter(scale_cls,requires_grad=learn_scale)


    def forward(self,x_in):

        assert x_in.dim() == 2
        return cosine_fully_connected_layer(
            x_in,
            self.weight.t(),
            scale = self.scale_cls,
            bias = self.bias,
            normalize_x=self.normalize_x,
            normalize_w=self.normalize_w,
        )
    
    def extra_repr(self):
        s = "num_channels={}, num_classes={}, scale_cls={} (learnable={})".format(
            self.num_channels,
            self.num_classes,
            self.scale_cls.item(),
            self.scale_cls.requires_grad,
        )

        learnable = self.scale_cls.requires_grad

        s = (
            f"num_channels={self.num_channels}, "
            f"num_classes={self.num_classes}, "
            f"scale_cls={self.scale_cls.item()} (learnable={learnable}), "
            f"normalize_x={self.normalize_x}, normalize_w={self.normalize_w}"
        )

        if self.bias is None:
            s += ", bias=False"
        
        return s



class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier_type = "cosine"
        self.num_channels = 2048
        self.num_classes = 4
        self.global_pooling = False

        if self.classifier_type == "cosine":
            bias = True
            #实例化一个cosine classifier layer
            self.layers = CosineClassifier(
                num_channels = self.num_channels,
                num_classes=self.num_classes,
                scale = 10.0,
                learn_scale=True,
                bias=bias,
            )
        #elif:nn.Lineaar()
        else:
            raise ValueError(
                "Not implemented / recognized classifier type {}".format(self.classifier_type)
            )
    
    def flatten(self):
        return (
            self.classifier_type == "cosine"
        )
    
    def forward(self,features):
        if self.global_pooling:
            features = global_pooling(features,pool_type="avg")
        if features.dim()>2 and self.flatten():
            features = features.view(features.size(0),-1)
        
        scores = self.layers(features)
        return scores

def create_model():
    return Classifier()


'''
#创建自监督分类器
classifier = create_model()
#定义图片[64,2048,7,7]
x = torch.arange(1.,64.*2048.*7.*7.+1).view(64,2048,7,7)
#旋转图片，获得旋转标签
images_rot,labels_rot = ut.randomly_rotate_images(x)
avgpool = nn.AvgPool2d(7,stride=1) #[64,2048,1,1]
x = avgpool(x).view(x.size(0),-1) #[64,2048]
loss_rot = rotloss.rotation_task(classifier,x,labels_rot)
print(loss_rot)
'''


'''
classifier = create_model()
x = torch.arange(1,64.*2048.*7.*7.+1).view(64,2048,7,7)
avgpool = nn.AvgPool2d(7, stride=1)
x = avgpool(x).view(x.size(0),-1)[64,2048]
scores = classifier(x)
print(scores)
'''