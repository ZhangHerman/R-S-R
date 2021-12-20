import torch
import torch.nn as nn
import ops.create5Dimages as create5D
import ops.create4Dimages as create4D

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

#新加模块
class Self_Attn(nn.Module):
    """Self attention Layer"""
    #(2048,4,16,7,7,4096)  （2048,4,16,7,7,256）
    #调用：self.Attention = attention.Self_Attn(2048,4,16,7,7,256)

    def __init__(self,in_dim,batch_size,num_frames,width,height,channels):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.r = 8
        self.conv1 = nn.Conv2d(in_channels = self.chanel_in,out_channels = self.chanel_in//self.r,kernel_size = 1)
        self.conv2 = nn.Conv2d(in_channels = self.chanel_in , out_channels = self.chanel_in//self.r , kernel_size= 1)
        self.conv3 = nn.Conv2d(in_channels = self.chanel_in,out_channels = self.chanel_in//self.r,kernel_size = 1)
        self.value_conv = nn.Conv3d(in_channels = self.chanel_in//self.r , out_channels = self.chanel_in , kernel_size= 1,stride = 1,padding = 0,dilation = 1,groups = 1,bias = True)
        
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim = -1)
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.channels = channels
        self.layer2_downgama = nn.Parameter(torch.zeros(1))
        self.layer2_upgamma = nn.Parameter(torch.zeros(1))
        #最后一层的1×1×1卷积核的初始化权重没有被全0初始化

        #DomainLayer
        #1th-layer
        self.layer1_downconv = nn.Conv2d(in_channels = self.chanel_in,out_channels = self.chanel_in//self.r,kernel_size = 1)
        self.layer1_upconv = nn.Conv3d(in_channels = self.chanel_in//self.r , out_channels = self.chanel_in , kernel_size= 1,stride = 1,padding = 0,dilation = 1,groups = 1,bias = True)
        self.layer1_downconv.weight = torch.nn.Parameter(self.conv1.weight)
        self.layer1_upconv.weight = torch.nn.Parameter(self.value_conv.weight)
        #2th-layer
        self.layer2_downconv = nn.Conv2d(in_channels = self.chanel_in,out_channels = self.chanel_in//self.r,kernel_size = 1)
        self.layer2_upconv = nn.Conv3d(in_channels = self.chanel_in//self.r , out_channels = self.chanel_in , kernel_size= 1,stride = 1,padding = 0,dilation = 1,groups = 1,bias = True)
        self.layer2_downconv.weight = torch.nn.Parameter(self.layer2_downconv.weight + self.layer2_downgama*self.layer1_downconv.weight)
        self.layer2_upconv.weight = torch.nn.Parameter(self.layer2_upconv.weight + self.layer2_upgamma*self.layer1_upconv.weight)
        '''
        #3th-layer
        self.layer2_downconv = nn.Conv2d(in_channels = self.chanel_in,out_channels = self.chanel_in//self.r,kernel_size = 1)
        self.layer2_upconv = nn.Conv3d(in_channels = self.chanel_in//self.r , out_channels = self.chanel_in , kernel_size= 1,stride = 1,padding = 0,dilation = 1,groups = 1,bias = True)
        self.layer2_downconv.weight = torch.nn.Parameter(self.layer2_downconv.weight + self.layer2_downgama*self.layer1_downconv.weight)
        self.layer2_upconv.weight = torch.nn.Parameter(self.layer2_upconv.weight + self.layer2_upgamma*self.layer1_upconv.weight)
        '''


    def forward(self,x,domainX):
        #domainX->[BT,C,H,W]

        #[64,2048,7,7]
        #此处的x是不带位置编码的，并且没有经过自注意力运算，是最原始的输入x
        temp = x 
        '''如果是先加PE再处理C,残差的是带位置编码,没有经过学习的feature map'''
        '''此时，第二种方案就是先处理C，然后再添加PE，残差的是不带位置编码，没有经过学习的feature map，并且qkv单独编码'''

        #DomainLayer
        domainX = self.layer1_downconv(domainX) #[BT,C,H,W]

        domainX = create5Dimages(domainX) #[B,T,C,H,W]
        domainX = domainX.permute(0,2,1,3,4) #[4,256,16,7,7]   
        domainX = self.layer1_upconv(domainX) #[4,2048,16,7,7]
        domainX = domainX.permute(0,2,1,3,4) #[4,16,2048,7,7]
        domainX = create4Dimages(domainX) #[64,2048,7,7]



        domainX = self.layer2_downconv(domainX) 

        domainX = create5Dimages(domainX) #[B,T,C,H,W]
        domainX = domainX.permute(0,2,1,3,4) #[4,256,16,7,7]   

        domainX = self.layer2_upconv(domainX)
        domainX = domainX.permute(0,2,1,3,4) #[4,16,2048,7,7]
        domainX = create4Dimages(domainX) #[64,2048,7,7]

        #Q
        q = self.conv1(domainX) #[64,4096,7,7]  [64,512,7,7]  [64,1024,7,7] [64,4096,7,7]

        

       

        #print('q.shape = ',q.shape) #[64,4096,7,7]

        x1 = q[0:16,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height) #[1,16,256,7,7] [1,16,4096,7,7]
        x2 = q[16:32,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        x3 = q[32:48,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        x4 = q[48:64,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        q = torch.cat([x1,x2,x3,x4],dim = 0).permute(0,1,3,4,2) #[4,16,256,7,7]->[4,16,7,7,256]
        q = q.reshape(self.batch_size,self.num_frames*self.width*self.height,self.channels)
        #q.shape = [B,THW,256] = [4,784,256]
        #print(q.shape,'q')

        '''
        #DomainLayer
        domainX = self.layer1_downconv(domainX) #[BT,C,H,W]

        domainX = create5Dimages(domainX) #[B,T,C,H,W]
        domainX = domainX.permute(0,2,1,3,4) #[4,256,16,7,7]   
        domainX = self.layer1_upconv(domainX) #[4,2048,16,7,7]
        domainX = domainX.permute(0,2,1,3,4) #[4,16,2048,7,7]
        domainX = create4Dimages(domainX) #[64,2048,7,7]



        domainX = self.layer2_downconv(domainX) 

        domainX = create5Dimages(domainX) #[B,T,C,H,W]
        domainX = domainX.permute(0,2,1,3,4) #[4,256,16,7,7]   

        domainX = self.layer2_upconv(domainX)
        domainX = domainX.permute(0,2,1,3,4) #[4,16,2048,7,7]
        domainX = create4Dimages(domainX) #[64,2048,7,7]
        '''
        
        #K
        k = self.conv2(x)
 



        x1 = k[0:16,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height) #[1,16,256,7,7] [1,16,4096,7,7]
        x2 = k[16:32,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        x3 = k[32:48,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        x4 = k[48:64,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        k = torch.cat([x1,x2,x3,x4],dim = 0).permute(0,1,3,4,2) #[4,16,256,7,7]->[4,16,7,7,256]
        k = k.permute(0, 2, 1, 3, 4).reshape(self.batch_size, self.channels, self.num_frames * self.width * self.height)
        #k.shape = [B,256,THW] = [4,256,784]
        #print(k.shape,'k')
        
        #V
        v = self.conv3(x) #[64,256,7,7]




        x1 = v[0:16,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height) #[1,16,256,7,7] [1,16,4096,7,7]
        x2 = v[16:32,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        x3 = v[32:48,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        x4 = v[48:64,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        v = torch.cat([x1,x2,x3,x4],dim = 0).permute(0,1,3,4,2) #[4,16,256,7,7]->[4,16,7,7,256]
        v = v.reshape(self.batch_size,self.num_frames*self.width*self.height,self.channels)
        #q.shape = [B,THW,256] = [4,784,256]
        #print(v.shape,'er')
        
        # Q*K^T
        #print(q.shape)
        #print(k.shape)
        energy = torch.bmm(q,k) #energy.shape = [B,THW,THW] = [4,784,784]
        #softmax(Q*K^T)
        attention = self.softmax(energy) #attentio.shape = [B,THW,THW] = [4,784,784]
        #softmax(Q*K^T)*V
        out = torch.bmm(attention,v).view(self.batch_size,self.num_frames,self.width,self.height,self.channels).permute(0,4,1,2,3)
        #out.shape = [B,THW,256] = [4,784,256] -> [4,256,16,7,7]
        out = self.value_conv(out).permute(0,2,1,3,4) #[4,2048,16,7,7] -> [4,16,2048,7,7]
        out1 = out[0:1,:,:,:,:].view(self.num_frames,self.chanel_in,self.width,self.height) #[1,16,2048,7,7] [1,16,4096,7,7]
        out2 = out[1:2,:,:,:,:].view(self.num_frames,self.chanel_in,self.width,self.height)
        out3 = out[2:3,:,:,:,:].view(self.num_frames,self.chanel_in,self.width,self.height)
        out4 = out[3:4,:,:,:,:].view(self.num_frames,self.chanel_in,self.width,self.height)
        out = torch.cat([out1,out2,out3,out4],dim = 0) #[64,2048,7,7]
        out = self.gamma * out + temp


        return out