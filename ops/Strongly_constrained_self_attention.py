import torch
import torch.nn as nn
import ops.create5Dimages as create5D
import ops.create4Dimages as create4D


def create5Dimages(images):  # [BT,C,H,W]->[B,T,C,H,W]
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


def create4Dimages(images):  # [B,T,C,H,W]->[BT,C,H,W]
    # images : 5D tensor with shape [B,T,C,H,W]
    B, T, C, H, W = images.size()
    image = torch.tensor([]).cuda()
    for b in range(B):
        image = torch.cat([image, images[b]], dim=0)
    return image


# 新加模块
class SCSAttention(nn.Module):


    def __init__(self, in_dim, batch_size, num_frames, width, height, channels):
        super(SCSAttention, self).__init__()
        self.chanel_in = in_dim
        self.r = 8
        self.conv1_1_pre = nn.Conv2d(in_channels=1, out_channels=2048, kernel_size=1)
        self.conv1_1 = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // (2*self.r), kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // (2*self.r), kernel_size=1)
        #self.conv2 = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // self.r, kernel_size=1)
        '''通道压缩倍数×2,目的是进行Concat操作之后维持原有的通道压缩倍数'''
        self.conv2_1 = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // (2*self.r), kernel_size=1)
        self.conv2_2 = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // (2*self.r), kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // self.r, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=self.chanel_in // self.r, out_channels=self.chanel_in, kernel_size=1,
                                    stride=1, padding=0, dilation=1, groups=1, bias=True)
        a = torch.tensor(0.01)
        a.cuda()
        a.requires_grad = True

        #self.pos_embedQ = nn.Parameter(torch.zeros(1,16,256*7*7)) 
        #self.pos_embedK = nn.Parameter(torch.zeros(1,16,256*7*7)) 
        self.pos_embedV = nn.Parameter(torch.randn(1, 8, (in_dim//8)*7*7))
        self.pos_embedV_dropout = 0.
        self.dropout = nn.Dropout(self.pos_embedV_dropout)
        '''a.requires_grad = True'''
        self.gamma = nn.Parameter(a)

        self.softmax = nn.Softmax(dim=-1)
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.channels = channels
        self.layer2_downgama = nn.Parameter(torch.zeros(1))
        self.layer2_upgamma = nn.Parameter(torch.zeros(1))
        self.layer2_downgama_Y = nn.Parameter(torch.zeros(1))
        self.layer2_upgamma_Y = nn.Parameter(torch.zeros(1))
        # 最后一层的1×1×1卷积核的初始化权重没有被全0初始化
        # DomainLayerX
        # 1th-layer
        self.layer1_downconv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // self.r,
                                         kernel_size=1)
        self.layer1_upconv = nn.Conv3d(in_channels=self.chanel_in // self.r, out_channels=self.chanel_in, kernel_size=1,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.layer1_downconv.weight = torch.nn.Parameter(self.conv3.weight)
        self.layer1_upconv.weight = torch.nn.Parameter(self.value_conv.weight)
        # 2th-layer
        self.layer2_downconv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // self.r,
                                         kernel_size=1)
        self.layer2_upconv = nn.Conv3d(in_channels=self.chanel_in // self.r, out_channels=self.chanel_in, kernel_size=1,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.layer2_downconv.weight = torch.nn.Parameter(
            self.layer2_downconv.weight + self.layer2_downgama * self.layer1_downconv.weight)
        self.layer2_upconv.weight = torch.nn.Parameter(
            self.layer2_upconv.weight + self.layer2_upgamma * self.layer1_upconv.weight)
        # DomainLayerY
        # 1th-layer
        self.layer1_downconv_Y = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // self.r,
                                         kernel_size=1)
        self.layer1_upconv_Y = nn.Conv3d(in_channels=self.chanel_in // self.r, out_channels=self.chanel_in, kernel_size=1,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.layer1_downconv_Y.weight = torch.nn.Parameter(self.conv3.weight)
        self.layer1_upconv_Y.weight = torch.nn.Parameter(self.value_conv.weight)
        # 2th-layer
        self.layer2_downconv_Y = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // self.r,
                                         kernel_size=1)
        self.layer2_upconv_Y = nn.Conv3d(in_channels=self.chanel_in // self.r, out_channels=self.chanel_in, kernel_size=1,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.layer2_downconv_Y.weight = torch.nn.Parameter(
            self.layer2_downconv_Y.weight + self.layer2_downgama_Y * self.layer1_downconv_Y.weight)
        self.layer2_upconv_Y.weight = torch.nn.Parameter(
            self.layer2_upconv_Y.weight + self.layer2_upgamma_Y * self.layer1_upconv_Y.weight)
        '''
        #3th-layer
        self.layer2_downconv = nn.Conv2d(in_channels = self.chanel_in,out_channels = self.chanel_in//self.r,kernel_size = 1)
        self.layer2_upconv = nn.Conv3d(in_channels = self.chanel_in//self.r , out_channels = self.chanel_in , kernel_size= 1,stride = 1,padding = 0,dilation = 1,groups = 1,bias = True)
        self.layer2_downconv.weight = torch.nn.Parameter(self.layer2_downconv.weight + self.layer2_downgama*self.layer1_downconv.weight)
        self.layer2_upconv.weight = torch.nn.Parameter(self.layer2_upconv.weight + self.layer2_upgamma*self.layer1_upconv.weight)
        '''

    def forward(self, x, domainX,domainY):
        # domainX->[BT,C,H,W]
        # domainY->[BT,C,H,W]
        # domainX <=> RBRF.image
        # domainY <=> RBRC.image

        # [64,2048,7,7]
        # 此处的x是不带位置编码的，并且没有经过自注意力运算，是最原始的输入x
        temp = x

        '''如果是先加PE再处理C,残差的是带位置编码,没有经过学习的feature map'''
        '''此时，第二种方案就是先处理C，然后再添加PE，残差的是不带位置编码，没有经过学习的feature map，并且qkv单独编码'''


        # DomainLayer
        domainY = self.conv1_1_pre(domainY)
        #print(domainY.shape)
        
        domainY = self.layer1_downconv_Y(domainX)  # [BT,C,H,W]

        domainY = create5Dimages(domainY)  # [B,T,C,H,W]
        domainY = domainY.permute(0, 2, 1, 3, 4)  # [4,256,16,7,7]
        domainY = self.layer1_upconv_Y(domainY)  # [4,2048,16,7,7]
        domainY = domainY.permute(0, 2, 1, 3, 4)  # [4,16,2048,7,7]
        domainY = create4Dimages(domainY)  # [64,2048,7,7]

        domainY = self.layer2_downconv_Y(domainY)

        domainY = create5Dimages(domainY)  # [B,T,C,H,W]
        domainY = domainY.permute(0, 2, 1, 3, 4)  # [4,256,16,7,7]

        domainY = self.layer2_upconv_Y(domainY)

        domainY = domainY.permute(0, 2, 1, 3, 4)  # [4,16,2048,7,7]
        domainY = create4Dimages(domainY)  # [64,2048,7,7]

        QuerrySource = self.conv1_1(x) #[64,128,7,7] #[BT,C/(r*2),H,W]
        QuerryRBRC = self.conv1_2(domainY)#[64,128,7,7] #[BT,C/(r*2),H,W]

        '''Key'''
        q = torch.cat([QuerrySource,QuerryRBRC],dim=1) #[BT,C/r,H,W]

        # Q
        #q = self.conv1(x)  # [64,4096,7,7]  [64,512,7,7]  [64,1024,7,7] [64,4096,7,7]

        # print('q.shape = ',q.shape) #[64,4096,7,7]

        x1 = q[0:8, :, :, :].view(1, self.num_frames, self.channels, self.width,
                                   self.height)  # [1,16,256,7,7] [1,16,4096,7,7]
        x2 = q[8:16, :, :, :].view(1, self.num_frames, self.channels, self.width, self.height)
        x3 = q[16:24, :, :, :].view(1, self.num_frames, self.channels, self.width, self.height)
        x4 = q[24:32, :, :, :].view(1, self.num_frames, self.channels, self.width, self.height)
        q = torch.cat([x1, x2, x3, x4], dim=0).permute(0, 1, 3, 4, 2)  # [4,16,256,7,7]->[4,16,7,7,256]
        q = q.reshape(self.batch_size, self.num_frames * self.width * self.height, self.channels)
        # q.shape = [B,THW,256] = [4,784,256]
        # print(q.shape,'q')





        # DomainLayer
        domainX = self.layer1_downconv(domainX)  # [BT,C,H,W]

        domainX = create5Dimages(domainX)  # [B,T,C,H,W]
        domainX = domainX.permute(0, 2, 1, 3, 4)  # [4,256,16,7,7]
        domainX = self.layer1_upconv(domainX)  # [4,2048,16,7,7]
        domainX = domainX.permute(0, 2, 1, 3, 4)  # [4,16,2048,7,7]
        domainX = create4Dimages(domainX)  # [64,2048,7,7]

        domainX = self.layer2_downconv(domainX)

        domainX = create5Dimages(domainX)  # [B,T,C,H,W]
        domainX = domainX.permute(0, 2, 1, 3, 4)  # [4,256,16,7,7]

        domainX = self.layer2_upconv(domainX)
        domainX = domainX.permute(0, 2, 1, 3, 4)  # [4,16,2048,7,7]
        domainX = create4Dimages(domainX)  # [64,2048,7,7]

        KeySource = self.conv2_1(x) #[64,128,7,7] #[BT,C/(r*2),H,W]
        KeyRBRF = self.conv2_2(domainX)#[64,128,7,7] #[BT,C/(r*2),H,W]

        '''Key'''
        k = torch.cat([KeySource,KeyRBRF],dim=1) #[BT,C/r,H,W]

        x1 = k[0:8, :, :, :].view(1, self.num_frames, self.channels, self.width,
                                   self.height)  # [1,16,256,7,7] [1,16,4096,7,7]
        x2 = k[8:16, :, :, :].view(1, self.num_frames, self.channels, self.width, self.height)
        x3 = k[16:24, :, :, :].view(1, self.num_frames, self.channels, self.width, self.height)
        x4 = k[24:32, :, :, :].view(1, self.num_frames, self.channels, self.width, self.height)
        k = torch.cat([x1, x2, x3, x4], dim=0).permute(0, 1, 3, 4, 2)  # [4,16,256,7,7]->[4,16,7,7,256]
        k = k.permute(0, 2, 1, 3, 4).reshape(self.batch_size, self.channels, self.num_frames * self.width * self.height)
        # k.shape = [B,256,THW] = [4,256,784]
        # print(k.shape,'k')

        # V
        v = self.conv3(x)  # [64,256,7,7]
        v = create5Dimages(v)
        v = v.reshape(4,8,256*7*7)
        v = v + self.pos_embedV
        v = self.dropout(v)
        v = v.view(4,8,256,7,7)
        v = create4Dimages(v)


        x1 = v[0:8, :, :, :].view(1, self.num_frames, self.channels, self.width,
                                   self.height)  # [1,16,256,7,7] [1,16,4096,7,7]
        x2 = v[8:16, :, :, :].view(1, self.num_frames, self.channels, self.width, self.height)
        x3 = v[16:24, :, :, :].view(1, self.num_frames, self.channels, self.width, self.height)
        x4 = v[24:32, :, :, :].view(1, self.num_frames, self.channels, self.width, self.height)
        v = torch.cat([x1, x2, x3, x4], dim=0).permute(0, 1, 3, 4, 2)  # [4,16,256,7,7]->[4,16,7,7,256]
        v = v.reshape(self.batch_size, self.num_frames * self.width * self.height, self.channels)
        # q.shape = [B,THW,256] = [4,784,256]
        # print(v.shape,'er')

        # Q*K^T
        # print(q.shape)
        # print(k.shape)
        energy = torch.bmm(q, k)  # energy.shape = [B,THW,THW] = [4,784,784]
        # softmax(Q*K^T)
        attention = self.softmax(energy)  # attentio.shape = [B,THW,THW] = [4,784,784]
        # softmax(Q*K^T)*V
        out = torch.bmm(attention, v).view(self.batch_size, self.num_frames, self.width, self.height,
                                           self.channels).permute(0, 4, 1, 2, 3)
        # out.shape = [B,THW,256] = [4,784,256] -> [4,256,16,7,7]
        out = self.value_conv(out).permute(0, 2, 1, 3, 4)  # [4,2048,16,7,7] -> [4,16,2048,7,7]
        out1 = out[0:1, :, :, :, :].view(self.num_frames, self.chanel_in, self.width,
                                         self.height)  # [1,16,2048,7,7] [1,16,4096,7,7]
        out2 = out[1:2, :, :, :, :].view(self.num_frames, self.chanel_in, self.width, self.height)
        out3 = out[2:3, :, :, :, :].view(self.num_frames, self.chanel_in, self.width, self.height)
        out4 = out[3:4, :, :, :, :].view(self.num_frames, self.chanel_in, self.width, self.height)
        out = torch.cat([out1, out2, out3, out4], dim=0)  # [64,2048,7,7]
        '''out = self.gamma * out  + temp'''
        out = ((self.gamma*out+temp)*out)+temp

        #print('self.gamma==',self.gamma)
        #print('before=',self.pos_embedV[:,0,:])
        #print('pos_embedV.grad=',self.pos_embedV)
        #if(self.pos_embedV.grad is not None):
            #self.pos_embedV.data=self.pos_embedV.grad
            #print('after=',self.pos_embedV[:,0,:])

        return out