import torch
import torch.nn as nn
import ops.create5Dimages as create5D
import ops.create4Dimages as create4D


#新加模块
class Self_Attn(nn.Module):
    """Self attention Layer"""
    def __init__(self,in_dim,batch_size,num_frames,width,height,channels):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.r = 2
        self.conv1 = nn.Conv2d(in_channels = self.chanel_in,out_channels = self.chanel_in*self.r,kernel_size = 1)
        self.conv2 = nn.Conv2d(in_channels = self.chanel_in , out_channels = self.chanel_in*self.r , kernel_size= 1)
        self.conv3 = nn.Conv2d(in_channels = self.chanel_in,out_channels = self.chanel_in*self.r,kernel_size = 1)
        self.value_conv = nn.Conv3d(in_channels = self.chanel_in*self.r , out_channels = self.chanel_in , kernel_size= 1,stride = 1,padding = 0,dilation = 1,groups = 1,bias = True)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.channels = channels
        zero = torch.zeros(2048,4096,1,1,1)
        self.value_conv.weight = torch.nn.Parameter(zero)
        #self.value_conv.weight = torch.nn.Parameter(zero)
        self.pos_embedQ = nn.Parameter(torch.zeros(1,16,4096*7*7)) 
        self.pos_embedK = nn.Parameter(torch.zeros(1,16,4096*7*7)) 
        self.pos_embedV = nn.Parameter(torch.zeros(1,16,4096*7*7)) 

    def forward(self,x):
        #[64,2048,7,7] 
        #此处的x是不带位置编码的，并且没有经过自注意力运算，是最原始的输入x
        temp = x 
        '''如果是先加PE再处理C,残差的是带位置编码,没有经过学习的feature map'''
        '''此时，第二种方案就是先处理C，然后再添加PE，残差的是不带位置编码，没有经过学习的feature map，并且qkv单独编码'''



        #Q
        q = self.conv1(x) #[64,4096,7,7]  [64,512,7,7]  [64,1024,7,7] [64,4096,7,7]
        q = create5D.create5Dimages(q) #[4,16,4096,7,7]
        q = q.view(4,16,4096*7*7) #[4,16,4096*7*7]
        q = q+self.pos_embedQ
        q = q.view(4,16,4096,7,7)
        q = create4D.create4Dimages(q)
        
       

        #print('q.shape = ',q.shape) #[64,4096,7,7]

        x1 = q[0:16,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height) #[1,16,256,7,7] [1,16,4096,7,7]
        x2 = q[16:32,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        x3 = q[32:48,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        x4 = q[48:64,:,:,:].view(1,self.num_frames,self.channels,self.width,self.height)
        q = torch.cat([x1,x2,x3,x4],dim = 0).permute(0,1,3,4,2) #[4,16,256,7,7]->[4,16,7,7,256]
        q = q.reshape(self.batch_size,self.num_frames*self.width*self.height,self.channels)
        #q.shape = [B,THW,256] = [4,784,256]
        #print(q.shape,'q')

        
        #K
        k = self.conv2(x)
        k = create5D.create5Dimages(k) #[4,16,4096,7,7]
        k = k.view(4,16,4096*7*7)
        k = k+self.pos_embedK
        k = k.view(4,16,4096,7,7)
        k = create4D.create4Dimages(k)



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
        v = create5D.create5Dimages(v) #[4,16,4096,7,7]
        v = v.view(4,16,4096*7*7)
        v = v+self.pos_embedV
        v = v.view(4,16,4096,7,7)
        v = create4D.create4Dimages(v)




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