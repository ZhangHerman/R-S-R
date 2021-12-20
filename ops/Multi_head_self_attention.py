import torch
import torch.nn as nn
import ops.create5Dimages as create5D
import ops.create4Dimages as create4D

import ops.Head1 as head1attention
import ops.Head2 as head2attention
import ops.Head3 as head3attention
import ops.Head4 as head4attention




def create5Dimages(images,T):  # [BT,C,H,W]->[B,T,C,H,W]
    # images : 4D tensor with shape [BT,C,H,W]
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
class Self_Attn(nn.Module):
    """Self attention Layer"""

    # (2048,4,16,7,7,4096)  （2048,4,16,7,7,256）
    # 调用：self.Attention = attention.Self_Attn(2048,4,16,7,7,256)

    def __init__(self, in_dim, batch_size, num_frames, width, height, channels):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.r = 8
        self.value_conv = nn.Conv3d(in_channels=self.chanel_in // self.r, out_channels=self.chanel_in, kernel_size=1,
                                    stride=1, padding=0, dilation=1, groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.channels = channels

        self.h1 = head1attention.Self_Attn(2048, 4, 4, 7, 7, 256)
        self.h2 = head2attention.Self_Attn(2048, 4, 4, 7, 7, 256)
        self.h3 = head3attention.Self_Attn(2048, 4, 4, 7, 7, 256)
        self.h4 = head4attention.Self_Attn(2048, 4, 4, 7, 7, 256)




    def forward(self, x):
        # [64,2048,7,7]
        # 此处的x是不带位置编码的，并且没有经过自注意力运算，是最原始的输入x
        temp = x
        x5d = create5Dimages(x,16) #[B,T,C,H,W]

        '''[B,T/head_number,C,H,W]'''
        HEAD1 = torch.cat([x5d[0, 0:4, :, :, :].view(1,self.num_frames  // 4,self.chanel_in,self.height,self.width),
                           x5d[1, 0:4, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           x5d[2, 0:4, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           x5d[3, 0:4, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           ])
        '''[B(T/head_number),C,H,W]'''
        #print(HEAD1.shape,'[][][][]')#[4,4,2048,7,7]
        HEAD1 = create4Dimages(HEAD1)
        #print(HEAD1.shape,'[][][][][][]')#[16,2048,7,7]
        HEAD2 = torch.cat([x5d[0, 4:8, :, :, :].view(1,self.num_frames//4,self.chanel_in,self.height,self.width),
                           x5d[1, 4:8, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           x5d[2, 4:8, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           x5d[3, 4:8, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           ])
        #print(HEAD2.shape,'[][][][][][]')#[4,4,2048,7,7]
        HEAD2 = create4Dimages(HEAD2)
        #print(HEAD2.shape,'[][][][]')#[16,2048,7,7]
        HEAD3 = torch.cat([x5d[0, 8:12, :, :, :].view(1,self.num_frames  // 4,self.chanel_in,self.height,self.width),
                           x5d[1, 8:12, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           x5d[2, 8:12, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           x5d[3, 8:12, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           ])
        #print(HEAD3.shape,'[][][][][][]')#[4,4,2048,7,7]
        HEAD3 = create4Dimages(HEAD3)
        #print(HEAD3.shape,'[][][][]')#[16,2048,7,7]
        HEAD4 = torch.cat([x5d[0, 12:16, :, :, :].view(1,self.num_frames  // 4,self.chanel_in,self.height,self.width),
                           x5d[1, 12:16, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           x5d[2, 12:16, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           x5d[3, 12:16, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                           ])
        #print(HEAD4.shape,'[][][][][][]')#[4,4,2048,7,7]
        HEAD4 = create4Dimages(HEAD4)
        #print(HEAD4.shape,'[][][][]')#[16,2048,7,7]

        
        HEAD1 = self.h1(HEAD1)#[B(T/head_number),C,H,W]
        #print(HEAD1.shape,'HEAD1') #[16,2048,7,7]
        
        HEAD1 = create5Dimages(HEAD1,4)#[B,(T/head_number),C,H,W]
        #print(HEAD1.shape,'HEAD1')
        

        HEAD2 = self.h2(HEAD2)#[B(T/head_number),C,H,W]
        HEAD2 = create5Dimages(HEAD2,4)#[B,(T/head_number),C,H,W]

        HEAD3 = self.h3(HEAD3)#[B(T/head_number),C,H,W]
        HEAD3 = create5Dimages(HEAD3,4)#[B,(T/head_number),C,H,W]

        HEAD4 = self.h4(HEAD4)#[B(T/head_number),C,H,W]
        HEAD4 = create5Dimages(HEAD4,4)#[B,(T/head_number),C,H,W]

        '''[1,T,C,H,W]'''
        #print(HEAD1.shape)#[1,16,2048,7,7]
        #print(HEAD2.shape)#[1,16,2048,7,7]
        #print(HEAD3.shape)#[1,16,2048,7,7]
        #print(HEAD4.shape)#[1,16,2048,7,7]
        
        BATCH1 = torch.cat([HEAD1[0, :, :, :, :].view(1,self.num_frames  // 4,self.chanel_in,self.height,self.width),
                            HEAD2[0, :, :, :, :].view(1,self.num_frames  // 4,self.chanel_in,self.height,self.width),
                            HEAD3[0, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            HEAD4[0, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            ],dim=1)


        BATCH2 = torch.cat([HEAD1[1, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            HEAD2[1, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            HEAD3[1, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            HEAD4[1, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            ], dim=1)


        BATCH3 = torch.cat([HEAD1[2, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            HEAD2[2, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            HEAD3[2, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            HEAD4[2, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            ], dim=1)


        BATCH4 = torch.cat([HEAD1[3, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            HEAD2[3, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            HEAD3[3, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            HEAD4[3, :, :, :, :].view(1, self.num_frames // 4, self.chanel_in, self.height, self.width),
                            ], dim=1)

        MTA = torch.cat([BATCH1,BATCH2,BATCH3,BATCH4],dim=0)#[B,T,C,H,W]
        MTA = create4Dimages(MTA)#[BT,C,H,W]
        #print(MTA.shape,'MTA')
        

        return MTA













