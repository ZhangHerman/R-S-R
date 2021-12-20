import torch

#images = torch.arange(0,3*3*3).view(3,3,3)
#print(images)
#print(images[:,:,0])
'''a = torch.arange(0,9).view(3,3)
print(a)
print(a[:,0])
#实现交换2D图片的列(镜像翻转列)
index = [2,1,0]
print(a[:,index])'''

#核心代码
'''a = torch.arange(0,16*27).view(16,3,3,3)
print(a)
index = [2,1,0]
print(a[:,:,index])
a = list(range(0,16,1))
a.reverse()
print(a)
'''




def TmirrorZong(image):
    #image[T,C,H,W] W->L
    T,C,H,L = image.size()
    index = list(range(0,L,1))
    index.reverse()
    return  image[:,:,:,index]

def TmirrorHeng(image):
    #image[T,C,H,W] 
    T,C,H,W = image.size()
    index = list(range(0,H,1))
    index.reverse()
    return image[:,:,index]

def TmirrorZong3D(image):
    #image[C,H,W] W->L
    C,H,L = image.size()
    index = list(range(0,L,1))
    index.reverse()
    return  image[:,:,index]

def TmirrorHeng3D(image):
    #image[C,H,W] 
    C,H,W = image.size()
    index = list(range(0,H,1))
    index.reverse()
    return image[:,index]

a = torch.arange(0,3*3*3).view(3,3,3)
print(TmirrorZong3D(a))


