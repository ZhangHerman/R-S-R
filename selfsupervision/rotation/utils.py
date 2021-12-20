import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def rotation_task(rotation_classifier, features, labels_rotation):  #旋转任务
    """Applies the rotation prediction head to the given features.""" #利用给定的旋转之后的feature maps来进行rotation的预测（定义预测头部）
    scores = rotation_classifier(features) #根据送过来的features来预测得分
    assert scores.size(1) == 4  #判断最后得到的分数是否是4个(一列是一个feature的分数)
    loss = F.cross_entropy(scores, labels_rotation) #如果前面的assert没错，则由scores和labels
    return loss


def apply_2d_rotation(input_tensor,rotation):
    assert input_tensor.dim() >= 2
    height_dim = input_tensor.dim() - 2
    width_dim = height_dim + 1
    
    flip_upside_down = lambda x: torch.flip(x, dims=(height_dim,))
    flip_left_right = lambda x: torch.flip(x, dims=(width_dim,))
    spatial_transpose = lambda x: torch.transpose(x, height_dim, width_dim)
    if rotation == 0:  # 0 degrees rotation
        return input_tensor
    elif rotation == 90:  # 90 degrees rotation
        return flip_upside_down(spatial_transpose(input_tensor))
    elif rotation == 180:  # 90 degrees rotation
        return flip_left_right(flip_upside_down(input_tensor))
    elif rotation == 270:  # 270 degrees rotation / or -90
        return spatial_transpose(flip_upside_down(input_tensor))
    else:
        raise ValueError(
            "rotation should be 0, 90, 180, or 270 degrees; input value {}".format(rotation)
        )



def randomly_rotate_images(images):
    batch_size = images.size(0)
    labels_rot = torch.from_numpy(np.random.randint(0,4,size = batch_size))

    for r in range(4):
        mask = labels_rot == r
        images_masked = images[mask].contiguous()
        images[mask] = apply_2d_rotation(images_masked,rotation=r*90)

    return images,labels_rot


x = torch.arange(1.,64.*2048.*7.*7.+1).view(64,2048,7,7)
rot_images,rot_labels = randomly_rotate_images(x)
print(rot_images.shape)
print(type(rot_labels))

