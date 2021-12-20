import torch
import torch.nn
import torch.nn.functional as F


def rotation_task(rotation_classifier, features, labels_rotation):  #旋转任务
    """Applies the rotation prediction head to the given features.""" #利用给定的旋转之后的feature maps来进行rotation的预测（定义预测头部）
    scores = rotation_classifier(features) #根据送过来的features来预测得分
    assert scores.size(1) == 4  #判断最后得到的分数是否是4个(一列是一个feature的分数)
    loss = F.cross_entropy(scores, labels_rotation) #如果前面的assert没错，则由scores和labels
    return loss