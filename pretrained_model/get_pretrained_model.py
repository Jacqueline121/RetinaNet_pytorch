import math
import torch
import torch.nn as nn
import torch.nn.init as init

from RetinaNet.FPN import FPN50
from RetinaNet.retinaNet import RetinaNet


def get_pretrained_model():
    print('loading pretrained ResNet50 model')
    R50 = torch.load('resnet50-caffe.pth')

    print('loading R50 into FPN50')
    fpn = FPN50()
    fpn50 = fpn.state_dict()
    for l in R50.keys():
        if not l.startswith('fc'):
            fpn50[l] = R50[l]

    print('saving RetinaNet')
    model = RetinaNet()
    for l in model.modules():
        if isinstance(l, nn.Conv2d):
            init.normal(l.weight, mean=0, std=0.01)
            if l.bias is not None:
                init.constant(l.bias, 0)
        elif isinstance(l, nn.BatchNorm2d):
            l.weight.data.fill_(1)
            l.bias.data.zero_()

    pi = 0.01
    init.constant(model.cls_head[-1].bias, -math.log((1 - pi) / pi))
    model.fpn.load_state_dict(fpn50)
    torch.save(model.state_dict(), 'model.pth')
    print('done')


if __name__ == '__main__':
    get_state_dict()