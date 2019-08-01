import torch
import torch.nn as nn
from RetinaNet.FPN import FPN50


class RetinaNet(nn.Module):
    def __init__(self, class_num=20):
        super(RetinaNet, self).__init__()
        self.anchor_num = 9
        self.class_num = class_num
        self.fpn = FPN50()
        self.cls_head = self._make_heads(self.anchor_num * class_num)
        self.loc_head = self._make_heads(self.anchor_num * 4)
        self.out_act = nn.Sigmoid()

    def _make_heads(self, out_dim):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, out_dim, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        fms = self.fpn(x)
        cls_preds = []
        loc_preds = []
        for fm in fms:
            cls_pred = self.cls_head(fm)
            cls_pred = self.out_act(cls_pred)
            loc_pred = self.loc_head(fm)

            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.class_num)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

            cls_preds.append(cls_pred)
            loc_preds.append(loc_pred)

        return torch.cat(cls_preds, 1), torch.cat(loc_preds, 1)

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == '__main__':
    x = torch.randn([1, 3, 600, 600])
    net = RetinaNet()
    cls,loc = net(x)
