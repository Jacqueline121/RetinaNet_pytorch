import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_dim, out_dim, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Conv2d(out_dim, out_dim * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim * self.expansion)

        self.downsample = nn.Sequential()
        if in_dim != out_dim * self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        # bottom-up layers
        self.in_dim = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)

        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layers(self, block, out_dim, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_dim, out_dim, stride))
            self.in_dim = out_dim * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))  # (m/2, n/2, 64)
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)  # (m/4, n/4, 64)

        l2 = self.layer1(c1)  # (m/4, n/4, 256)
        l3 = self.layer2(l2)  # (m/8, n/8, 512)
        l4 = self.layer3(l3)  # (m/16, n/16, 1024)
        l5 = self.layer4(l4)  # (m/32, n/32, 2048)

        p6 = F.relu(self.conv6(l5))  # (m/64, n/64, 256)
        p7 = self.conv7(p6)  # (m/128, n/128, 256)

        p5 = self.latlayer1(l5)
        p4 = self._upsample_add(p5, self.latlayer2(l4))
        p4 = self.toplayer1(p4)

        p3 = self._upsample_add(p4, self.latlayer3(l3))
        p3 = self.toplayer2(p3)

        return p3, p4, p5, p6, p7


def FPN50():
    return FPN(Bottleneck, [3, 4, 6, 3])


def FPN100():
    return FPN(Bottleneck, [2, 4, 23, 3])


if __name__ == '__main__':
    net = FPN50()
    fms = net(torch.randn(1, 3, 600, 300))
    for fm in fms:
        print(fm.size())


