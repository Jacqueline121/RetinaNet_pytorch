import math
import torch
from config.config import cfg


def get_anchor_wh():
    anchor_sizes = torch.Tensor(cfg.ANCHOR_SIZES)
    aspect_ratios = torch.Tensor(cfg.ASPECT_RATIOS)
    scale_ratios = torch.Tensor(cfg.SCALE_RATIOS)
    anchor_wh = []
    for size in anchor_sizes:
        area = size * size
        for ar in aspect_ratios:
            h = torch.sqrt(area / ar)
            w = h * ar
            for sr in scale_ratios:
                anchor_w = w * sr
                anchor_h = h * sr
                anchor_wh.append([anchor_w, anchor_h])
    num_fms = len(anchor_sizes)
    return torch.Tensor(anchor_wh).view(num_fms, -1, 2)


def generate_anchors():
    image_size = cfg.INPUT_SIZE
    anchor_wh = get_anchor_wh()
    num_fms, _, _ = anchor_wh.size()

    fm_ws = [math.ceil(image_size[0] / pow(2, i+3)) for i in range(num_fms)]
    fm_hs = [math.ceil(image_size[1] / pow(2, i+3)) for i in range(num_fms)]

    all_anchors = []

    for i in range(num_fms):
        fm_w, fm_h = int(fm_ws[i]), int(fm_hs[i])

        A = anchor_wh[i].size(0)
        K = fm_w * fm_h

        grid_size = image_size[0] / fm_w

        shift_x = torch.arange(0, fm_w)
        shift_y = torch.arange(0, fm_h)

        shifts_x, shifts_y = torch.meshgrid([shift_x, shift_y])

        shifts_x = (shifts_x + 0.5) * grid_size
        shifts_y = (shifts_y + 0.5) * grid_size

        ctrs = torch.cat([shifts_x.view(-1, 1), shifts_y.view(-1, 1)], dim=-1).float()

        anchor = torch.cat([ctrs.view(K, 1, 2).expand(K, A, 2),
                            anchor_wh[i].view(1, A, 2).expand(K, A, 2)], dim=-1)

        all_anchors.append(anchor.view(-1, 4))

    return torch.cat(all_anchors, dim=0)


if __name__ == '__main__':
    anchors = generate_anchors()
    print(anchors.size())
    print(anchors[4500:4600, :])

