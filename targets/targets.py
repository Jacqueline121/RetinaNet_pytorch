import torch
from config.config import cfg
from targets.anchors import generate_anchors
from utils.box_operation import box_ious, xywh2xyxy, xyxy2xywh, box_transform, box_iou


def get_target(gt_boxes, label):
    '''

    :param gt_boxes: xyxy
    :param label:
    :return:
    '''

    anchors = generate_anchors()
    num_anchors = anchors.size(0)
    anchors_xyxy = xywh2xyxy(anchors)

    anchor_gt_iou = box_ious(anchors_xyxy, gt_boxes)
    iou_max, iou_argmax = torch.max(anchor_gt_iou, dim=1)

    # loc target
    gt_boxes_xywh = xyxy2xywh(gt_boxes)
    loc_targets = box_transform(anchors, gt_boxes_xywh[iou_argmax])

    # class label target
    cls_targets = torch.ones((num_anchors, cfg.CLASS_NUM)) * -1  # 先全部初始化为-1
    cls_targets[torch.lt(iou_max, 0.4), :] = 0  # 0.4以下的是0，代表背景类
    pos_idx = torch.ge(iou_max, 0.5)  # 正样本的index
    cls_targets[pos_idx, :] = 0
    assigned_cls = label[iou_argmax]
    cls_targets[pos_idx, assigned_cls[pos_idx].long()] = 1

    return cls_targets, loc_targets


if __name__ == '__main__':
    anchors = generate_anchors()
    cls, loc = get_target(torch.Tensor([[356, 32, 411, 72], [114, 36, 182, 86]]), torch.Tensor([[9],[1]]))
    print(loc[54084, :])
    print(cls[54084, :])

