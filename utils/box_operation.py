import torch


def box_ious(boxes1, boxes2):

    N = boxes1.size(0)
    K = boxes2.size(0)

    area1 = ((boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)).view(N, 1)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1).view(1, K)

    ix1 = torch.max(boxes1[:, 0].view(N, 1), boxes2[:, 0].view(1, K))
    iy1 = torch.max(boxes1[:, 1].view(N, 1), boxes2[:, 1].view(1, K))
    ix2 = torch.min(boxes1[:, 2].view(N, 1), boxes2[:, 2].view(1, K))
    iy2 = torch.min(boxes1[:, 3].view(N, 1), boxes2[:, 3].view(1, K))

    iw = torch.max(ix2 - ix1 + 1, boxes1.new(1).fill_(0))
    ih = torch.max(iy2 - iy1 + 1, boxes1.new(1).fill_(0))

    inter_area = iw * ih
    union_area = area1 + area2 - inter_area

    IoU = inter_area / union_area

    return IoU


def box_transform(box, target_box):
    '''

    :param box: xywh
    :param target_box: xywh
    :return:
    '''
    t_x = (target_box[:, 0] - box[:, 0]) / box[:, 2]
    t_y = (target_box[:, 1] - box[:, 1]) / box[:, 3]
    t_w = torch.log(target_box[:, 2] / box[:, 2])
    t_h = torch.log(target_box[:, 3] / box[:, 3])

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    return torch.cat([t_x, t_y, t_w, t_h], dim=1)


def box_transform_inv(boxes, delta):

    p_x = delta[:, 0] * boxes[:, 2] + boxes[:, 0]
    p_y = delta[:, 1] * boxes[:, 3] + boxes[:, 1]
    p_w = torch.exp(delta[:, 2]) * boxes[:, 2]
    p_h = torch.exp(delta[:, 3]) * boxes[:, 3]

    p_x = p_x.view(-1, 1)
    p_y = p_y.view(-1, 1)
    p_w = p_w.view(-1, 1)
    p_h = p_h.view(-1, 1)

    pred_boxes = torch.cat([p_x, p_y, p_w, p_h], dim=1)

    return pred_boxes


def xyxy2xywh(box):
    w = box[:, 2] - box[:, 0] + 1
    h = box[:, 3] - box[:, 1] + 1
    ctr_x = (box[:, 0] + box[:, 2]) / 2
    ctr_y = (box[:, 1] + box[:, 3]) / 2

    ctr_x = ctr_x.view(-1, 1)
    ctr_y = ctr_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([ctr_x, ctr_y, w, h], dim=1)
    return xywh_box


def xywh2xyxy(box):
    x1 = box[:, 0] - box[:, 2] / 2
    y1 = box[:, 1] - box[:, 3] / 2
    x2 = box[:, 0] + box[:, 2] / 2
    y2 = box[:, 1] + box[:, 3] / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xyxy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xyxy_box


def box_iou(box1, box2, order='xyxy'):
    '''Compute the intersection over union of two set of boxes.

    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou


if __name__ == '__main__':
    box1 = torch.Tensor([[356, 42, 411, 72], [500, 60, 700, 100]])
    box2 = torch.Tensor([[400, 50, 600, 80]])
    print(box_ious(box1, box2))
    print(box_iou(box1, box2))