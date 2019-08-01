import torch
from config.config import cfg
from target.anchors import generate_anchors
from utils.box_operation import box_transform_inv, xywh2xyxy, box_ious


def get_pred_boxes(loc_preds):
    anchors = generate_anchors().view(-1, 4)
    pred_boxes = box_transform_inv(anchors, loc_preds)
    return pred_boxes


def filter_boxes(cls_preds, pred_boxes, cls_thresh):
    '''
    filter boxes whose cls_pred is less than cls_thresh
    :param cls_preds: [n, class_num]
    :param pred_boxes: [n, 4]
    :param cls_thresh: 0.5
    :return:
    '''

    max_cls, _ = torch.max(cls_preds, dim=-1)
    keep = (max_cls > cls_thresh).view(-1)

    cls_preds = cls_preds[keep]
    pred_boxes = pred_boxes[keep]

    return cls_preds, pred_boxes


def rescale_boxes(pred_boxes, im_sizes):
    '''

    :param pred_boxes: xywh
    :return:
    '''
    image_w, image_h = im_sizes[0], im_sizes[1]
    input_w, input_h = cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1]

    scale_w = input_w / image_w
    scale_h = input_h / image_h

    pred_boxes[:, 0::2] /= scale_w
    pred_boxes[:, 1::2] /= scale_h

    pred_boxes = xywh2xyxy(pred_boxes)

    pred_boxes[:, 0::2].clamp_(0, image_w - 1)
    pred_boxes[:, 1::2].clamp_(0, image_h - 1)

    return pred_boxes


def nms(pred_boxes, cls_max, nms_thresh):
    order = torch.sort(cls_max, dim=0, descending=True)[1]
    keep = []

    while order.numel() > 0:
        keep.append(order[0])

        if order.numel() == 1:
            break

        cur_box = pred_boxes[order[0], :].view(-1, 4)
        res_box = pred_boxes[order[1:], :].view(-1, 4)

        ious = box_ious(cur_box, res_box).view(-1)

        idx = torch.nonzero(ious < nms_thresh).squeeze()
        order = order[idx + 1].view(-1)

    return torch.LongTensor(keep)

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


def eval(output, cls_thresh=0.5, nms_thresh=0.05):

    cls_preds = output[0].view(-1, cfg.NUM_CLASS).cpu()

    # testing
    #cls_preds = output[0].view(-1)
    #cls_preds = one_hot_embedding(cls_preds, 20).view(-1, 20)

    loc_preds = output[1].view(-1, 4).cpu()

    im_sizes = output[2].cpu()

    # 1. get predicted boxes
    pred_boxes = get_pred_boxes(loc_preds)

    # 2. filter boxes whose cls_score is less than cls_thresh
    cls_preds, pred_boxes = filter_boxes(cls_preds, pred_boxes, cls_thresh)

    # 3. rescale the pred_boxes
    pred_boxes = rescale_boxes(pred_boxes, im_sizes) # xyxy

    # 4. nms
    detections = []
    cls_max, cls_argmax = torch.max(cls_preds, dim=1, keepdim=True)

    for i in range(1, cfg.NUM_CLASS):
        idx = torch.nonzero(cls_argmax == i).squeeze()

        if idx.numel() == 0:
            continue

        pred_boxes_cls = pred_boxes[idx, :].view(-1, 4)
        cls_max_cls = cls_max[idx, :].view(-1, 1)
        cls_argmax_cls = cls_argmax[idx, :].view(-1, 1)

        keep = nms(pred_boxes_cls, cls_max_cls, nms_thresh)

        pred_boxes_cls = pred_boxes_cls[keep, :]
        cls_max_cls = cls_max_cls[keep, :]
        cls_label = cls_argmax_cls[keep, :].float()

        detections_cls = torch.cat([pred_boxes_cls, cls_max_cls, cls_label], dim=-1)

        detections.append(detections_cls)

    return torch.cat(detections, dim=0)



