import os
import torch
import argparse
from data.VOCDataset import VOCDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from RetinaNet.retinaNet import RetinaNet
from eval import eval
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from utils.visualize import draw_detection_boxes
from config.config import cfg
from data.evaluate_detections import evaluate_detections


def parse_args():
    parser = argparse.ArgumentParser(description='RetinaNet Training')
    parser.add_argument('--dataset', dest='dataset', default='VOC07')
    parser.add_argument('--batch_size', dest='batch_size', default=4, type=int)
    parser.add_argument('--GPU', dest='use_GPU', default=True, type=bool)
    parser.add_argument('--mGPUs', dest='mGPUs', default=True, type=bool)
    parser.add_argument('--output_dir', dest='output_dir', default='output')
    parser.add_argument('--vis', dest='vis', default=False)
    return parser.parse_args()


def test():
    args = parse_args()
    # data loader
    print('load data')
    cfg.DATASET_NAME = args.dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = VOCDataset(transform=transform, train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model = RetinaNet()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # load trained model
    weight_file_path = os.path.join('output', 'retinanet_epoch_{}.pth'.format(100))

    if torch.cuda.is_available:
        state_dict = torch.load(weight_file_path)
    else:
        state_dict = torch.load(weight_file_path, map_location='cpu')

    model.load_state_dict(state_dict['model'])
    if args.use_GPU:
        model = model.cuda()
    model.eval()

    num_data = len(dataset)
    class_num = cfg.CLASS_NUM

    all_boxes = [[[] for _ in range(num_data)] for _ in range(class_num)]

    img_id = -1
    det_file = os.path.join(args.output_dir, 'detections.pkl')

    with torch.no_grad():
        for batch_size, (im_data, cls_targets, loc_targets, im_infos) in enumerate(dataloader):

            if args.use_GPU:
                im_data = im_data.cuda()

            im_data = Variable(im_data)
            cls_preds, loc_preds = model(im_data)

            for i in range(im_data.size(0)):
                img_id += 1
                im_info = im_infos[i]
                cls_pred = cls_preds[i, :, :]
                loc_pred = loc_preds[i, :, :]
                im_size = im_info[1:]
                detections = eval([cls_pred, loc_pred, im_size])
                if args.vis:
                    im_name = '00' + format(str(int(im_info[0].data)), '0>4s')
                    im_path = os.path.join('./data/dataset/voc07_JPEGImages', im_name + '.jpg')
                    img_show = Image.open(im_path)

                if len(detections) > 0:
                    for cls in range(class_num):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 5))
                            cls_det[:, :4] = detections[inds, :4]
                            cls_det[:, 4] = detections[inds, 4]
                            all_boxes[cls][img_id] = cls_det.cpu().numpy()
                            if args.vis:
                                cls_name = detections[inds, -1].cpu().numpy()
                                img_show = draw_detection_boxes(img_show, cls_det.cpu().numpy(), cls_name)

                    if args.vis:
                        plt.imshow(img_show)
                        plt.show()
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    evaluate_detections(all_boxes, args.output_dir)


if __name__ == '__main__':
    test()



