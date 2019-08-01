import os
import torch
import argparse
from data.VOCDataset import VOCDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from RetinaNet.retinaNet import RetinaNet
from loss.focal_loss import FocalLoss
from eval import eval


def parse_args():
    parser = argparse.ArgumentParser(description='RetinaNet Training')
    parser.add_argument('--dataset', dest='dataset', default='voc07trainval')
    parser.add_argument('--batch_size', dest='batch_size', default=1, type=int)
    parser.add_argument('--lr', dest='lr', default=0.001, type=float)
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', default=1e-4, type=float)
    parser.add_argument('--resume', dest='resume', default=None, action='store_true')
    parser.add_argument('--GPU', dest='use_GPU', default=True, type=bool)
    parser.add_argument('--mGPUs', dest='mGPUs', default=True, type=bool)
    parser.add_argument('--epochs', dest='epochs', default=10, type=int)
    parser.add_argument('--use_tfboard', dest='use_tfboard', default=True, type=bool)
    parser.add_argument('--output_dir', dest='output_dir', default='output')
    return parser.parse_args()


def test():
    args = parse_args()
    # data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = VOCDataset(transform=transform, train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model = RetinaNet()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # load trained model
    weight_file_path = os.path.join('output', 'retinanet_epoch_{}.pth'.format(0))

    if torch.cuda.is_available:
        state_dict = torch.load(weight_file_path)
    else:
        state_dict = torch.load(weight_file_path, map_location='cpu')

    model.load_state_dict(state_dict['model'])
    if args.use_GPU:
        model = model.cuda()
    model.eval()

    num_data = len(dataset)

    num_class = 20

    all_boxes = [[[] for _ in range(num_data)] for _ in range(num_class)]

    img_id = -1
    det_file = os.path.join(args.output_dir, 'detections.pkl')

    with torch.no_grad():
        for batch_size, (im_data, cls_targets, loc_targets, im_sizes) in enumerate(dataloader):

            if args.use_GPU:
                im_data = im_data.cuda()
                cls_targets = cls_targets.cuda()
                loc_targets = loc_targets.cuda()

            im_data = Variable(im_data)
            cls_preds, loc_preds = model(im_data)

            for i in range(args.batch_size):
                cls_pred = cls_preds[i, :, :]
                loc_pred = loc_preds[i, :, :]
                im_size = im_sizes[i, :]
            detections = eval([cls_pred, loc_pred, im_size])


if __name__ == '__main__':
    test()



