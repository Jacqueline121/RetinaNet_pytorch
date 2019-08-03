from __future__ import division

import os
import torch
import argparse
from data.VOCDataset import VOCDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from RetinaNet.retinaNet import RetinaNet
from tensorboardX import SummaryWriter
from loss.focal_loss import FocalLoss
from config.config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description='RetinaNet Training')
    parser.add_argument('--dataset', dest='dataset', default='VOC07')
    parser.add_argument('--batch_size', dest='batch_size', default=4, type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=2, type=int)
    parser.add_argument('--lr', dest='lr', default=0.001, type=float)
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', default=1e-4, type=float)
    parser.add_argument('--GPU', dest='use_GPU', default=True, type=bool)
    parser.add_argument('--mGPUs', dest='mGPUs', default=True, type=bool)
    parser.add_argument('--epochs', dest='epochs', default=100, type=int)
    parser.add_argument('--display_interval', dest='display_interval', default=10, type=int)
    parser.add_argument('--use_tfboard', dest='use_tfboard', default=True, type=bool)
    parser.add_argument('--output_dir', dest='output_dir', default='output')
    parser.add_argument('--save_interval', dest='save_interval', default=10)
    return parser.parse_args()


def train():
    args = parse_args()

    if args.use_tfboard:
        writer = SummaryWriter()

    # data loader
    print('load data')
    cfg.DATASET_NAME = args.dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = VOCDataset(transform=transform, train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=dataset.collate_fn)

    iter_per_epoch = int(len(dataset) / args.batch_size)

    # load model
    print('load model')
    model = RetinaNet()
    model.load_state_dict(torch.load('./pretrained_model/model.pth'))
    model.freeze_bn()
    if args.use_GPU:
        model = model.cuda()
    if args.mGPUs:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model.train()
    # criterion
    criterion = FocalLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print('start training')
    for epoch in range(args.epochs):
        train_data_iter = iter(dataloader)
        train_loss = 0
        fg, tp = 0, 0
        for step in range(iter_per_epoch):
            im_data, cls_targets, loc_targets, im_sizes = next(train_data_iter)
            if args.use_GPU:
                im_data = im_data.cuda()
                cls_targets = cls_targets.cuda()
                loc_targets = loc_targets.cuda()
            im_data = Variable(im_data)
            cls_targets = Variable(cls_targets)
            loc_targets = Variable(loc_targets)

            cls_preds, loc_preds = model(im_data)
            cls_loss, loc_loss = criterion(cls_preds, cls_targets, loc_preds, loc_targets)

            loss = cls_loss + loc_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # calculate classification acc
            cls_t = cls_targets.clone()
            cls_t = cls_t.view(-1, cfg.CLASS_NUM)
            cls_max, cls_argmax = torch.max(cls_t, dim=-1)
            fg_inds = torch.eq(cls_max, 1.)
            cls_p = cls_preds.clone()
            cls_p = cls_p.view(-1, cfg.CLASS_NUM)
            pred_info = torch.argmax(cls_p, dim=-1)
            tp += torch.sum(pred_info[fg_inds] == cls_argmax[fg_inds])
            fg += fg_inds.sum()

            if (step + 1) % args.display_interval == 0:
                train_loss /= args.display_interval
                print('[%d epoch | %d step]cls_loss: %.3f | loc_loss: %.3f | avg_loss: %.3f | cls_acc: %.3f' %
                    (epoch, step, cls_loss.item(), loc_loss.item(), train_loss, float(tp)/float(fg)))
                if args.use_tfboard:
                    n_iter = epoch * iter_per_epoch + step + 1
                    writer.add_scalar('losses/loss', train_loss, n_iter)
                    writer.add_scalar('losses/cls_loss', cls_loss.item(), n_iter)
                    writer.add_scalar('losses/loc_loss', loc_loss.item(), n_iter)
                    writer.add_scalar('acc/cls_acc', float(tp) / float(fg), n_iter)
                train_loss = 0
                fg, tp = 0, 0

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        if (epoch+1) % args.save_interval == 0:
            print('saving model')
            save_name = os.path.join(args.output_dir, 'retinanet_epoch_{}.pth'.format(epoch + 1))
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
            }, save_name)


if __name__ == '__main__':
    train()
