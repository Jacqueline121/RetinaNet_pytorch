import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torchvision
from utils.data_augmentation import resize, random_flip, random_crop, center_crop
#from target.target import get_target
from target.my_targets import get_target
from config.config import cfg
from PIL import ImageDraw, Image
from target.anchors import generate_anchors
import numpy as np


root_dir = os.path.join(os.path.dirname(__file__), '')

image_dir = os.path.join(root_dir, 'dataset/voc07_JPEGImages')
anno_file = os.path.join(root_dir, 'dataset/voc07_val.txt')


class VOCDataset(Dataset):
    def __init__(self, transform, train):
        self.transform = transform
        self.train = train

        self.image_names = []
        self.boxes = []
        self.labels = []
        with open(anno_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            annos = line.split(' ')
            self.image_names.append(annos[0])
            num_boxes = (len(annos) - 1) // 5

            box = []
            label = []
            for i in range(num_boxes):
                xmin = annos[i * 5 + 1]
                ymin = annos[i * 5 + 2]
                xmax = annos[i * 5 + 3]
                ymax = annos[i * 5 + 4]
                l = annos[i * 5 + 5]
                box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                label.append(int(l))
            self.boxes.append(torch.Tensor(box))  #xyxy
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(os.path.join(image_dir, image_name))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        im_size = image.size

        input_size = cfg.INPUT_SIZE

        if self.train:
            image, boxes = random_flip(image, boxes)
            image, boxes = resize(image, boxes, input_size)
        else:
            image, boxes = resize(image, boxes, input_size)
            image, boxes = center_crop(image, boxes, input_size)

        image = self.transform(image)
        return image, boxes, labels, im_size

    def collate_fn(self, batch):
        images = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        im_sizes = [x[3] for x in batch]

        w, h = cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1]
        num_images = len(images)
        inputs = torch.zeros(num_images, 3, h, w)
        cls_targets = []
        loc_targets = []
        im_sizes = torch.Tensor(im_sizes)

        for i in range(num_images):
            inputs[i] = images[i]
            cls_target, loc_target = get_target(boxes[i], labels[i])
            cls_targets.append(cls_target)
            loc_targets.append(loc_target)
        return inputs, torch.stack(cls_targets), torch.stack(loc_targets), im_sizes

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = VOCDataset(train=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
    for i in range(10):
        for images, cls_targets, loc_targets, im_sizes in dataloader:
            print(images.size())
            print(loc_targets.size())
            print(cls_targets.size())
            print(im_sizes.size())
            grid = torchvision.utils.make_grid(images, 1)
            torchvision.utils.save_image(grid, 'a.jpg')
            break
        break

