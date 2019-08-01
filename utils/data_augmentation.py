import math
import random
import torch
from PIL import Image, ImageDraw


def resize(image, boxes, size, max_size=1333):
    w, h = image.size
    if isinstance(size, int):
        size_min = min(w, h)
        size_max = max(w, h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return image.resize((ow, oh), Image.BILINEAR), \
           boxes * torch.Tensor([sw, sh, sw, sh])


def random_crop(image, boxes):
    '''
    crop the given image to a random size and aspect ratio
    :param image:
    :param boxes:
    :return:
    '''

    success = False
    image_w, image_h = image.size
    for attempt in range(10):
        area = image_w * image_h
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= image_w and h <= image_h:
            x = random.randint(0, image_w - w)
            y = random.randint(0, image_h - h)
            success = True
            break

        if not success:
            w = h = min(image_w, image_h)
            x = (image_w - w) // 2
            y = (image_h - h) // 2

        image = image.crop([x, y, x + w, y + h])
        boxes -= torch.Tensor([x, y, x, y])
        boxes[:, 0::2].clamp_(min=0, max=w-1)
        boxes[:, 1::2].clamp_(min=0, max=h-1)
        return image, boxes


def center_crop(image, boxes, size):
    w, h = image.size
    ow, oh = size
    i = int(round((w - ow) / 2.))
    j = int(round((h - oh) / 2.))
    image = image.crop((i, j, i + ow, j + oh))
    boxes -= torch.Tensor([i, j, i, j])
    boxes[:, 0::2].clamp_(min=0, max=ow - 1)
    boxes[:, 1::2].clamp_(min=0, max=oh - 1)
    return image, boxes


def random_flip(image, boxes):
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        w = image.width
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
    return image, boxes


def draw(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    image.show()


if __name__ == '__main__':
    image_path = '/home/ilab/hk/lung/RetinaNet_Pytorch/data/000006.jpg'
    image = Image.open(image_path)
    boxes = torch.Tensor([[48, 240, 195, 371], [8, 12, 352, 498]])
    img, boxes = center_crop(image, boxes, (300, 300))
    print(img.size)
    draw(img, boxes)



