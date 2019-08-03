import math
from easydict import EasyDict

__C = EasyDict()

cfg = __C

__C.DATASET_NAME = ''
__C.INPUT_SIZE = (600, 600)

__C.CLASS_NUM = 20
__C.CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

__C.ANCHOR_SIZES = [32, 64, 128, 256, 512]
__C.ASPECT_RATIOS = [0.5, 1.0, 2.0]
__C.SCALE_RATIOS = [pow(2, 0/3.), pow(2, 1/3.), pow(2, 2/3.)]

