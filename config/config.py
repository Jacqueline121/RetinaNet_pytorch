import math
from easydict import EasyDict

__C = EasyDict()

cfg = __C

__C.INPUT_SIZE = (600, 600)

__C.NUM_CLASS = 20

__C.ANCHOR_SIZES = [32, 64, 128, 256, 512]
__C.ASPECT_RATIOS = [0.5, 1.0, 2.0]
__C.SCALE_RATIOS = [pow(2, 0/3.), pow(2, 1/3.), pow(2, 2/3.)]

