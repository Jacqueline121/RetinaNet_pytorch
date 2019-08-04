# RetinaNet_pytorch

A Python3.5/Pytroch implementation of RetinaNet: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). And the official implementations are available [here](https://github.com/facebookresearch/Detectron). Besides, special thanks for those two repositories：
* [pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet)
* [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet)

### Prerequisites
* python 3.5.x
* pytorch 0.4.1
* tensorboardX
* pillow
* scipy
* numpy
* matplotlib
* easydict

### Repo Organization
* **RetinaNet**: neural networks and components that form parts of RetinaNet.
* **config**: define configuration information of Faster RCNN.
* **data**: scripts for creating, downloading, organizing datasets.
* **loss**: implementation of focal loss.
* **pretrained_model**: get and store pretrained ResNet model.
* **targets**: generate anchors and calculate targets.
* **utils**: tools package, containing some necessary functions.

### Installation

1. Clone this repository (RetinaNet_pytorch):
    
        git clone --recursive https://github.com/Jacqueline121/RetinaNet_pytorch.git

2. Install dependencies:
    
        cd RetinaNet_pytorch 
        pip install -r requirements.txt

### Train
#### Prepare the Data
For PASCAL VOC, you can follow the instructions in this [repository](https://github.com/Jacqueline121/Faster_RCNN_pytorch) to download the data. And then, you can store date according the following structure:
    .
    +-- data
    |   +-- dataset
    |       +-- VOC2007
    |           +-- Annotations
    |               +-- xxxx.xml
    |           +-- Cache
    |           +-- ImageSets
    |           +-- JPEGImages
    |           +-- Results
    |       +-- VOC2012
    |           +-- Annotations
    |               +-- xxxx.xml
    |           +-- Cache
    |           +-- ImageSets
    |           +-- JPEGImages
    |           +-- Results

* **Annotations**: store annotaion information(.xml file) for each images.
* **Cache**: store annotaion cache.
* **ImageSets**: store training dataset and testing dataset(.txt file) with the format:
* **JPEGImages**: store images.
* **Results**: store detection results.

You can also use your own dataset as long as you follow the file structure desribed above to store the data.

#### Get pretrained model
1. Download the pretrained ResNet model: [ResNet50](https://drive.google.com/open?id=0B7fNdx_jAqhtbllXbWxMVEdZclE), [ResNet101](https://drive.google.com/open?id=0B7fNdx_jAqhtbllXbWxMVEdZclE).

2. Put the pretrained model in '$PROJECT/pretrained_model'

3. cd '$PROJECT/pretrained_model'
    '''python
    python get_pretrained_model.py
    '''

    It will produce a 'model.pth' file.

#### Train
    python train.py --dataset='VOC2007'

### Test
    python test.py --dataset='VOC2007'

If you want to visualize the detection result, you can use:
    
    python test.py --vis
