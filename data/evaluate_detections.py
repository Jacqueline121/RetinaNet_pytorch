import os
import pickle
import numpy as np
from data.VOC_eval import voc_eval
from config.config import cfg

dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')


def load_image_set_index(image_set):
    image_set_file = os.path.join(dataset_dir, cfg.DATASET_NAME, 'ImageSets',  image_set + '.txt')
    assert os.path.exists(image_set_file), \
        'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index = [(x.split(' ')[0]).split('.')[0] for x in f.readlines()]
    return image_index


def get_results_file_template(image_set):
    filename = 'comp4' + '_det_' + image_set + '_{:s}.txt'
    filedir = os.path.join(dataset_dir, cfg.DATASET_NAME, 'Results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_results_file(all_boxes):
    image_index = load_image_set_index('test')  # get val image index(id)
    for cls_ind, cls in enumerate(cfg.CLASSES):
        print('Writing {} results file'.format(cls))
        filename = get_results_file_template('test').format(cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(image_index):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir, use_07_metric):
    anno_path = os.path.join(dataset_dir, cfg.DATASET_NAME, 'Annotations', '{:s}.xml')
    imageset_file = os.path.join(dataset_dir, cfg.DATASET_NAME, 'ImageSets', 'test.txt')
    cache_dir = os.path.join(dataset_dir, cfg.DATASET_NAME, 'Cache')
    aps = []

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i, cls in enumerate(cfg.CLASSES):
        file_name = get_results_file_template('test').format(cls)
        rec, prec, ap = voc_eval(
            anno_path=anno_path,
            imageset_file=imageset_file,
            cache_dir=cache_dir,
            class_name=cls,
            det_path=file_name,
            ious_thresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')


def evaluate_detections(all_boxes, output_dir):
    write_results_file(all_boxes)
    do_python_eval(output_dir, True)

















