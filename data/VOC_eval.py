import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET


def parse_rec(filename):
    '''
    parse a xml file
    :param filename: xml filename
    :return: a list
    '''
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=True):
    '''
    :param rec: recall
    :param prec: precision
    :param use_07_metric: whether or not use o7 metric
    :return: return ap(average precision)
    '''
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(anno_path, imageset_file, cache_dir, class_name, det_path, ious_thresh, use_07_metric):
    cache_files = os.path.join(cache_dir, 'annots.pkl')

    # read test image names
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_names = [(x.split(' ')[0]).split('.')[0] for x in lines]

    # read annotations and store them in catch_file
    if not os.path.isfile(cache_files):
        # load annotations
        recs = {}
        for i, image_name in enumerate(image_names):
            recs[image_name] = parse_rec(anno_path.format(image_name))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(image_names)))
        # save
        print('Saving cached annotations to {:s}'.format(cache_files))
        with open(cache_files, 'wb') as f:
            pickle.dump(recs, f)

    else:
        with open(cache_files, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')

    class_resc = {}
    npos = 0
    for image_name in image_names:
        R = [obj for obj in recs[image_name] if obj['name'] == class_name]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult']for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos += sum(~difficult)
        class_resc[image_name] = {'bbox': bbox,
                                  'difficult': difficult,
                                  'det': det}

    det_file = det_path.format(class_name)
    print(det_file)
    with open(det_file, 'r') as f:
        lines = f.readlines()

    split_lines = [x.strip().split(' ') for x in lines]
    image_inds = [x[0] for x in split_lines]
    confidence = np.array([float(x[1]) for x in split_lines])
    bbox = np.array([[float(z) for z in x[2:]] for x in split_lines])

    num_image = len(image_inds)
    tp = np.zeros(num_image)
    fp = np.zeros(num_image)

    if bbox.shape[0] > 0:
        sorted_inds = np.argsort(-confidence)
        bbox = bbox[sorted_inds, :]
        image_inds = [image_inds[i] for i in sorted_inds]

        for i in range(num_image):
            R = class_resc[image_inds[i]]
            bb = bbox[i, :].astype(float)
            ious_max = -np.inf
            bbgt = R['bbox'].astype(float)

            if bbgt.size > 0:
                ixmin = np.maximum(bbgt[:, 0], bb[0])
                iymin = np.maximum(bbgt[:, 1], bb[1])
                ixmax = np.minimum(bbgt[:, 2], bb[2])
                iymax = np.minimum(bbgt[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                       (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

                ious = inters / uni
                ious_max = np.max(ious)
                ious_argmax = np.argmax(ious)

            if ious_max > ious_thresh:
                if not R['difficult'][ious_argmax]:
                    if not R['det'][ious_argmax]:
                        tp[i] = 1.
                        R['det'][ious_argmax] = 1.
                    else:
                        fp[i] = 1.
                else:
                    fp[i] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

