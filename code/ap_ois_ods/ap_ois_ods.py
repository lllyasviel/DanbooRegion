# A very short and easy-to-read script to compute AP/OIS/ODS between region maps.
# Implemented from the paper
# [Contour Detection and Hierarchical Image Segmentation
# P. Arbelaez, M. Maire, C. Fowlkes and J. Malik.
# IEEE TPAMI, Vol. 33, No. 5, pp. 898-916, May 2011.]
#
# Note that this implementation is based (and only based) on the above official paper
# rather than any other publicly available codes.

import numpy as np
import cv2


def resize_inter_nearest(x, scale):
    if x.shape[0] < x.shape[1]:
        s0 = scale
        s1 = int(float(scale) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(scale) / float(x.shape[1]) * float(x.shape[0]))
        s1 = scale
    return cv2.resize(x, (s1, s0), interpolation=cv2.INTER_NEAREST)


def region2edge(region_map):
    cols = cv2.filter2D(region_map.astype(np.float32), cv2.CV_32F, np.array([[-1, +1]])) ** 2
    rows = cv2.filter2D(region_map.astype(np.float32), cv2.CV_32F, np.array([[-1], [+1]])) ** 2
    result = np.sum(cols + rows, 2)
    result[result > 0] = 1.0
    return result


def compute_precision(ground_truth_region_map, estimated_region_map, scale):
    ground_truth_edge_map = region2edge(resize_inter_nearest(ground_truth_region_map, scale))
    estimated_edge_map = region2edge(resize_inter_nearest(estimated_region_map, scale))
    return np.sum(ground_truth_edge_map * estimated_edge_map) / np.sum(estimated_edge_map)


def AP(image_list, scale=512):
    ap = 0.0
    for img_path in image_list:
        ground_truth = cv2.imread(img_path + '.ground_truth.png')
        estimation = cv2.imread(img_path + '.estimation.png')
        ap += compute_precision(ground_truth, estimation, scale)
    ap /= float(len(image_list))
    return ap


def OIS(image_list):
    ap = 0.0
    for img_path in image_list:
        ground_truth = cv2.imread(img_path + '.ground_truth.png')
        estimation = cv2.imread(img_path + '.estimation.png')
        ap += max([compute_precision(ground_truth, estimation, scale) for scale in range(256, 1024, 64)])
    ap /= float(len(image_list))
    return ap


def ODS(image_list):
    return max([AP(image_list, scale) for scale in range(256, 1024, 64)])


val_list = ['./images/1', './images/2', './images/3']
print('AP (Average precision) = %.5f' % AP(val_list))
print('OIS (Optimal Image Scale) = %.5f' % OIS(val_list))
print('ODS (Optimal Dataset Scale) = %.5f' % ODS(val_list))
