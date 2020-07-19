import numpy as np
import cv2

from numba import njit
from scipy.ndimage import label


def thinning(fillmap, max_iter=100):
    """Fill area of line with surrounding fill color.

    # Arguments
        fillmap: an image.
        max_iter: max iteration number.

    # Returns
        an image.
    """
    line_id = 0
    h, w = fillmap.shape[:2]
    result = fillmap.copy()

    for iterNum in range(max_iter):
        # Get points of line. if there is not point, stop.
        line_points = np.where(result == line_id)
        if not len(line_points[0]) > 0:
            break

        # Get points between lines and fills.
        line_mask = np.full((h, w), 255, np.uint8)
        line_mask[line_points] = 0
        line_border_mask = cv2.morphologyEx(line_mask, cv2.MORPH_DILATE,
                                            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), anchor=(-1, -1),
                                            iterations=1) - line_mask
        line_border_points = np.where(line_border_mask == 255)

        result_tmp = result.copy()
        # Iterate over points, fill each point with nearest fill's id.
        for i, _ in enumerate(line_border_points[0]):
            x, y = line_border_points[1][i], line_border_points[0][i]

            if x - 1 > 0 and result[y][x - 1] != line_id:
                result_tmp[y][x] = result[y][x - 1]
                continue

            if x - 1 > 0 and y - 1 > 0 and result[y - 1][x - 1] != line_id:
                result_tmp[y][x] = result[y - 1][x - 1]
                continue

            if y - 1 > 0 and result[y - 1][x] != line_id:
                result_tmp[y][x] = result[y - 1][x]
                continue

            if y - 1 > 0 and x + 1 < w and result[y - 1][x + 1] != line_id:
                result_tmp[y][x] = result[y - 1][x + 1]
                continue

            if x + 1 < w and result[y][x + 1] != line_id:
                result_tmp[y][x] = result[y][x + 1]
                continue

            if x + 1 < w and y + 1 < h and result[y + 1][x + 1] != line_id:
                result_tmp[y][x] = result[y + 1][x + 1]
                continue

            if y + 1 < h and result[y + 1][x] != line_id:
                result_tmp[y][x] = result[y + 1][x]
                continue

            if y + 1 < h and x - 1 > 0 and result[y + 1][x - 1] != line_id:
                result_tmp[y][x] = result[y + 1][x - 1]
                continue

        result = result_tmp.copy()

    return result


def topo_compute_normal(dist):
    c = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, +1]]))
    r = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1], [+1]]))
    h = np.zeros_like(c + r, dtype=np.float32) + 0.75
    normal_map = np.stack([h, r, c], axis=2)
    normal_map /= np.sum(normal_map ** 2.0, axis=2, keepdims=True) ** 0.5
    return normal_map


@njit
def count_all(labeled_array, all_counts):
    M = labeled_array.shape[0]
    N = labeled_array.shape[1]
    for x in range(M):
        for y in range(N):
            i = labeled_array[x, y] - 1
            if i > -1:
                all_counts[i] = all_counts[i] + 1
    return


@njit
def trace_all(labeled_array, xs, ys, cs):
    M = labeled_array.shape[0]
    N = labeled_array.shape[1]
    for x in range(M):
        for y in range(N):
            current_label = labeled_array[x, y] - 1
            if current_label > -1:
                current_label_count = cs[current_label]
                xs[current_label][current_label_count] = x
                ys[current_label][current_label_count] = y
                cs[current_label] = current_label_count + 1
    return


def find_all(labeled_array):
    hist_size = int(np.max(labeled_array))
    if hist_size == 0:
        return []
    all_counts = [0 for _ in range(hist_size)]
    count_all(labeled_array, all_counts)
    xs = [np.zeros(shape=(item, ), dtype=np.uint32) for item in all_counts]
    ys = [np.zeros(shape=(item, ), dtype=np.uint32) for item in all_counts]
    cs = [0 for item in all_counts]
    trace_all(labeled_array, xs, ys, cs)
    filled_area = []
    for _ in range(hist_size):
        filled_area.append((xs[_], ys[_]))
    return filled_area


def mk_resize(x, k):
    if x.shape[0] < x.shape[1]:
        s0 = k
        s1 = int(x.shape[1] * (k / x.shape[0]))
        s1 = s1 - s1 % 128
        _s0 = 32 * s0
        _s1 = int(x.shape[1] * (_s0 / x.shape[0]))
        _s1 = (_s1 + 64) - (_s1 + 64) % 128
    else:
        s1 = k
        s0 = int(x.shape[0] * (k / x.shape[1]))
        s0 = s0 - s0 % 128
        _s1 = 32 * s1
        _s0 = int(x.shape[0] * (_s1 / x.shape[1]))
        _s0 = (_s0 + 64) - (_s0 + 64) % 128
    new_min = min(_s1, _s0)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
    return y


def k_resize(x, k):
    if x.shape[0] < x.shape[1]:
        s0 = k
        s1 = int(x.shape[1] * (k / x.shape[0]))
        s1 = s1 - s1 % 64
        _s0 = 16 * s0
        _s1 = int(x.shape[1] * (_s0 / x.shape[0]))
        _s1 = (_s1 + 32) - (_s1 + 32) % 64
    else:
        s1 = k
        s0 = int(x.shape[0] * (k / x.shape[1]))
        s0 = s0 - s0 % 64
        _s1 = 16 * s1
        _s0 = int(x.shape[0] * (_s1 / x.shape[1]))
        _s0 = (_s0 + 32) - (_s0 + 32) % 64
    new_min = min(_s1, _s0)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
    return y


def sk_resize(x, k):
    if x.shape[0] < x.shape[1]:
        s0 = k
        s1 = int(x.shape[1] * (k / x.shape[0]))
        s1 = s1 - s1 % 16
        _s0 = 4 * s0
        _s1 = int(x.shape[1] * (_s0 / x.shape[0]))
        _s1 = (_s1 + 8) - (_s1 + 8) % 16
    else:
        s1 = k
        s0 = int(x.shape[0] * (k / x.shape[1]))
        s0 = s0 - s0 % 16
        _s1 = 4 * s1
        _s0 = int(x.shape[0] * (_s1 / x.shape[1]))
        _s0 = (_s0 + 8) - (_s0 + 8) % 16
    new_min = min(_s1, _s0)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
    return y


def d_resize(x, d, fac=1.0):
    new_min = min(int(d[1] * fac), int(d[0] * fac))
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
    return y


def n_resize(x, d):
    y = cv2.resize(x, (d[1], d[0]), interpolation=cv2.INTER_NEAREST)
    return y


def s_resize(x, s):
    if x.shape[0] < x.shape[1]:
        s0 = x.shape[0]
        s1 = int(float(s0) / float(s[0]) * float(s[1]))
    else:
        s1 = x.shape[1]
        s0 = int(float(s1) / float(s[1]) * float(s[0]))
    new_max = max(s1, s0)
    raw_max = max(x.shape[0], x.shape[1])
    if new_max < raw_max:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (s1, s0), interpolation=interpolation)
    return y


def min_resize(x, m):
    if x.shape[0] < x.shape[1]:
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
    new_max = max(s1, s0)
    raw_max = max(x.shape[0], x.shape[1])
    if new_max < raw_max:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (s1, s0), interpolation=interpolation)
    return y


def n_min_resize(x, m):
    if x.shape[0] < x.shape[1]:
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
    new_max = max(s1, s0)
    raw_max = max(x.shape[0], x.shape[1])
    y = cv2.resize(x, (s1, s0), interpolation=cv2.INTER_NEAREST)
    return y


def h_resize(x, m):
    s0 = m
    s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    new_max = max(s1, s0)
    raw_max = max(x.shape[0], x.shape[1])
    if new_max < raw_max:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (s1, s0), interpolation=interpolation)
    return y


def w_resize(x, m):
    s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
    s1 = m
    new_max = max(s1, s0)
    raw_max = max(x.shape[0], x.shape[1])
    if new_max < raw_max:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (s1, s0), interpolation=interpolation)
    return y


def max_resize(x, m):
    if x.shape[0] > x.shape[1]:
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
    new_max = max(s1, s0)
    raw_max = max(x.shape[0], x.shape[1])
    if new_max < raw_max:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (s1, s0), interpolation=interpolation)
    return y
