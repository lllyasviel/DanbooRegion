from tricks import *
from ai import *


def go_flipped_vector(x):
    a = go_vector(x)
    b = np.fliplr(go_vector(np.fliplr(x)))
    c = np.flipud(go_vector(np.flipud(x)))
    d = np.flipud(np.fliplr(go_vector(np.flipud(np.fliplr(x)))))
    return (a + b + c + d) / 4.0


def go_transposed_vector(x):
    a = go_flipped_vector(x)
    b = np.transpose(go_flipped_vector(np.transpose(x, [1, 0, 2])), [1, 0, 2])
    return (a + b) / 2.0


def get_fill(image):
    labeled_array, num_features = label(image / 255)
    filled_area = find_all(labeled_array)
    return filled_area


def up_fill(fills, cur_fill_map):
    new_fillmap = cur_fill_map.copy()
    padded_fillmap = np.pad(cur_fill_map, [[1, 1], [1, 1]], 'constant', constant_values=0)
    max_id = np.max(cur_fill_map)
    for item in fills:
        points0 = padded_fillmap[(item[0] + 1, item[1] + 0)]
        points1 = padded_fillmap[(item[0] + 1, item[1] + 2)]
        points2 = padded_fillmap[(item[0] + 0, item[1] + 1)]
        points3 = padded_fillmap[(item[0] + 2, item[1] + 1)]
        all_points = np.concatenate([points0, points1, points2, points3], axis=0)
        pointsets, pointcounts = np.unique(all_points[all_points > 0], return_counts=True)
        if len(pointsets) == 1 and item[0].shape[0] < 128:
            new_fillmap[item] = pointsets[0]
        else:
            max_id += 1
            new_fillmap[item] = max_id
    return new_fillmap


def segment(image):
    raw_img = go_srcnn(min_resize(image, 512)).clip(0, 255).astype(np.uint8)
    img_2048 = min_resize(raw_img, 2048)
    height = d_resize(go_transposed_vector(mk_resize(raw_img, 64)), img_2048.shape) * 255.0
    final_height = height.copy()
    height += (height - cv2.GaussianBlur(height, (0, 0), 3.0)) * 10.0
    height = height.clip(0, 255).astype(np.uint8)
    marker = height.copy()
    marker[marker > 135] = 255
    marker[marker < 255] = 0
    fills = get_fill(marker / 255)
    for fill in fills:
        if fill[0].shape[0] < 64:
            marker[fill] = 0
    filter = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]],
        dtype=np.uint8)
    big_marker = cv2.erode(marker, filter, iterations=5)
    fills = get_fill(big_marker / 255)
    for fill in fills:
        if fill[0].shape[0] < 64:
            big_marker[fill] = 0
    big_marker = cv2.dilate(big_marker, filter, iterations=5)
    small_marker = marker.copy()
    small_marker[big_marker > 127] = 0
    fin_labels, nil = label(big_marker / 255)
    fin_labels = up_fill(get_fill(small_marker), fin_labels)
    water = cv2.watershed(img_2048.clip(0, 255).astype(np.uint8), fin_labels.astype(np.int32)) + 1
    water = thinning(water)
    all_region_indices = find_all(water)
    regions = np.zeros_like(img_2048, dtype=np.uint8)
    for region_indices in all_region_indices:
        regions[region_indices] = np.random.randint(low=0, high=255, size=(3,)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(img_2048, dtype=np.uint8)
    for region_indices in all_region_indices:
        result[region_indices] = np.median(img_2048[region_indices], axis=0)
    return final_height.clip(0, 255).astype(np.uint8), regions.clip(0, 255).astype(np.uint8), result.clip(0, 255).astype(np.uint8)


if __name__=='__main__':
    import sys
    image = cv2.imread(sys.argv[1])
    skeleton, region, flatten = segment(image)
    cv2.imwrite('./current_skeleton.png', skeleton)
    cv2.imwrite('./current_region.png', region)
    cv2.imwrite('./current_flatten.png', flatten)
    print('./current_skeleton.png')
    print('./current_region.png')
    print('./current_flatten.png')
    print('ok!')
