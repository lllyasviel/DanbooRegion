from tricks import *


def get_regions(skeleton_map):
    marker = skeleton_map[:, :, 0]
    normal = topo_compute_normal(marker) * 127.5 + 127.5
    marker[marker > 100] = 255
    marker[marker < 255] = 0
    labels, nil = label(marker / 255)
    water = cv2.watershed(normal.clip(0, 255).astype(np.uint8), labels.astype(np.int32)) + 1
    water = thinning(water)
    all_region_indices = find_all(water)
    regions = np.zeros_like(skeleton_map, dtype=np.uint8)
    for region_indices in all_region_indices:
        regions[region_indices] = np.random.randint(low=0, high=255, size=(3,)).clip(0, 255).astype(np.uint8)
    return regions


if __name__=='__main__':
    import sys
    skeleton_map = cv2.imread(sys.argv[1])
    cv2.imshow('vis', get_regions(skeleton_map))
    cv2.waitKey(0)
