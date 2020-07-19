from tricks import *


def vis(region_map, color_map):
    color = d_resize(color_map, region_map.shape)
    indexs = (region_map.astype(np.float32)[:, :, 0] * 255 + region_map.astype(np.float32)[:, :, 1]) * 255 + region_map.astype(np.float32)[:, :, 2]
    result = np.zeros_like(color, dtype=np.uint8)
    for ids in [np.where(indexs == idsn) for idsn in np.unique(indexs).tolist()]:
        result[ids] = np.median(color[ids], axis=0)
    return result


if __name__=='__main__':
    import sys
    color_map = cv2.imread(sys.argv[1])
    region_map = cv2.imread(sys.argv[2])
    cv2.imshow('vis', vis(region_map, color_map))
    cv2.waitKey(0)