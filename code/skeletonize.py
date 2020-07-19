from tricks import *
from skimage.morphology import skeletonize, dilation


def get_skeleton(region_map):
    Xp = np.pad(region_map, [[0, 1], [0, 0], [0, 0]], 'symmetric').astype(np.float32)
    Yp = np.pad(region_map, [[0, 0], [0, 1], [0, 0]], 'symmetric').astype(np.float32)
    X = np.sum((Xp[1:, :, :] - Xp[:-1, :, :]) ** 2.0, axis=2) ** 0.5
    Y = np.sum((Yp[:, 1:, :] - Yp[:, :-1, :]) ** 2.0, axis=2) ** 0.5
    edge = np.zeros_like(region_map)[:, :, 0]
    edge[X > 0] = 255
    edge[Y > 0] = 255
    edge[0, :] = 255
    edge[-1, :] = 255
    edge[:, 0] = 255
    edge[:, -1] = 255
    skeleton = 1.0 - dilation(edge.astype(np.float32) / 255.0)
    skeleton = skeletonize(skeleton)
    skeleton = (skeleton * 255.0).clip(0, 255).astype(np.uint8)
    field = np.random.uniform(low=0.0, high=255.0, size=edge.shape).clip(0, 255).astype(np.uint8)
    field[skeleton > 0] = 255
    field[edge > 0] = 0
    filter = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]],
        dtype=np.float32) / 5.0
    height = np.random.uniform(low=0.0, high=255.0, size=field.shape).astype(np.float32)
    for _ in range(512):
        height = cv2.filter2D(height, cv2.CV_32F, filter)
        height[skeleton > 0] = 255.0
        height[edge > 0] = 0.0
    return height.clip(0, 255).astype(np.uint8)


if __name__=='__main__':
    import sys
    region_map = cv2.imread(sys.argv[1])
    cv2.imshow('vis', get_skeleton(region_map))
    cv2.waitKey(0)
