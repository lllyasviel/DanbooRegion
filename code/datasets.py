from rotate import *


def d_resize(x, d, fac=1.0):
    new_min = min(int(d[1] * fac), int(d[0] * fac))
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
    return y


def np_RGB2GRAY(img):
    R = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    B = img[:, :, 2].astype(np.float32)
    r = np.random.rand()
    g = np.random.rand()
    b = np.random.rand()
    s = r + g + b
    r /= s
    g /= s
    b /= s
    light = R * r + G * g + B * b
    light -= np.min(light)
    light /= np.max(light)
    a = np.random.rand() * 0.4
    b = 1 - np.random.rand() * 0.4
    light = light.clip(a, b)
    light -= np.min(light)
    light /= np.max(light)
    light = light.clip(0, 1)
    light = (light * 255.0).astype(np.uint8)
    return light


def handle_next():
    indice = np.random.randint(low=0, high=3377)
    paint_r_mat = cv2.imread('./DanbooRegion2020/train/' + str(indice) + '.image.png')
    sketch_r_mat = cv2.imread('./DanbooRegion2020/train/' + str(indice) + '.skeleton.png')

    image_height, image_width = sketch_r_mat.shape[0:2]
    paint_r_mat = d_resize(paint_r_mat, sketch_r_mat.shape)

    if np.random.rand() < 0.5:
        ri = np.random.rand() * 360.0
        paint_r_mat = crop_around_center(rotate_image(paint_r_mat, ri), *largest_rotated_rect(image_width, image_height, math.radians(ri)))
        sketch_r_mat = crop_around_center(rotate_image(sketch_r_mat, ri), *largest_rotated_rect(image_width, image_height, math.radians(ri)))
        kernel = np.random.randint(520, 650)
    else:
        kernel = np.random.randint(520, 1024)
    raw_s0 = float(paint_r_mat.shape[0])
    raw_s1 = float(paint_r_mat.shape[1])
    if raw_s0 < raw_s1:
        new_s0 = int(kernel)
        new_s1 = int(kernel / raw_s0 * raw_s1)
    else:
        new_s1 = int(kernel)
        new_s0 = int(kernel / raw_s1 * raw_s0)
    c0 = int(np.random.rand() * float(new_s0 - 512))
    c1 = int(np.random.rand() * float(new_s1 - 512))
    paint_mat = d_resize(paint_r_mat, (new_s0, new_s1))[c0:c0 + 512, c1:c1 + 512, :]
    sketch_mat = d_resize(sketch_r_mat, (new_s0, new_s1))[c0:c0 + 512, c1:c1 + 512, 0:1]

    if np.random.rand() < 0.5:
        sketch_mat = np.fliplr(sketch_mat)
        paint_mat = np.fliplr(paint_mat)

    if np.random.rand() < 0.5:
        sketch_mat = np.flipud(sketch_mat)
        paint_mat = np.flipud(paint_mat)

    if np.random.rand() < 0.5:
        paint_mat = np.stack([np_RGB2GRAY(paint_mat), np_RGB2GRAY(paint_mat), np_RGB2GRAY(paint_mat)], axis=2)

    if np.random.rand() < 0.5:
        for _ in range(int(np.random.randint(low=0, high=5))):
            paint_mat = cv2.GaussianBlur(paint_mat, (0, 0), 2.0)

    if np.random.rand() < 0.5:
        for _ in range(int(np.random.randint(low=0, high=5))):
            paint_mat = cv2.medianBlur(paint_mat, 3)

    return sketch_mat, paint_mat
