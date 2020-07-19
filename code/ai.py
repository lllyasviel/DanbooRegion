from model import *
import numpy as np
import keras.backend as K
from keras.models import load_model


def build_repeat_mulsep(x, m, i):
    a = m[:, :, 0]
    b = m[:, :, 1]
    c = m[:, :, 2]
    d = m[:, :, 3]
    e = m[:, :, 4]
    y = x
    for _ in range(i):
        p = tf.pad(y, [[1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
        y = p[:-2, 1:-1, :] * a + p[1:-1, :-2, :] * b + y * c + p[1:-1, 2:, :] * d + p[2:, 1:-1, :] * e
    return y


def np_expand_image(x):
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), 'symmetric')
    r = []
    r.append(p[:-2, 1:-1, :])
    r.append(p[1:-1, :-2, :])
    r.append(p[1:-1, 1:-1, :])
    r.append(p[1:-1, 2:, :])
    r.append(p[2:, 1:-1, :])
    return np.stack(r, axis=2)


def build_sketch_sparse(x):
    x = x[:, :, None].astype(np.float32)
    expanded = np_expand_image(x)
    distance = x[:, :, None] - expanded
    distance = np.abs(distance)
    weight = 8 - distance
    weight[weight < 0] = 0.0
    weight /= np.sum(weight, axis=2, keepdims=True)
    return weight


def go_refine_sparse(x, sparse_matrix):
    return session.run(tf_sparse_op_H, feed_dict={ipsp3: x[:, :, None], ipsp9: sparse_matrix})[:, :, 0]


session = tf.Session()
K.set_session(session)

ip3 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

vector = make_diff_net()
vector_op = 1.0 - vector(ip3 / 255.0)

ipsp9 = tf.placeholder(dtype=tf.float32, shape=(None, None, 5, 1))
ipsp3 = tf.placeholder(dtype=tf.float32, shape=(None, None, 1))
tf_sparse_op_H = build_repeat_mulsep(ipsp3, ipsp9, 256)

srcnn = load_model('srcnn.net')
pads = 7
srcnn_op = srcnn(tf.pad(ip3 / 255.0, [[0, 0], [pads, pads], [pads, pads], [0, 0]], 'REFLECT'))[:, pads * 2:-pads * 2, pads * 2:-pads * 2, :][:, 1:-1, 1:-1, :] * 255.0

session.run(tf.global_variables_initializer())

print('begin load')
vector.load_weights('DanbooRegion2020UNet.net')
srcnn.load_weights('srcnn.net')


def go_vector(x):
    return session.run(vector_op, feed_dict={
        ip3: x[None, :, :, :]
    })[0]


def go_srcnn(x):
    return session.run(srcnn_op, feed_dict={
        ip3: x[None, :, :, :]
    })[0]
