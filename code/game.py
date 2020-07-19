from config import *
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import keras
from model import *
import time
import random
import keras.backend as K
import os

seed = random.randint(0, 2**31 - 1)
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


def clip2float(map):
    map = map.clip(0, 255).astype(np.float) / 255.0
    return map


def clip2unit8(map):
    return (map * 255.0).clip(0, 255).astype(np.uint8)


def tackle_batch(batch):
    if batch.shape[3] == 1:
        batch = np.tile(batch, [1, 1, 1, 3])
    if batch.shape[3] >= 3:
        batch = batch[:, :, :, 0:3]
    batch = clip2unit8(batch)
    temp = []
    for _ in range(batch_size):
        temp.append(cv2.resize(batch[_], (show_size, show_size), interpolation=cv2.INTER_NEAREST))
    batch = np.reshape(np.stack(temp, axis=0), (batch_size, 1, show_size, show_size, 3))
    batch = np.transpose(batch, [0, 2, 1, 3, 4])
    batch = np.reshape(batch, (batch_size * show_size, show_size, 3))
    return batch


def load_weight(net, name):
    try:
        net.load_weights(name)
    except Exception:
        net.save(name)
        print(name + '--created')


global_step = 0
EPS = 1e-12

session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=str(gpu))))
K.set_session(session)

with tf.variable_scope("generator"):
    vector_net = make_diff_net()

with tf.variable_scope("discriminator"):
    discriminator = make_cnet512()

generator_var_list = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
discriminator_var_list = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

generator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

paint_input = tf.placeholder(dtype=tf.float32, shape=(batch_size, 512, 512, 3))
sketch_input = tf.placeholder(dtype=tf.float32, shape=(batch_size, 512, 512, 1))

real_samples = sketch_input
fake_samples = 1.0 - vector_net(paint_input)

G_loss, D_loss, Grad = main_gan(real_samples=real_samples, fake_samples=fake_samples, discriminator=discriminator)

generator_train = generator_optimizer.minimize(loss=G_loss, var_list=generator_var_list)
discriminator_train = discriminator_optimizer.minimize(loss=D_loss, var_list=discriminator_var_list)
with tf.control_dependencies([generator_train, discriminator_train]):
    train = tf.no_op()

refed = [
    paint_input,
    real_samples,
    fake_samples,
    Grad
]

session.run(tf.global_variables_initializer())


def load_all_weights():
    load_weight(vector_net, 'saved_model.net')
    load_weight(discriminator, 'saved_discriminator.net')
    return


def save_all_weights():
    vector_net.save('saved_model.net')
    discriminator.save('saved_discriminator.net')
    return


os.makedirs('results', exist_ok=True)


def train_on_batch(sketch_batch, paint_batch):
    global global_step
    sketch_batch = clip2float(sketch_batch)
    paint_batch = clip2float(paint_batch)
    feed_dict = {sketch_input: sketch_batch, paint_input: paint_batch}
    session.run(train, feed_dict=feed_dict)
    if global_step % 10 == 0:
        bs = session.run(refed, feed_dict=feed_dict)
        for _ in range(len(bs)):
            bs[_] = tackle_batch(bs[_])
        cv2.imwrite('results/' + str(global_step) + ".jpg", np.concatenate(bs, axis=1))
    global_step = global_step + 1
    return
