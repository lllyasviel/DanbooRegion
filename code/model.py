from keras.layers import Conv2D, Activation, Input, Concatenate, LeakyReLU, Lambda, AveragePooling2D, UpSampling2D, Convolution2D, BatchNormalization, Deconvolution2D, Add
from keras.models import Model

import tensorflow as tf


def make_diff_net():

    def conv(x, filters, name):
        return Conv2D(filters=filters, strides=(1, 1), kernel_size=(3, 3), padding='same', name=name)(x)

    def relu(x):
        return Activation('relu')(x)

    def lrelu(x):
        return LeakyReLU(alpha=0.1)(x)

    def r_block(x, filters, name=None):
        return relu(conv(relu(conv(x, filters, None if name is None else name + '_c1')), filters,
                         None if name is None else name + '_c2'))

    def cat(a, b):
        return Concatenate()([UpSampling2D((2, 2))(a), b])

    def dog(x):
        down = AveragePooling2D((2, 2))(x)
        up = UpSampling2D((2, 2))(down)
        diff = Lambda(lambda p: p[0] - p[1])([x, up])
        return down, diff

    ip = Input(shape=(512, 512, 3))

    c512 = r_block(ip, 16, 'c512')

    c256, l512 = dog(c512)
    c256 = r_block(c256, 32, 'c256')

    c128, l256 = dog(c256)
    c128 = r_block(c128, 64, 'c128')

    c64, l128 = dog(c128)
    c64 = r_block(c64, 128, 'c64')

    c32, l64 = dog(c64)
    c32 = r_block(c32, 256, 'c32')

    c16, l32 = dog(c32)
    c16 = r_block(c16, 512, 'c16')

    d32 = cat(c16, l32)
    d32 = r_block(d32, 256, 'd32')

    d64 = cat(d32, l64)
    d64 = r_block(d64, 128, 'd64')

    d128 = cat(d64, l128)
    d128 = r_block(d128, 64, 'd128')

    d256 = cat(d128, l256)
    d256 = r_block(d256, 32, 'd256')

    d512 = cat(d256, l512)
    d512 = r_block(d512, 16, 'd512')

    op = conv(d512, 1, 'op')

    return Model(inputs=ip, outputs=op)


def make_cnet512():

    def conv(x, filters, strides=(1, 1), kernel_size=(3, 3)):
        return Conv2D(filters=filters, strides=strides, kernel_size=kernel_size, padding='same')(x)

    def relu(x):
        return Activation('relu')(x)

    def lrelu(x):
        return LeakyReLU(alpha=0.1)(x)

    def cat(a, b):
        return Concatenate()([UpSampling2D((2, 2))(a), b])

    def dog(x):
        return AveragePooling2D((2, 2))(x)

    ip = Input(shape=(512, 512, 1))

    c512 = lrelu(conv(ip, 16, strides=(1, 1), kernel_size=(3, 3)))

    c256 = lrelu(conv(c512, 32, strides=(2, 2), kernel_size=(4, 4)))
    c256 = lrelu(conv(c256, 32, strides=(1, 1), kernel_size=(3, 3)))

    c128 = lrelu(conv(c256, 64, strides=(2, 2), kernel_size=(4, 4)))
    c128 = lrelu(conv(c128, 64, strides=(1, 1), kernel_size=(3, 3)))

    c64 = lrelu(conv(c128, 128, strides=(2, 2), kernel_size=(4, 4)))
    c64 = lrelu(conv(c64, 128, strides=(1, 1), kernel_size=(3, 3)))

    c32 = lrelu(conv(c64, 256, strides=(2, 2), kernel_size=(4, 4)))
    c32 = lrelu(conv(c32, 256, strides=(1, 1), kernel_size=(3, 3)))

    c16 = lrelu(conv(c32, 512, strides=(2, 2), kernel_size=(4, 4)))
    c16 = lrelu(conv(c16, 512, strides=(1, 1), kernel_size=(3, 3)))

    c8 = lrelu(conv(c16, 512, strides=(2, 2), kernel_size=(4, 4)))
    c8 = lrelu(conv(c8, 512, strides=(1, 1), kernel_size=(3, 3)))

    op = conv(c8, 1, strides=(1, 1), kernel_size=(3, 3))

    return Model(inputs=ip, outputs=[c512, c256, c128, c64, c32, c16, op])


def main_gan(real_samples, fake_samples, discriminator):

    def visualize_grad(dy, dx):
        debug_grad = tf.gradients(dy, dx)[0]
        debug_grad -= tf.reduce_min(debug_grad, axis=[1, 2, 3], keep_dims=True)
        debug_grad /= tf.reduce_mean(debug_grad, axis=[1, 2, 3], keep_dims=True)
        debug_grad = (1 - debug_grad) * 0.5 + 0.5
        return debug_grad

    real_features = discriminator(real_samples)
    fake_features = discriminator(fake_samples)

    real_score = real_features[6]
    fake_score = fake_features[6]

    L1_loss = 0

    for l in [0, 1, 2, 3, 4, 5]:
        L1_loss += tf.reduce_mean(tf.abs(real_features[l] - fake_features[l]))

    G_loss = tf.reduce_mean(- fake_score)
    D_loss = tf.reduce_mean(- real_score + fake_score)

    GP = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(tf.gradients(fake_score, fake_samples)[0]), axis=[1, 2, 3])) - 1.))

    visulized_grad = visualize_grad(G_loss, fake_samples)

    return L1_loss + G_loss * 0.01, D_loss + GP * 10, visulized_grad