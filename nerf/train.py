import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

def positional_encoding(pos_vec, L):
    enc = [pos_vec]
    for i in range(L):
        for fn in [tf.sin, tf.cos]:
            enc.append(fn(2.**i * pos_vec))
    return tf.concat(enc, -1)

def _dense(W=256, act):
    return tf.keras.layers.Dense(W, activation=act)

def model(D=8, W=256, L):
    relu = tf.keras.layers.ReLU()
    inputs = tf.keras.Input(shape=(3 + 6 * L))
    outputs = inputs
    for layer in range(D):
        outputs = _dense(W, relu)(outputs)
        if layer == 4:
            outputs = tf.concat([outputs, inputs], -1)
    outputs = _dense(4, None)(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def train(images, poses, focal, L, Nc, iterations):
    H, W = images.shape[1: 3]
    test_image, test_pose = images[-1], poses[-1]
    train_images, train_poses = images[:-1], poses[:-1]
    model = model(L=L)
    optimizer = tf.keras.optimizers.Adam(5e-4)
    for i in range(iterations):
        train_index = np.random.randint(train_images.shape[0])
        train_image = train_images[train_index]
        train_pose = train_poses[train_index]
        rays_o, rays_d = utils.get_rays(H, W, focal, train_pose)
        with tf.GradientTape() as tape:
            rgb, depth, acc = utils.render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=Nc, rand=True)
            loss = tf.reduce_mean(tf.square(rgb - train_image))
            print("PSNR is ", -10. * tf.math.log(loss) / tf.math.log(10.))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables)
    return model
