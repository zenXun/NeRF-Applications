import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

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

# reference: https://github.com/bmild/nerf
def get_rays(H, W, focal, c2w):
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d))
    return rays_o, rays_d

# reference: https://github.com/bmild/nerf
def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):

    def batchify(fn, chunk=1024*32):
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    # Compute 3D query points
    z_vals = tf.linspace(near, far, N_samples)
    if rand:
      z_vals += tf.random.uniform(list(rays_o.shape[:-1]) + [N_samples]) * (far-near)/N_samples
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    # Run network
    pts_flat = tf.reshape(pts, [-1,3])
    pts_flat = embed_fn(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    raw = tf.reshape(raw, list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    sigma_a = tf.nn.relu(raw[...,3])
    rgb = tf.math.sigmoid(raw[...,:3])

    # Do volume rendering
    dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], tf.broadcast_to([1e10], z_vals[...,:1].shape)], -1)
    alpha = 1.-tf.exp(-sigma_a * dists)
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    rgb_map = tf.reduce_sum(weights[...,None] * rgb, -2)
    depth_map = tf.reduce_sum(weights * z_vals, -1)
    acc_map = tf.reduce_sum(weights, -1)

    return rgb_map, depth_map, acc_map

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
        rays_o, rays_d = get_rays(H, W, focal, train_pose)
        with tf.GradientTape() as tape:
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=Nc, rand=True)
            loss = tf.reduce_mean(tf.square(rgb - train_image))
            print("PSNR is ", -10. * tf.math.log(loss) / tf.math.log(10.))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables)

