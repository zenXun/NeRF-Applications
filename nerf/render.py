import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm_notebook as tqdm

trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)

#reference: https://github.com/bmild/nerf
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def generate_video_frames(H, W, focal, model):
    frames = []
    for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
        c2w = pose_spherical(th, -30., 4.)
        rays_o, rays_d = utils.get_rays(H, W, focal, c2w[:3,:4])
        rgb, depth, acc = utils.render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        frames.append((255*np.clip(rgb,0,1)).astype(np.uint8))
    return frames 
