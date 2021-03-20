import numpy as np
import imageio
import render
import train

if __name__ == "__main__":
    data = np.load('../data/tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = images.shape[1:3]
    for L in [6, 10]:
        for Nc in [64, 128]:
            model = train.train(images, poses, focal, L, Nc, 2000)
            frames = render.generate_video_frames(H, W, focal, model, Nc) 
            video_name = 'video_' + L + '_' + Nc + '.mp4'
            imageio.minwrite(video_name, frames, fps=30, quality=7)
