import os
from functools import reduce

import tensorflow as tf
import numpy as np

from lib.math import *
from lib.rays import *

class GaussianRadianceField(tf.keras.Model):
    def __init__(self, data, 
                 num_layers=6, units=256, output=4,
                 num_samples=128):
        super().__init__()
        
        hwf = data['hwf']

        self.img_rgb = data['img_rgb']
        self.num_img = len(self.img_rgb)
        self.hwf = [int(hwf[0]), int(hwf[1]), hwf[2]]
        self.num_samples = num_samples
        self.f_cam_rays = data['f_cam_rays']
        self.f_img_rgb = data['f_img_rgb']

        self.sequence = []
        self.lie = Lie()
        
        self.pose = tf.zeros(6)
        self.pose = tf.Variable(tf.repeat(self.pose[tf.newaxis, ...], self.num_img, axis=0))
        
        for k in range(num_layers - 1):
            self.sequence.append(tf.keras.layers.Dense(units, activation=gaussian))
            
        self.sequence.append(tf.keras.layers.Dense(output, activation=None))
    
    def get_pose(self):
        #return poses
        return self.lie.se3_to_SE3(self.pose)
    
    def call(self, x):
        return reduce(lambda x, f : f(x), [x] + self.sequence)
    
    def compile(self):
        # params from paper for real-world scenes...
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, 
                                                                          decay_steps=200000, 
                                                                          decay_rate=0.5,
                                                                          staircase=False)
        self.pose_lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(3e-3, 
                                                                          decay_steps=200000, 
                                                                          decay_rate=1/300.,
                                                                          staircase=False)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.pose_optimizer = tf.keras.optimizers.Adam(learning_rate=self.pose_lr_sched)
    
    def save(self, path, opt=True):
        if not os.path.exists(path):
            os.makedirs(path)

        if opt:
            opt_path = os.path.join(path, 'opt.npy')
            p_opt_path = os.path.join(path, 'pose-opy.npy')

            opt_state = self.optimizer.get_weights()
            p_opt_state = self.pose_optimizer.get_weights()

            np.save(opt_path, opt_state)
            np.save(p_opt_path, p_opt_state)

        param_path = os.path.join(path, 'model.h5')
        pose_path = os.path.join(path, 'pose.npy')

        self.save_weights(param_path)
        np.save(pose_path, self.pose)

        tf.print(f'Model saved to {path}')

    def load(self, path, opt=True, iters=2):
        if not os.path.exists(path):
            raise Exception(f'{path} does not exist')

        if opt:
            opt_path = os.path.join(path, 'opt.npy')
            p_opt_path = os.path.join(path, 'pose-opy.npy')
            opt_state = np.load(opt_path, allow_pickle=True)
            p_opt_state = np.load(p_opt_path, allow_pickle=True)

            from train import train

            tf.print(f'Training for {iters} iteration(s) to initialize optimizer state.')
            train(self, iters, val_idx='rand')
            tf.print('Loading optimizer state...')

            self.optimizer.set_weights(opt_state)
            self.pose_optimizer.set_weights(p_opt_state)

        param_path = os.path.join(path, 'model.h5')
        pose_path = os.path.join(path, 'pose.npy')
        pose = np.load(pose_path, allow_pickle=True)

        self.load_weights(param_path)
        self.pose = tf.Variable(pose)

        tf.print(f'Model loaded to {path}')

    def sample_rays(self, n, batch_size):
        '''Sample data from the first n images of a certain batch size.'''
        n_rays = tf.shape(self.f_cam_rays)[0]
        idx = tf.random.uniform([n, batch_size], minval=0, maxval=n_rays, dtype=tf.int32)
        
        app_idx = tf.range(n, dtype=tf.int32)[:, tf.newaxis]
        app_idx = tf.broadcast_to(app_idx, (n, batch_size))
        
        rgb_idx = tf.reshape(tf.stack([app_idx, idx], axis=-1), [-1, 2])
        
        return tf.gather(self.f_cam_rays, idx), tf.gather_nd(self.f_img_rgb, rgb_idx)

    def get_slice(self, batch_size, n=-1):
        '''Get a batch.'''
        if n == -1:
            n = self.num_img

        rays, rgb = self.sample_rays(n, batch_size)
        r = get_rays(rays, self.get_pose()[:n, ...])
        
        return tf.reshape(r[0], [-1, 3]), tf.reshape(r[1], [-1, 3]), rgb


    def process_raw(self, raw, z_vals):
        '''Process raw output to rgb, depth, and acc maps.

        From https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb
        '''
        # Compute opacities and colors
        sigma_a = tf.nn.relu(raw[..., 3])
        rgb = tf.math.sigmoid(raw[..., :3]) 

        # Do volume rendering
        dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], tf.broadcast_to([1e10], tf.shape(z_vals[..., :1]))], -1) 
        alpha = 1. - tf.exp(-sigma_a * dists) 
        weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

        rgb_map = tf.reduce_sum(weights[...,None] * rgb, -2) 
        depth_map = tf.reduce_sum(weights * z_vals, -1) 
        acc_map = tf.reduce_sum(weights, -1)

        return rgb_map, depth_map, acc_map

    def batch(self, batch):
        _, r_o, r_d = tf.unstack(batch, axis=1)
        coords, t = get_coords(r_o, r_d, self.num_samples)
        
        r_d_exp = tf.repeat(tf.reshape(r_d, [-1,3])[:, tf.newaxis], tf.shape(coords)[1], axis=1)
        model_in = tf.concat([coords, r_d_exp], axis=-1)
        
        return self.process_raw(self(model_in), t)

    def render(self, img_number):
        tf.print(f'Rendering image index {img_number}')
        
        pose = self.get_pose()[img_number]
        rgb = self.img_rgb[img_number]
        r_o, r_d = get_scene_rays(*self.hwf, pose)

        raw_dataset = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.stack([rgb, r_o, r_d], axis=2), [-1, 3, 3]))
        dataset = raw_dataset.batch(4096)
        
        rendered_img = []
        for batch in dataset:
            pred_rgb, depth, acc = self.batch(batch)
            rendered_img.append(pred_rgb)

        rendered_img = tf.reshape(tf.concat(rendered_img, axis=0), rgb.shape)
        
        return rendered_img