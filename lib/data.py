import os
import pathlib

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from lib.rays import *

def load_data(data_path):
  def img_transform(img, img_ratio=8):
      img_shape = [tf.shape(img)[0] // img_ratio, tf.shape(img)[1] // img_ratio]
      
      img = tf.image.resize(img, img_shape,
                            method='bicubic', antialias=True, 
                            preserve_aspect_ratio=True) / 255.
      
      img = tf.clip_by_value(img, 0., 1.)
      
      return img

  img_list = list(pathlib.Path(data_path).glob('./images/*'))
  img_rgb = [img_transform(tf.image.decode_image(tf.io.read_file(str(path)))) for path in img_list]

  num_img = len(img_list)
  f_img_rgb = tf.reshape(tf.stack(img_rgb, axis=0), [num_img, -1, 3])

  plt.imshow(img_rgb[0])
  plt.show()

  poses_arr = np.load(os.path.join(data_path, 'poses_bounds.npy'))
  poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
  bds = poses_arr[:, -2:].transpose([1,0])

  # Correct rotation matrix ordering and move variable dim to axis 0
  poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
  poses = np.moveaxis(poses, -1, 0).astype(np.float32)
  bds = np.moveaxis(bds, -1, 0).astype(np.float32)

  hwf = poses[0, :3, -1]
  poses = poses[:, :3, :4]

  hwf /= 8
  cam_rays = get_local_rays(int(hwf[0]), int(hwf[1]), hwf[2])
  f_cam_rays = tf.reshape(cam_rays, [-1, 3])

  data = {'hwf': hwf, 'img_rgb': img_rgb, 'f_img_rgb': f_img_rgb, 'f_cam_rays': f_cam_rays}

  return data