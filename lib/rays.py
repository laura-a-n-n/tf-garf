import tensorflow as tf
import numpy as np

def get_local_rays(H, W, focal):
    '''Get camera space rays from a pinhole camera, to be transformed to world space.'''
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    
    return dirs

def get_rays(dirs, c2w):
    '''Transform camera space rays to world space.'''
    rays_d = tf.reduce_sum(dirs[..., tf.newaxis, :] * c2w[..., tf.newaxis, :3, :3], -1)
    rays_o = tf.broadcast_to(c2w[..., tf.newaxis, :3, -1], tf.shape(rays_d))
    
    # normalize
    rays_d = tf.linalg.normalize(rays_d, axis=-1)[0]
    
    return rays_o, rays_d

def get_scene_rays(H, W, focal, c2w):
    '''Get ray origins, directions from a pinhole camera.'''
    dirs = get_local_rays(H, W, focal)
    rays_o, rays_d = get_rays(dirs, c2w)
    
    return rays_o, rays_d

def get_coords(r_o, r_d, num_samples, near=2., far=6.):
    '''Sample along rays between near and far bounds.'''
    t = tf.linspace(near, far, num_samples)
    coords = r_o[..., tf.newaxis, :] + t[tf.newaxis, :, tf.newaxis] * r_d[..., tf.newaxis, :]

    return coords, t

def create_spiral_poses(radii, focus_depth, n_poses=10):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3
    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path
    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)
