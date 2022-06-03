import os

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from lib.rays import create_spiral_poses
from model.garf import GaussianRadianceField
from model.loss import generalized_mean_norm

from IPython.display import clear_output

def cond_clear(notebook):
    if notebook:
        clear_output(wait=True)
    else:
        if os.name == 'nt':
            _ = os.system('cls')
        else:
            _ = os.system('clear')

def train_step(model, batch_size):
    r_o, r_d, rgb = model.get_slice(batch_size)
    batch = tf.stack([rgb, r_o, r_d], axis=1)

    pred_rgb, depth, acc = model.batch(batch)
    loss = generalized_mean_norm(pred_rgb, gt=rgb)
    return loss

def train(model, epochs, batch_size=64, model_name='saved_model', init_epoch=0,
          val=True, val_idx='rand', val_steps=50, chk_steps=10000,
          notebook=True, save=False, out_path='out.png', overwrite=False,
          spiral_n_steps=60, spiral_axes=None, spiral_depth=1.2):
    val_type = val_idx
    spiral = val_type == 'spiral'
    
    if val and spiral:
        spiral_steps = spiral_n_steps * val_steps
        n_circ = epochs // spiral_steps
        n_poses = epochs // val_steps
        
        if spiral_axes is None:
            spiral_axes = np.array([.2, .2, .05])
        
        poses_all = tf.cast(create_spiral_poses(spiral_axes, spiral_depth, n_poses=n_poses, n_circ=n_circ), tf.float32)
    
    for epoch in range(init_epoch, epochs):
        if val and not spiral and val_type == 'rand':
            val_idx = int(tf.random.uniform((), minval=0, maxval=model.num_img))
        
        if val and (epoch % val_steps == 0):
            cond_clear(notebook)

            tf.print(f'Step {epoch}')

            if not spiral:
                rendered_img = model.render(val_idx)
            else:
                rendered_img = model.render(0, pose=poses_all[epoch // val_steps])

            plt.imshow(rendered_img)

            if not save:
                plt.show()
                plt.imshow(model.img_rgb[val_idx])
                plt.show()
            else:
                if overwrite:
                    plt.savefig(out_path)
                else:
                    N = len(str(epochs))
                    name = list('0' * N)
                    name[-len(str(epoch)):N] = str(epoch)
                    
                    plt.savefig(out_path[:-4] + (''.join(name)) + out_path[-4:])
                
                plt.show() # clear mem
            
            if (epoch > 0) and (epoch % chk_steps == 0):
                j = str(epoch // chk_steps)
                model.save(model_name + j)
        
        with tf.GradientTape(persistent=True) as tape:
            loss = train_step(model, batch_size)
        
        model.optimizer.minimize(loss, var_list=model.trainable_variables[:-1], tape=tape)
        model.pose_optimizer.minimize(loss, var_list=model.pose, tape=tape)
        
        # clear memory
        tape = None
        
        tf.print(f'Loss {loss:.5f}')
