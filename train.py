import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

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

def train(model, epochs, batch_size=64, model_name='saved_model',
          val=True, val_idx='rand', val_steps=300, chk_steps=10000,
          notebook=True, out_path='out.png'):
    val_type = val_idx
    
    for epoch in range(epochs):
        if val_type == 'rand':
            val_idx = int(tf.random.uniform((), minval=0, maxval=model.num_img))
        
        if val and (epoch % val_steps == 0):
            cond_clear(notebook)

            tf.print(f'Step {epoch}')

            rendered_img = model.render(val_idx)

            plt.imshow(rendered_img)

            if notebook:
                plt.show()
                plt.imshow(model.img_rgb[val_idx])
                plt.show()
            else:
                plt.savefig(out_path)
                plt.show() # clear mem
            
            if (epoch > 0) and (epoch % chk_steps == 0):
                j = str(epoch // chk_steps)
                model_name += j
                pose_name += j
            
                model.save(model_name)
        
        with tf.GradientTape(persistent=True) as tape:
            loss = train_step(model, batch_size)
        
        model.optimizer.minimize(loss, var_list=model.trainable_variables[:-1], tape=tape)
        model.pose_optimizer.minimize(loss, var_list=model.pose, tape=tape)
        
        # clear memory
        tape = None
        
        tf.print(f'Loss {loss:.5f}')
