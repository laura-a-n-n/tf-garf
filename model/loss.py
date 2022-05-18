import tensorflow as tf

def generalized_mean_norm(val, gt=None, p=2, axis=-1):
    '''Compute the mean of ||input||^p, where ||.|| denotes Lp norm.
    
    Arguments:
        val: tf.Tensor -- input to be normed and averaged
        
    Keyword arguments:
        gt: tf.Tensor -- ground truth; if not None, subtracted with val
        p: int -- order of norm
        axis: int -- axis to norm over
    '''
    if gt is not None:
        val = val - gt
        
    val = tf.norm(val, ord=p, axis=axis)
    
    return tf.math.reduce_mean(val**p)