import tensorflow as tf

def gaussian(x, c=0, sigma=.1):
    '''Gaussian centered at c with standard deviation sigma.'''
    return tf.math.exp(-(x-c)**2 / (2 * sigma**2))

class Lie():
    '''SE(3) Lie algebra in TensorFlow.
    
    Translated from https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/camera.py
    '''
    
    def skew_symmetric(self, w):
        '''Compute skew-symmetric matrix.'''
        w_0, w_1, w_2 = tf.unstack(w, axis=-1)
        O = tf.zeros_like(w_0)
        wx = tf.stack([tf.stack([O, -w_2, w_1], axis=-1),
                       tf.stack([w_2, O, -w_0], axis=-1),
                       tf.stack([-w_1, w_0, O], axis=-1)], axis=-2)
        
        return wx

    def taylor_A(self, x, n=10):
        '''Compute Taylor expansion of sin(x)/x.'''
        
        ans = tf.zeros_like(x)
        denom = 1.
        
        for i in range(n+1):
            if i > 0: 
                denom *= (2*i) * (2*i+1)
            ans += (-1)**i * x**(2 * i) / denom
            
        return ans
    
    def taylor_B(self, x, n=10):
        '''Compute Taylor expansion of (1-cos(x))/x**2'''
        
        ans = tf.zeros_like(x)
        denom = 1.
        
        for i in range(n+1):
            denom *= (2*i+1) * (2*i+2)
            ans += (-1)**i * x**(2 * i) / denom
            
        return ans
    
    def taylor_C(self, x, n=10):
        '''Compute Taylor expansion of (x-sin(x))/x**3.'''
        
        ans = tf.zeros_like(x)
        denom = 1.
        
        for i in range(n+1):
            denom *= (2*i+2) * (2*i+3)
            ans += (-1)**i * x**(2 * i) / denom
        
        return ans
    
    def se3_to_SE3(self, wu):
        '''Map 6D representation to 3x4 R|t matrix.'''
        w, u = tf.split(wu, [3, 3], axis=-1)
        wx = self.skew_symmetric(w)
        
        if tf.math.reduce_any(w != 0.):
            theta = tf.linalg.norm(w, axis=-1)[..., None, None]
        else:
            theta = tf.zeros((tf.shape(w)[0], 1, 1))
        
        I = tf.eye(3, dtype=tf.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        
        Rt = tf.concat([R, (V @ u[...,None])], axis=-1)
        
        return Rt