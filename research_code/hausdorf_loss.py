import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.utils.extmath import cartesian
import math

## weighted hausdorff distance based on "Locating Objects Without Bounding Boxes" (https://arxiv.org/pdf/1806.07564.pdf)
## based implementation off of original author's PyTorch implementation (https://github.com/HaipengXiong/weighted-hausdorff-loss)

'''
Created on Oct 12, 2018
@author: daniel
'''

import tensorflow as tf
import keras.backend as K
import numpy as np
from sklearn.utils.extmath import cartesian
import math

## weighted hausdorff distance based on "Locating Objects Without Bounding Boxes" (https://arxiv.org/pdf/1806.07564.pdf)
## based implementation off of original author's PyTorch implementation (https://github.com/HaipengXiong/weighted-hausdorff-loss)

class HausdorffLoss:
    def __init__(self, W=128, H=128, alpha = 2):
        self.W = W
        self.H = H
        self.alpha = alpha
        self.all_img_locations = tf.convert_to_tensor(cartesian([np.arange(W), np.arange(H)]), dtype=tf.float32)
        self.max_dist = math.sqrt(W**2 + H**2)


    def cdist (self,A, B):  
    
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)
        
        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])
        
        # return pairwise euclidead difference matrix
        D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
        return D

    def unstack_and_stack(self, tensor):
        chan_list = tf.unstack(tensor, num=6, axis=-1)
        return tf.stack(chan_list, axis=0)


    def multilabel_hausdorf(self, y_true, y_pred):
        y_true = self.unstack_and_stack(y_true)
        y_pred = self.unstack_and_stack(y_pred)
        batched_losses = tf.map_fn(lambda x: 
                self.weighted_hausdorff_distance(x[0], x[1]), 
                (y_true, y_pred), 
                dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    def weighted_hausdorff_distance(self,y_true, y_pred, multilabel=True):
            all_img_locations = self.all_img_locations
            W = self.W
            H = self.H
            alpha = self.alpha
            max_dist = self.max_dist
            eps = 1e-6
    
            y_true = K.reshape(y_true, [W,H])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype = tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
        
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p,axis=-1), num_gt_points))
        
            d_matrix = self.cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))
        
        
            d_div_p = K.min((d_matrix + eps) / (p_replicated**alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0) 
        
            return term_1 + term_2
            
        
    def hausdorff_loss(self,y_true, y_pred):
        batched_losses = tf.map_fn(lambda x: 
                self.multilabel_hausdorf(x[0], x[1]), 
                (y_true, y_pred), 
                dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))


""" class HausdorffLoss:
    def __init__(self, alpha = 2):
        # self.W = W
        # self.H = H
        self.alpha = alpha
       #self.all_img_locations = tf.convert_to_tensor(cartesian([np.arange(W), np.arange(H)]), dtype=tf.float32)
        # self.max_dist = math.sqrt(W**2 + H**2)


    def cdist (self,A, B):  
    
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)
        
        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])
        
        # return pairwise euclidead difference matrix
        D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
        return D
          
    @staticmethod
    def cartesian_product(W, H):
        a = tf.range(W)
        b = tf.range(H)

        tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0]])  
        tile_a = tf.expand_dims(tile_a, 2) 
        tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1]) 
        tile_b = tf.expand_dims(tile_b, 2) 

        cartesian_product = tf.concat([tile_a, tile_b], axis=2) 
        return cartesian_product
        
    def weighted_hausdorff_distance(self,y_true, y_pred):
            # all_img_locations = self.all_img_locations
            W = tf.shape(y_true)[1]# y_true.shape[1]
            H = tf.shape(y_true)[2]
            print(H, W)
            all_img_locations = self.cartesian_product(W, H) # tf.convert_to_tensor(cartesian([np.arange(W), np.arange(H)]), dtype=tf.float32)

            alpha = self.alpha
            max_dist = tf.math.sqrt(K.pow(W, 2) + K.pow(H, 2)) #self.max_dist 
            eps = 1e-6
    
            #y_true = K.reshape(y_true, [W,H])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype = tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
        
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p,axis=-1), num_gt_points))
        
            d_matrix = self.cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))
        
        
            d_div_p = K.min((d_matrix + eps) / (p_replicated**alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0) 
        
            return term_1 + term_2
            
        
    def hausdorff_loss(self,y_true, y_pred):
        print(y_true.shape)
        batched_losses = tf.map_fn(lambda x: 
                self.weighted_hausdorff_distance(x[0], x[1]), 
                (y_true, y_pred), 
                dtype=tf.float32)
        return K.mean(tf.stack(batched_losses)) """