# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:22:11 2019
@author: Steve Alejandro Avendaño García
"""
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
import numpy as np

class model:
            
    def conv_block(self, input_tensor, num_filters):
        encoder = self.Convolution(num_filters = num_filters, IN = input_tensor)
        encoder = layers.BatchNormalization(axis=-1)(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.SpatialDropout2D(0.2)(encoder)
        encoder = self.Convolution(num_filters = num_filters, IN = encoder)
        encoder = layers.BatchNormalization(axis=-1)(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.SpatialDropout2D(0.2)(encoder)
        return encoder

    def encoder_block(self, input_tensor, num_filters):
        encoder = self.conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    
        return encoder_pool, encoder

    def decoder_block(self, input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer = keras.initializers.he_uniform(seed=np.random.randint(1,100)))(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization(axis=-1)(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = self.Convolution(num_filters = num_filters, IN = decoder)
        decoder = layers.BatchNormalization(axis=-1)(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.SpatialDropout2D(0.2)(decoder)
        decoder = self.Convolution(num_filters = num_filters, IN = decoder)
        decoder = layers.BatchNormalization(axis=-1)(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.SpatialDropout2D(0.2)(decoder)
        return decoder
    
    def generalised_dice_loss(self, y_true, y_pred):
        """
        Function to calculate the Generalised Dice Loss defined in
    
        :param y_pred: the logits
        :param y_true: the segmentation ground truth
        
        :return: the loss
        """
        epsilon = tf.constant(value=1., dtype = 'float32')
        intersection1 = 2 * ( tf.reduce_sum( tf.multiply(y_true[:,:,:,0],y_pred[:,:,:,0])) )
        intersection2 = 2 * ( tf.reduce_sum( tf.multiply(y_true[:,:,:,1],y_pred[:,:,:,1])) )
        intersection3 = 2 * ( tf.reduce_sum( tf.multiply(y_true[:,:,:,2],y_pred[:,:,:,2])) ) 
    
        union1 = tf.reduce_sum(y_true[:,:,:,0] + y_pred[:,:,:,0]) 
        union2 = tf.reduce_sum(y_true[:,:,:,1] + y_pred[:,:,:,1])
        union3 = tf.reduce_sum(y_true[:,:,:,2] + y_pred[:,:,:,2]) 
    
        dice1 = tf.math.divide(intersection1 + epsilon, union1 + epsilon)
        dice2 = tf.math.divide(intersection2 + epsilon, union2 + epsilon)
        dice3 = tf.math.divide(intersection3 + epsilon, union3 + epsilon)
    
        # Compute weights: "the contribution of each label is divided equally"
        dice_coef = tf.reduce_mean([dice1, dice2, dice3])
        return 1 - dice_coef

    def cce_dice_loss(self, y_true, y_pred):
        loss = losses.categorical_crossentropy(y_true, y_pred) + self.generalised_dice_loss(y_true, y_pred)
        return loss
    
    def Unet(self, ):
        return self.model        
        
    def Convolution(self, num_filters ,IN):
        if self.mode == 'Dense':
            OUT = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer = keras.initializers.he_uniform(seed=np.random.randint(1,100)))(IN)
        elif self.mode == 'Separable':
            OUT = layers.SeparableConv2D(num_filters, (3, 3), padding='same', depth_multiplier=1, kernel_initializer = keras.initializers.he_uniform(seed=np.random.randint(1,100)))(IN)
        
        return OUT
        
    
    
    def __init__(self, mode = 'Dense'):
        """
        :param mode: can be Dense which is the conventional convolution operation
        or Separable, which stands for separable convolution
        """
        self.mode = mode
        
        """
        --------------------------------------------------------------------------
        """
        self.inputs1 = layers.Input(shape=(256,256,1))
        # 256
        
        encoder0_pool, encoder0 = self.encoder_block(self.inputs1, 64)
        # 128
        
        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 128)
        # 64
        
        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 256)
        # 32
        
        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 512)
        # 16
        
        center = self.conv_block(encoder3_pool, 1024)
        # center 16
        
        decoder3 = self.decoder_block(center, encoder3, 512)
        # 32
        
        decoder2 = self.decoder_block(decoder3, encoder2, 256)
        # 64
        
        decoder1 = self.decoder_block(decoder2, encoder1, 128)
        # 128
        
        decoder0 = self.decoder_block(decoder1, encoder0, 64)
        # 256
        
        self.outputs1 = layers.Conv2D(3, (1, 1), activation = 'softmax')(decoder0)
        
        self.model = models.Model(inputs=[self.inputs1], outputs=[self.outputs1])
        
        self.model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False), loss=self.cce_dice_loss, metrics=[self.generalised_dice_loss])
        

