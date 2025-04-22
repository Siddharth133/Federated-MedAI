import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

# ==============================================
# 1. Corrected Custom Layers (with proper config)
# ==============================================

class InstanceNormalization(layers.Layer):
    """Custom Instance Normalization Layer with proper config handling"""
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)  # This handles trainable, name, etc.
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config

class ReflectionPadding2D(layers.Layer):
    """Custom Reflection Padding Layer with proper config handling"""
    def __init__(self, padding=(1, 1), **kwargs):
        super().__init__(**kwargs)  # Handle base layer arguments
        self.padding = tuple(padding)

    def call(self, input_tensor):
        pw, ph = self.padding
        return tf.pad(input_tensor, [[0, 0], [ph, ph], [pw, pw], [0, 0]], mode='REFLECT')

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config


class BioPhysicalLayer(layers.Layer):
    """Enforces tissue physics between CT and MRI"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Handle base layer arguments

    def call(self, inputs):
        ct, mri = inputs
        # CT Hounsfield Units to tissue masks
        bone_mask = tf.cast(ct > 0.7, tf.float32)       # Bones (HU > 700)
        tissue_mask = tf.cast(ct > -0.1, tf.float32)    # Soft tissues

        # Physics constraints
        mri_bone = mri * bone_mask * 0.8     # Bones darker in MRI
        mri_tissue = mri * tissue_mask * 1.2  # Soft tissue brighter
        return mri_bone + mri_tissue + (1.0 - tf.maximum(bone_mask, tissue_mask)) * mri


class NeuroSymbolicLayer(layers.Layer):
    """Incorporates anatomical rules"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Handle base layer arguments

    def build(self, input_shape):
        self.ventricle_kernel = self.add_weight(
            name='ventricle_kernel',
            shape=(3, 3, 1, 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True)

    def call(self, inputs):
        ct, mri = inputs
        # Ventricle detection (learned)
        ventricle_map = tf.nn.conv2d(ct, self.ventricle_kernel, strides=1, padding='SAME')
        ventricle_map = tf.sigmoid(ventricle_map * 10)

        # Anatomical adjustment
        return mri * (1.0 - ventricle_map * 0.3)  # Darken ventricles