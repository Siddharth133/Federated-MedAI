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
# ==============================================
# 2. Federated Learning Implementation
# ==============================================

custom_objects = {
    'InstanceNormalization': InstanceNormalization,
    'ReflectionPadding2D': ReflectionPadding2D,
    'BioPhysicalLayer': BioPhysicalLayer,
    'NeuroSymbolicLayer': NeuroSymbolicLayer
}

with keras.utils.custom_object_scope(custom_objects):
    model = keras.models.load_model(r'..\backend\core\global_model\bpnsgm_ct_to_mri.h5')
    

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
    loss='mae',  # Or a custom GAN + L1 loss
    metrics=['mae', 'ssim']  # Add PSNR if needed
)

model.get_weights()


# # ==============================================
# # Federated Averaging Function
# # ==============================================

# def federated_averaging(models, weights=None):
#     """Average model weights from different hospitals"""
#     if weights is None:
#         weights = [1.0/len(models)] * len(models)
    
#     # Get trainable weights from each model
#     model_weights = [model.get_weights() for model in models]
    
#     # Average weights
#     avg_weights = []
#     for weights_list in zip(*model_weights):
#         avg_weights.append(
#             np.sum([w * weight for w, weight in zip(weights_list, weights)], axis=0))
    
#     # Create new model with averaged weights
#     global_model = keras.models.clone_model(models[0])
#     global_model.set_weights(avg_weights)
#     return global_model

# # ==============================================
# # Single Image Processing
# # ==============================================

# def load_and_preprocess_image(image_path, target_size=(256, 256)):
#     """Load and preprocess a single image"""
#     img = imread(image_path, as_gray=True)
#     img = resize(img, target_size, anti_aliasing=True)
#     img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
#     return img[np.newaxis, ..., np.newaxis].astype(np.float32)  # Add batch and channel dim

# # ==============================================
# # Manual Image Loading
# # ==============================================

# # Replace these paths with your actual single image paths
# CT_PATH = '/content/drive/MyDrive/trainA/ct1001.png'  # Change to your CT image path
# MRI_PATH = '/content/drive/MyDrive/trainB/mri1001.jpg' # Change to your MRI image path

# # Load single images
# ct_image = load_and_preprocess_image(CT_PATH)
# mri_image = load_and_preprocess_image(MRI_PATH)

# # ==============================================
# # Evaluation Function for Single Images
# # ==============================================

# def evaluate_single(model, ct_input, mri_target):
#     """Evaluate model on single image pair"""
#     # Generate prediction
#     pred_mri = model.predict(ct_input)
    
#     # Calculate metrics
#     mae = tf.keras.losses.MAE(mri_target, pred_mri).numpy().mean()
#     ssim = tf.image.ssim(mri_target, pred_mri, max_val=1.0).numpy().mean()
    
#     # Visual comparison
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 3, 1)
#     plt.imshow(ct_input[0, :, :, 0], cmap='gray')
#     plt.title('Input CT')
#     plt.axis('off')
    
#     plt.subplot(1, 3, 2)
#     plt.imshow(mri_target[0, :, :, 0], cmap='gray')
#     plt.title('Real MRI')
#     plt.axis('off')
    
#     plt.subplot(1, 3, 3)
#     plt.imshow(pred_mri[0, :, :, 0], cmap='gray')
#     plt.title('Predicted MRI')
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()
    
#     return {'MAE': mae, 'SSIM': ssim}

# # ==============================================
# # Main Execution with Single Images
# # ==============================================

# print("Evaluating Hospital A Model (CT to MRI)")
# hospital_a_metrics = evaluate_single(hospital_a_model, ct_image, mri_image)

# print("\nCreating Federated Global Model")
# global_model = federated_averaging([hospital_a_model])  # Add other models if available

# # Compile the global model
# global_model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
#     loss='mae',
#     metrics=['mae', 'ssim']
# )

# print("\nEvaluating Federated Global Model")
# global_metrics = evaluate_single(global_model, ct_image, mri_image)

# # Print comparison
# print("\nPerformance Comparison:")
# print(f"Hospital A Model - MAE: {hospital_a_metrics['MAE']:.4f}, SSIM: {hospital_a_metrics['SSIM']:.4f}")
# print(f"Federated Global Model - MAE: {global_metrics['MAE']:.4f}, SSIM: {global_metrics['SSIM']:.4f}")