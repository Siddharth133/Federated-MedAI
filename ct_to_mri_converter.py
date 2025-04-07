import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# Configuration
IMAGE_SHAPE = (256, 256, 3)
LATENT_SPACE = 256
KERNEL = 3
PADDING = 'same'
FILTER = 16
STRIDES = 1

def module(inputs, filter, kernel, padding, strides, activation, use_norm, dilation_rate):
    """Basic convolution module with 3 conv layers and group normalization"""
    x = inputs
    x = layers.Conv2D(filter, kernel_size=kernel, padding=padding, strides=strides,
                    dilation_rate=dilation_rate)(x)
    if activation == 'LeakyReLU':
        x = layers.LeakyReLU(negative_slope=0.2)(x)
    else:
        x = layers.Activation(activation)(x)
    if use_norm:
        x = layers.GroupNormalization(groups=1)(x)
    x = layers.Conv2D(filter, kernel_size=kernel, padding=padding, strides=strides,
                      dilation_rate=dilation_rate)(x)
    if activation == 'LeakyReLU':
        x = layers.LeakyReLU(negative_slope=0.2)(x)
    else:
        x = layers.Activation(activation)(x)
    if use_norm:
        x = layers.GroupNormalization(groups=1)(x)
    x = layers.Conv2D(filter, kernel_size=kernel, padding=padding, strides=strides,
                      dilation_rate=dilation_rate)(x)
    if activation == 'LeakyReLU':
        x = layers.LeakyReLU(negative_slope=0.2)(x)
    else:
        x = layers.Activation(activation)(x)
    if use_norm:
        x = layers.GroupNormalization(groups=1)(x)
    return x

def convolution(inputs, filters, kernel, padding, strides, activation, use_norm, dilation_rate, name_prefix=''):
    """Convolution block with residual connection"""
    x = inputs
    x = module(x, filters, kernel, padding, strides, activation, use_norm, dilation_rate)
    y = layers.Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(inputs)
    if activation == 'LeakyReLU':
        y = layers.LeakyReLU(negative_slope=0.2)(y)
    else:
        y = layers.Activation(activation)(y)
    if use_norm:
        y = layers.GroupNormalization(groups=1)(y)
    
    x = layers.Maximum(name=f'maximum_{name_prefix}' if name_prefix else None)([x, y])
    return x

def encoder(inputs, filters, padding, strides, activation, kernel, use_norm, dilation_rate):
    """Encoder block with maxpooling"""
    conv = convolution(inputs, filters, kernel, padding, strides, activation, use_norm, dilation_rate)
    return layers.MaxPooling2D()(conv), conv

def decoder(inputs, skip, filters, padding, strides, kernel, activation, use_norm, dilation_rate):
    """Decoder block with skip connections"""
    x = layers.Conv2DTranspose(filters, kernel_size=kernel, padding=padding,
                              strides=2, activation=activation)(inputs)
    x = layers.Maximum()([x, skip])
    x = convolution(x, filters, kernel, padding, strides, activation, use_norm, dilation_rate)
    return x

def sampling(args):
    """Reparameterization trick for VAE"""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_generator():
    """Build the U-Net generator model exactly matching the architecture"""
    inputs = layers.Input(shape=IMAGE_SHAPE, name='source')
    
    # First block - 16 channels
    x = layers.Conv2D(FILTER, kernel_size=KERNEL, padding=PADDING, name='conv2d')(inputs)
    x = layers.Activation('relu', name='activation')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization')(x)
    
    x = layers.Conv2D(FILTER, kernel_size=KERNEL, padding=PADDING, name='conv2d_1')(x)
    x = layers.Activation('relu', name='activation_1')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_1')(x)
    
    x = layers.Conv2D(FILTER, kernel_size=KERNEL, padding=PADDING, name='conv2d_2')(x)
    x = layers.Activation('relu', name='activation_2')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_2')(x)
    
    # Residual connection for block 1
    res = layers.Conv2D(FILTER, kernel_size=1, padding=PADDING, name='conv2d_3')(inputs)
    res = layers.GroupNormalization(groups=1, name='group_normalization_3')(res)
    
    block1_output = layers.Maximum(name='maximum_block1')([x, res])
    
    # Block 2 - 32 channels
    x = layers.MaxPooling2D(name='max_pooling2d_block1')(block1_output)
    
    x = layers.Conv2D(FILTER*2, kernel_size=KERNEL, padding=PADDING, name='conv2d_4')(x)
    x = layers.Activation('relu', name='activation_3')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_4')(x)
    
    x = layers.Conv2D(FILTER*2, kernel_size=KERNEL, padding=PADDING, name='conv2d_5')(x)
    x = layers.Activation('relu', name='activation_4')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_5')(x)
    
    x = layers.Conv2D(FILTER*2, kernel_size=KERNEL, padding=PADDING, name='conv2d_6')(x)
    x = layers.Activation('relu', name='activation_5')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_6')(x)
    
    # Residual connection for block 2
    res = layers.Conv2D(FILTER*2, kernel_size=1, padding=PADDING, name='conv2d_7')(
        layers.MaxPooling2D(name='max_pooling2d_res1')(block1_output))
    res = layers.GroupNormalization(groups=1, name='group_normalization_7')(res)
    
    block2_output = layers.Maximum(name='maximum_block2')([x, res])
    
    # Block 3 - 64 channels
    x = layers.MaxPooling2D(name='max_pooling2d_block2')(block2_output)
    
    x = layers.Conv2D(FILTER*4, kernel_size=KERNEL, padding=PADDING, name='conv2d_8')(x)
    x = layers.Activation('relu', name='activation_6')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_8')(x)
    
    x = layers.Conv2D(FILTER*4, kernel_size=KERNEL, padding=PADDING, name='conv2d_9')(x)
    x = layers.Activation('relu', name='activation_7')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_9')(x)
    
    x = layers.Conv2D(FILTER*4, kernel_size=KERNEL, padding=PADDING, name='conv2d_10')(x)
    x = layers.Activation('relu', name='activation_8')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_10')(x)
    
    # Residual connection for block 3
    res = layers.Conv2D(FILTER*4, kernel_size=1, padding=PADDING, name='conv2d_11')(
        layers.MaxPooling2D(name='max_pooling2d_res2')(block2_output))
    res = layers.GroupNormalization(groups=1, name='group_normalization_11')(res)
    
    block3_output = layers.Maximum(name='maximum_block3')([x, res])
    
    # Block 4 - 128 channels
    x = layers.MaxPooling2D(name='max_pooling2d_block3')(block3_output)
    
    x = layers.Conv2D(FILTER*8, kernel_size=KERNEL, padding=PADDING, name='conv2d_12')(x)
    x = layers.Activation('relu', name='activation_9')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_12')(x)
    
    x = layers.Conv2D(FILTER*8, kernel_size=KERNEL, padding=PADDING, name='conv2d_13')(x)
    x = layers.Activation('relu', name='activation_10')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_13')(x)
    
    x = layers.Conv2D(FILTER*8, kernel_size=KERNEL, padding=PADDING, name='conv2d_14')(x)
    x = layers.Activation('relu', name='activation_11')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_14')(x)
    
    # Residual connection for block 4
    res = layers.Conv2D(FILTER*8, kernel_size=1, padding=PADDING, name='conv2d_15')(x)
    res = layers.GroupNormalization(groups=1, name='group_normalization_15')(res)
    
    block4_output = layers.Maximum(name='maximum_block4')([x, res])
    
    # Block 5 - 256 channels
    x = layers.MaxPooling2D(name='max_pooling2d_block4')(block4_output)
    
    x = layers.Conv2D(FILTER*16, kernel_size=KERNEL, padding=PADDING, name='conv2d_16')(x)
    x = layers.Activation('relu', name='activation_12')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_16')(x)
    
    x = layers.Conv2D(FILTER*16, kernel_size=KERNEL, padding=PADDING, name='conv2d_17')(x)
    x = layers.Activation('relu', name='activation_13')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_17')(x)
    
    x = layers.Conv2D(FILTER*16, kernel_size=KERNEL, padding=PADDING, name='conv2d_18')(x)
    x = layers.Activation('relu', name='activation_14')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_18')(x)
    
    # Residual connection for block 5
    res = layers.Conv2D(FILTER*16, kernel_size=1, padding=PADDING, name='conv2d_19')(x)
    res = layers.GroupNormalization(groups=1, name='group_normalization_19')(res)
    
    block5_output = layers.Maximum(name='maximum_block5')([x, res])
    
    # Block 6 - 512 channels (bottleneck)
    x = layers.MaxPooling2D(name='max_pooling2d_block5')(block5_output)
    
    x = layers.Conv2D(FILTER*32, kernel_size=KERNEL, padding=PADDING, name='conv2d_20')(x)
    x = layers.Activation('relu', name='activation_15')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_20')(x)
    
    x = layers.Conv2D(FILTER*32, kernel_size=KERNEL, padding=PADDING, name='conv2d_21')(x)
    x = layers.Activation('relu', name='activation_16')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_21')(x)
    
    x = layers.Conv2D(FILTER*32, kernel_size=KERNEL, padding=PADDING, name='conv2d_22')(x)
    x = layers.Activation('relu', name='activation_17')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_22')(x)
    
    # Residual connection for block 6
    res = layers.Conv2D(FILTER*32, kernel_size=1, padding=PADDING, name='conv2d_23')(x)
    res = layers.GroupNormalization(groups=1, name='group_normalization_23')(res)
    
    x = layers.Maximum(name='maximum_block6')([x, res])
    
    # Decoder path with skip connections
    # Upsampling and skip connection with block 5
    x = layers.Conv2DTranspose(FILTER*16, kernel_size=KERNEL, strides=2, padding=PADDING, name='conv2d_transpose_1')(x)
    x = layers.Maximum(name='maximum_up_block5')([x, block5_output])
    x = layers.Activation('relu', name='activation_up_1')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_up_1')(x)
    
    # Upsampling and skip connection with block 4
    x = layers.Conv2DTranspose(FILTER*8, kernel_size=KERNEL, strides=2, padding=PADDING, name='conv2d_transpose_2')(x)
    x = layers.Maximum(name='maximum_up_block4')([x, block4_output])
    x = layers.Activation('relu', name='activation_up_2')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_up_2')(x)
    
    # Upsampling and skip connection with block 3
    x = layers.Conv2DTranspose(FILTER*4, kernel_size=KERNEL, strides=2, padding=PADDING, name='conv2d_transpose_3')(x)
    x = layers.Maximum(name='maximum_up_block3')([x, block3_output])
    x = layers.Activation('relu', name='activation_up_3')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_up_3')(x)
    
    # Upsampling and skip connection with block 2
    x = layers.Conv2DTranspose(FILTER*2, kernel_size=KERNEL, strides=2, padding=PADDING, name='conv2d_transpose_4')(x)
    x = layers.Maximum(name='maximum_up_block2')([x, block2_output])
    x = layers.Activation('relu', name='activation_up_4')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_up_4')(x)
    
    # Upsampling and skip connection with block 1
    x = layers.Conv2DTranspose(FILTER, kernel_size=KERNEL, strides=2, padding=PADDING, name='conv2d_transpose_5')(x)
    x = layers.Maximum(name='maximum_up_block1')([x, block1_output])
    x = layers.Activation('relu', name='activation_up_5')(x)
    x = layers.GroupNormalization(groups=1, name='group_normalization_up_5')(x)
    
    # Output
    outputs = layers.Conv2D(3, kernel_size=1, padding=PADDING, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='xTOy')
    return model

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess a single image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def convert_ct_to_mri(ct_image_path, output_path=None):
    """Convert a CT scan to an MRI-like image"""
    # Load and preprocess CT image
    ct_image = load_and_preprocess_image(ct_image_path)
    
    # Build and load the model
    model = build_generator()
    
    # Load weights
    weights_path = os.path.join("weights", "g_target_5_28.h5")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    
    try:
        model.load_weights(weights_path)
        print("Successfully loaded weights")
    except Exception as e:
        raise Exception(f"Error loading weights: {str(e)}")
    
    # Generate MRI-like image
    mri_image = model.predict(ct_image)
    
    # Post-process the output
    mri_image = mri_image[0]  # Remove batch dimension
    
    # Convert to uint8
    mri_image = (mri_image * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    mri_image = cv2.cvtColor(mri_image, cv2.COLOR_RGB2BGR)
    
    # Save or return the result
    if output_path:
        cv2.imwrite(output_path, mri_image)
        return None
    else:
        return mri_image

def batch_convert_ct_to_mri(input_dir, output_dir):
    """Convert multiple CT scans to MRI-like images"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CT scan images in the input directory
    ct_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Convert each CT scan to MRI
    output_paths = []
    for ct_file in ct_files:
        ct_path = os.path.join(input_dir, ct_file)
        output_path = os.path.join(output_dir, f"mri_{os.path.splitext(ct_file)[0]}.png")
        
        try:
            convert_ct_to_mri(ct_path, output_path=output_path)
            output_paths.append(output_path)
            print(f"Converted {ct_file} to {os.path.basename(output_path)}")
        except Exception as e:
            print(f"Error converting {ct_file}: {str(e)}")
    
    return output_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CT scans to MRI-like images')
    parser.add_argument('--input', type=str, help='Path to input CT scan image or directory')
    parser.add_argument('--output', type=str, help='Path to output MRI image or directory')
    parser.add_argument('--batch', action='store_true', help='Process all images in the input directory')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.input or not args.output:
            print("Error: --input and --output directories are required for batch processing")
        else:
            batch_convert_ct_to_mri(args.input, args.output)
    else:
        if not args.input:
            # Default to example image if no input is provided
            ct_path = os.path.join("Dataset", "test", "ct", "ct1925.png")
            output_path = os.path.join("Dataset", "mri_output.png")
        else:
            ct_path = args.input
            output_path = args.output if args.output else None
        
        try:
            convert_ct_to_mri(ct_path, output_path=output_path)
            if output_path:
                print(f"Successfully converted CT scan to MRI and saved at {output_path}")
            else:
                print("Successfully converted CT scan to MRI")
        except Exception as e:
            print(f"Error during conversion: {str(e)}") 