
# ==============================================
# 2. Federated Learning Implementation
# ==============================================



# with keras.utils.custom_object_scope(custom_objects):
#     hospital_a_model = keras.models.load_model('/content/drive/MyDrive/bpnsgm_ct_to_mri.h5')
#     hospital_b_model = keras.models.load_model('/content/bpnsgm_mri_to_ct.h5')

# hospital_a_model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
#     loss='mae',  # Or a custom GAN + L1 loss
#     metrics=['mae', 'ssim']  # Add PSNR if needed
# )

# hospital_b_model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
#     loss='mae',  # Or a custom GAN + L1 loss
#     metrics=['mae', 'ssim']  # Add PSNR if needed
# )

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