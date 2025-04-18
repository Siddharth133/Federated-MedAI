from django.shortcuts import render

# Create your views here.

import os
import io
import json
import base64
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import time
import keras
import matplotlib.pyplot as plt


from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser

from keras.utils import custom_object_scope

from .models import *


from core.model.model_defination import InstanceNormalization, ReflectionPadding2D, BioPhysicalLayer, NeuroSymbolicLayer

custom_objects = {
    'InstanceNormalization': InstanceNormalization,
    'ReflectionPadding2D': ReflectionPadding2D,
    'BioPhysicalLayer': BioPhysicalLayer,
    'NeuroSymbolicLayer': NeuroSymbolicLayer
}

# Global training progress tracker
training_progress = {}

@csrf_exempt
def upload_weights(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return JsonResponse({'status': 'Invalid user'}, status=400)

        weights_file = request.FILES.get('weights')
        loss = float(request.POST.get('loss', 0))
        time_taken = float(request.POST.get('time_taken', 0))

        if not weights_file:
            return JsonResponse({'status': 'error', 'message': 'No weights file provided'}, status=400)

        try:
            latest_version = ModelVersion.objects.latest('created_at')
            
            update = TrainingUpdate.objects.create(
                client=user,
                version=latest_version,
                weights_file=weights_file,
                loss=loss,
                time_taken=time_taken
            )
            
            return JsonResponse({
                'status': 'success',
                'message': 'Weights received successfully',
                'update_id': update.id
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def get_model(request):
    try:
        latest_version = ModelVersion.objects.latest('created_at')
        filepath = "core/global_model/bpnsgm_ct_to_mri.h5"
        
        if not os.path.exists(filepath):
            return JsonResponse({'status': 'error', 'message': 'Model file not found'}, status=404)
            
        with open(filepath, 'rb') as f:
            response = HttpResponse(f.read(), content_type="application/octet-stream")
            response['Content-Disposition'] = 'attachment; filename="model.h5"'
            return response
            
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def average_weights(weights_list):
    avg_weights = []
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

@csrf_exempt
def aggregate_weights(request):
    # Load all updates for latest version
    latest_version = ModelVersion.objects.latest('created_at')
    updates = TrainingUpdate.objects.filter(version=latest_version)

    if updates.count() == 0:
        return JsonResponse({'status': 'No updates found'})

    weights_list = []
    filepath = "core/global_model/bpnsgm_ct_to_mri.h5"
    with custom_object_scope(custom_objects):
        model = load_model(model_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
        loss='mae',  # Or a custom GAN + L1 loss
        metrics=['mae', 'ssim']  # Add PSNR if needed
    )
    for update in updates:
        model.load_weights(update.weights_file.path)
        weights_list.append(model.get_weights())

    # Average and create new global model
    averaged_weights = average_weights(weights_list)

    model.set_weights(averaged_weights)

    # Save new version
    new_version_num = latest_version.version_number + 1
    model_path = os.path.join(settings.MEDIA_ROOT, f'models/model_v{new_version_num}.h5')
    model.save(model_path)

    new_version = ModelVersion.objects.create(
        version_number=new_version_num,
        model_file=f'models/model_v{new_version_num}.h5'
    )

    return JsonResponse({'status': 'New global model aggregated', 'version': new_version_num})

@csrf_exempt
def convert_ct_to_mri(request):
    if request.method == 'POST':
        username = request.POST['username']
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return JsonResponse({'status': 'Invalid user'}, status=400)

        ct_image = request.FILES['image']
        latest_version = ModelVersion.objects.latest('created_at')

        try:
            # Load the model with the custom objects scope
            with keras.utils.custom_object_scope(custom_objects):
                model = keras.models.load_model('core/global_model/bpnsgm_ct_to_mri.h5')

            # Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                loss='mae',
                metrics=['mae', 'ssim']
            )

            # Process the input image
            img = Image.open(ct_image).convert('L').resize((256, 256))  # Convert to grayscale
            img_arr = np.array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
            img_arr = img_arr[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

            # Make the prediction
            pred = model.predict(img_arr)

            # Ensure the output is in the correct range [0, 255]
            output = (pred[0, ..., 0] + 1.0) * 127.5  # Convert from [-1, 1] to [0, 255]
            output = np.clip(output, 0, 255).astype(np.uint8)  # Ensure values are in valid range

            # Convert the array to an image
            output_img = Image.fromarray(output)

            # Save the output image to a buffer
            buffer = io.BytesIO()
            output_img.save(buffer, format="PNG")
            buffer.seek(0)

            # Log the inference
            InferenceLog.objects.create(version=latest_version)

            # Return the image directly as a PNG
            encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return JsonResponse({'mri_image': encoded_img})

        except Exception as e:
            # Log the detailed exception for debugging
            print(f"Error during model processing: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full traceback
            return JsonResponse({'error': str(e)}, status=500)

def get_metrics(request):
    total_updates = TrainingUpdate.objects.count()
    total_inference = InferenceLog.objects.count()
    latest_version = ModelVersion.objects.latest('created_at')

    client_stats = []
    users = User.objects.all()
    for user in users:
        updates = TrainingUpdate.objects.filter(client=user)
        if updates.exists():
            avg_loss = updates.aggregate(models.Avg('loss'))['loss__avg']
            avg_time = updates.aggregate(models.Avg('time_taken'))['time_taken__avg']
            client_stats.append({
                'username': user.username,
                'avg_loss': avg_loss,
                'avg_time': avg_time,
                'contributions': updates.count()
            })

    return JsonResponse({
        'total_updates': total_updates,
        'inference_count': total_inference,
        'current_version': latest_version.version_number,
        'clients': client_stats
    })

from tensorflow.keras import backend as K

def ssim_metric(y_true, y_pred):
    # Convert inputs to float32 to ensure type consistency
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

@csrf_exempt
def get_training_progress(request):
    """Get the current training progress for a user"""
    try:
        # Handle both POST and GET requests
        if request.method == 'POST':
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                data = request.POST
        else:
            data = request.GET

        username = data.get('username')
        
        if not username:
            return JsonResponse({
                'status': 'error',
                'error': 'Username is required'
            }, status=400)
            
        progress = training_progress.get(username, {
            'status': 'idle',
            'current_epoch': 0,
            'total_epochs': 0,
            'current_loss': 0,
            'time_elapsed': 0,
            'estimated_time_remaining': 0
        })
        
        # Convert NumPy types to Python native types
        if progress:
            progress = {
                'status': str(progress['status']),
                'current_epoch': int(progress['current_epoch']),
                'total_epochs': int(progress['total_epochs']),
                'current_loss': float(progress['current_loss']),
                'time_elapsed': float(progress['time_elapsed']),
                'estimated_time_remaining': float(progress['estimated_time_remaining'])
            }
        
        return JsonResponse({
            'status': 'success',
            'progress': progress
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Print the full traceback for debugging
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)

@csrf_exempt
def train_model(request):
    if request.method == 'POST':
        try:
            # Get and validate all required parameters
            required_params = {
                'username': request.POST.get('username'),
                'data_path': request.POST.get('data_path'),
                'epochs': request.POST.get('epochs'),
                'batch_size': request.POST.get('batch_size'),
                'learning_rate': request.POST.get('learning_rate'),
                'beta1': request.POST.get('beta1')
            }

            # Check if any required parameter is missing
            missing_params = [k for k, v in required_params.items() if v is None]
            if missing_params:
                return JsonResponse({
                    'status': 'error',
                    'error': f'Missing required parameters: {", ".join(missing_params)}'
                }, status=400)

            # Convert and validate numeric parameters
            try:
                epochs = int(required_params['epochs'])
                batch_size = int(required_params['batch_size'])
                learning_rate = float(required_params['learning_rate'])
                beta_1 = float(required_params['beta1'])

                if epochs < 1:
                    raise ValueError("Epochs must be at least 1")
                if batch_size < 1:
                    raise ValueError("Batch size must be at least 1")
                if learning_rate <= 0:
                    raise ValueError("Learning rate must be positive")
                if not (0 <= beta_1 <= 1):
                    raise ValueError("Beta1 must be between 0 and 1")

            except ValueError as e:
                return JsonResponse({
                    'status': 'error',
                    'error': f'Invalid parameter value: {str(e)}'
                }, status=400)

            # Validate user
            try:
                user = User.objects.get(username=required_params['username'])
            except User.DoesNotExist:
                return JsonResponse({
                    'status': 'error',
                    'error': f'Invalid username: {required_params["username"]}'
                }, status=400)

            # Convert path to proper format and validate
            try:
                import os.path
                import pathlib
                data_path = pathlib.Path(required_params['data_path'])
                ct_path = data_path / 'ct'
                mri_path = data_path / 'mri'

                if not data_path.exists():
                    return JsonResponse({
                        'status': 'error',
                        'error': f'Data path not found: {str(data_path)}'
                    }, status=400)

                if not (ct_path.exists() and mri_path.exists()):
                    return JsonResponse({
                        'status': 'error',
                        'error': f'Required subdirectories not found: {str(ct_path)} or {str(mri_path)}'
                    }, status=400)

            except Exception as e:
                return JsonResponse({
                    'status': 'error',
                    'error': f'Invalid path format: {str(e)}'
                }, status=400)

            # Get the latest model version
            try:
                latest_version = ModelVersion.objects.latest('created_at')
            except ModelVersion.DoesNotExist:
                return JsonResponse({
                    'status': 'error',
                    'error': 'No model version found'
                }, status=400)

            # Load and validate the model
            model_path = 'core/global_model/bpnsgm_ct_to_mri.h5'
            if not os.path.exists(model_path):
                return JsonResponse({
                    'status': 'error',
                    'error': 'Model file not found'
                }, status=400)

            try:
                with keras.utils.custom_object_scope(custom_objects):
                    model = keras.models.load_model(model_path)
            except Exception as e:
                return JsonResponse({
                    'status': 'error',
                    'error': f'Error loading model: {str(e)}'
                }, status=500)

            # Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1),
                loss='mae',
                metrics=['mae', ssim_metric]
            )

            # Load and validate training data
            ct_files = [f for f in os.listdir(ct_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not ct_files:
                return JsonResponse({
                    'status': 'error',
                    'error': 'No training images found in CT directory'
                }, status=400)

            # Initialize progress tracking for this user
            start_time = time.time()
            training_progress[user.username] = {
                'status': 'training',
                'current_epoch': 0,
                'total_epochs': epochs,
                'current_loss': 0,
                'time_elapsed': 0,
                'estimated_time_remaining': 0,
                'start_time': start_time,
                'epoch_times': []  # Track time for each epoch
            }

            try:
                # Training loop
                total_loss = 0
                for epoch in range(epochs):
                    epoch_start_time = time.time()
                    epoch_loss = 0
                    
                    # Simple time tracking
                    elapsed_time = time.time() - start_time
                    
                    # Calculate estimated time remaining
                    if training_progress[user.username]['epoch_times']:
                        # Use average of previous epochs to estimate remaining time
                        avg_epoch_time = sum(training_progress[user.username]['epoch_times']) / len(training_progress[user.username]['epoch_times'])
                        remaining_epochs = epochs - (epoch + 1)
                        estimated_remaining = avg_epoch_time * remaining_epochs
                    else:
                        # First epoch, estimate based on elapsed time so far
                        if epoch > 0:
                            # We're in the first epoch, estimate based on progress so far
                            progress_ratio = (ct_files.index(ct_file) + 1) / len(ct_files) if ct_files else 0
                            if progress_ratio > 0:
                                estimated_remaining = (elapsed_time / progress_ratio) * (1 - progress_ratio)
                            else:
                                estimated_remaining = 0
                        else:
                            estimated_remaining = 0
                    
                    # Update progress
                    training_progress[user.username].update({
                        'current_epoch': epoch + 1,
                        'total_epochs': epochs,
                        'current_loss': epoch_loss / len(ct_files) if ct_files else 0,
                        'time_elapsed': elapsed_time,
                        'estimated_time_remaining': estimated_remaining
                    })

                    for ct_file in ct_files:
                        # Get base filename without extension and convert CT to MRI naming
                        ct_base_name = os.path.splitext(ct_file)[0]
                        mri_base_name = ct_base_name.replace('ct', 'mri')
                        
                        # Find matching MRI file with any supported extension
                        mri_file = None
                        for ext in ['.png', '.jpg', '.jpeg']:
                            potential_mri = mri_base_name + ext
                            if os.path.exists(os.path.join(mri_path, potential_mri)):
                                mri_file = potential_mri
                                break
                        
                        if not mri_file:
                            return JsonResponse({
                                'status': 'error',
                                'error': f'No matching MRI file found for CT file: {ct_base_name} (looked for {mri_base_name}.png/jpg/jpeg)'
                            }, status=400)

                        # Load CT image
                        ct_img = Image.open(os.path.join(ct_path, ct_file)).convert('L').resize((256, 256))
                        ct_arr = np.array(ct_img, dtype=np.float32) / 127.5 - 1.0
                        ct_arr = ct_arr[np.newaxis, ..., np.newaxis]

                        # Load corresponding MRI image
                        mri_img = Image.open(os.path.join(mri_path, mri_file)).convert('L').resize((256, 256))
                        mri_arr = np.array(mri_img, dtype=np.float32) / 127.5 - 1.0
                        mri_arr = mri_arr[np.newaxis, ..., np.newaxis]

                        # Train on batch
                        loss = model.train_on_batch(ct_arr, mri_arr)[0]
                        epoch_loss += loss
                        
                        # Update current loss in progress
                        training_progress[user.username]['current_loss'] = epoch_loss / (ct_files.index(ct_file) + 1)
                        
                        # Update time elapsed after each batch
                        training_progress[user.username]['time_elapsed'] = time.time() - start_time
                        
                        # Update estimated time remaining during epoch
                        if epoch == 0 and ct_files.index(ct_file) > 0:
                            progress_ratio = (ct_files.index(ct_file) + 1) / len(ct_files)
                            if progress_ratio > 0:
                                current_epoch_time = time.time() - epoch_start_time
                                estimated_epoch_time = current_epoch_time / progress_ratio
                                remaining_epochs = epochs - 1
                                estimated_remaining = estimated_epoch_time * remaining_epochs + (estimated_epoch_time - current_epoch_time)
                                training_progress[user.username]['estimated_time_remaining'] = estimated_remaining

                    avg_epoch_loss = epoch_loss / len(ct_files)
                    total_loss += avg_epoch_loss
                    
                    # Record epoch time for future estimates
                    epoch_time = time.time() - epoch_start_time
                    training_progress[user.username]['epoch_times'].append(epoch_time)
                    
                    # Update progress after epoch completion
                    training_progress[user.username].update({
                        'current_loss': avg_epoch_loss,
                        'time_elapsed': time.time() - start_time,
                        'estimated_time_remaining': sum(training_progress[user.username]['epoch_times']) / len(training_progress[user.username]['epoch_times']) * (epochs - (epoch + 1)) if training_progress[user.username]['epoch_times'] else 0
                    })
                    
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Time: {time.time() - epoch_start_time:.2f}s, Elapsed: {time.time() - start_time:.2f}s, Remaining: {training_progress[user.username]['estimated_time_remaining']:.2f}s")

                # Training completed
                training_progress[user.username]['status'] = 'completed'
                
                # Calculate final metrics
                avg_loss = total_loss / epochs
                time_taken = time.time() - start_time

                # Save weights
                weights_filename = f'client_weights/weights_{user.username}_{int(time.time())}.h5'
                model.save_weights(weights_filename)

                # Create training update record
                update = TrainingUpdate.objects.create(
                    client=user,
                    version=latest_version,
                    weights_file=weights_filename,
                    loss=float(avg_loss),
                    time_taken=float(time_taken)
                )

                return JsonResponse({
                    'status': 'success',
                    'loss': float(avg_loss),
                    'time_taken': float(time_taken),
                    'message': 'Training completed successfully'
                })

            except Exception as e:
                # Update progress to error state
                training_progress[user.username]['status'] = 'error'
                training_progress[user.username]['error'] = str(e)
                import traceback
                print(traceback.format_exc())
                return JsonResponse({
                    'status': 'error',
                    'error': f'Training error: {str(e)}'
                }, status=500)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({
                'status': 'error',
                'error': f'Unexpected error: {str(e)}'
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'error': 'Invalid request method'
    }, status=405)
