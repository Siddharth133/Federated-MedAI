from django.shortcuts import render

# Create your views here.

import os
import io
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

@csrf_exempt
def upload_weights(request):
    if request.method == 'POST':
        username = request.POST['username']  # Expect hospital name
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return JsonResponse({'status': 'Invalid user'}, status=400)

        weights_file = request.FILES['weights']
        loss = float(request.POST['loss'])
        time_taken = float(request.POST['time_taken'])

        try:
            # Attempt to get the latest version by version_number
            latest_version = ModelVersion.objects.get(version_number=1)
        except ModelVersion.DoesNotExist:
            # If no version with version_number=1, create it
            print("[+] No version found. Creating version 1.")
            latest_version = ModelVersion.objects.create(
                version_number=1,
                created_at= time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) # Assuming you have a `created_at` field
            )
            print(f"[+] Created version: {latest_version.version_number}")

        update = TrainingUpdate.objects.create(
            client=user,
            version=latest_version,
            weights_file=weights_file,
            loss=loss,
            time_taken=time_taken
        )
        return JsonResponse({'status': 'weights received'})



def get_model(request):
    # latest_version = ModelVersion.objects.latest('created_at')
    filepath = "core/global_model/bpnsgm_ct_to_mri.h5"
    print("model is downloaing")
    with open(filepath, 'rb') as f:
        return HttpResponse(f.read(), content_type="application/h5")

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
    for update in updates:
        client_model = BPNSGM_CycleGAN()
        client_model.compile(
            g_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            d_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        )
        client_model.load_weights(update.weights_file.path)
        weights_list.append(client_model.get_weights())

    # Average and create new global model
    averaged_weights = average_weights(weights_list)

    global_model = BPNSGM_CycleGAN()
    global_model.compile(
        g_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        d_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    )
    global_model.set_weights(averaged_weights)

    # Save new version
    new_version_num = latest_version.version_number + 1
    model_path = os.path.join(settings.MEDIA_ROOT, f'models/model_v{new_version_num}.h5')
    global_model.save(model_path)

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

        with keras.utils.custom_object_scope(custom_objects):
            model = keras.models.load_model('global_model/bpnsgm_ct_to_mri.h5')

        model.compile(
            g_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            d_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        )

        img = Image.open(ct_image).resize((256, 256))
        img_arr = np.array(img) / 127.5 - 1.0
        img_arr = img_arr[np.newaxis, ...]

        pred = model.predict(img_arr)
        output = (pred[0] + 1.0) * 127.5
        output_img = Image.fromarray(np.uint8(output))

        buffer = io.BytesIO()
        output_img.save(buffer, format="PNG")
        encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

        InferenceLog.objects.create(version=latest_version)

        return JsonResponse({'mri_image': encoded_img})


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

class TriggerTrainingView(APIView):
    def post(self, request):
        # Simulate training
        print("Training started...")

        # You could run a subprocess here or trigger async logic
        time.sleep(3)  # Simulate time-consuming process

        # Sample response
        result = {
            'status': 'success',
            'accuracy': 0.91,
            'loss': 0.12,
        }
        return Response(result)
