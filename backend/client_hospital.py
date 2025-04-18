import argparse
import requests
import time
import os
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import custom_object_scope
from PIL import Image
import tensorflow as tf
import base64
import io
import uuid

from core.model.model_defination import InstanceNormalization, ReflectionPadding2D, BioPhysicalLayer, NeuroSymbolicLayer

custom_objects = {
    'InstanceNormalization': InstanceNormalization,
    'ReflectionPadding2D': ReflectionPadding2D,
    'BioPhysicalLayer': BioPhysicalLayer,
    'NeuroSymbolicLayer': NeuroSymbolicLayer
}


def download_model(server_url):
    url = f"{server_url}/api/get_model/"
    print(f"[+] Downloading global model from: {url}")
    response = requests.get(url)
    model_path = f"temp_model_{uuid.uuid4().hex}.h5"
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print(f"[✓] Model saved to {model_path}")
    return model_path


def mock_train_model(model_path):
    with custom_object_scope(custom_objects):
        model = load_model(model_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
        loss='mae',  # Or a custom GAN + L1 loss
        metrics=['mae', 'ssim']  # Add PSNR if needed
    )

    print("[+] Simulating training...")
    time.sleep(2)
    fake_loss = np.random.uniform(0.01, 0.5)
    fake_time = np.random.uniform(10, 30)

    output_path = f"trained_weights_{uuid.uuid4().hex}.h5"
    model.save(output_path)

    print(f"[✓] Training done | Loss: {fake_loss:.4f} | Time: {fake_time:.2f}s")
    return output_path, fake_loss, fake_time


def upload_weights(server_url,username, weights_path, loss, time_taken):
    url = f"{server_url}/api/upload_weights/"
    with open(weights_path, 'rb') as f:
        files = {'weights': f}
        data = {
            'username': username,
            'loss': str(loss),
            'time_taken': str(time_taken)
        }
        print(f"[+] Uploading weights to: {url}")
        response = requests.post(url, files=files, data=data)
    print(f"[✓] Server response: {response.json()}")


def infer_ct_to_mri(server_url, ct_image_path):
    url = f"{server_url}/api/convert_ct_to_mri/"
    with open(ct_image_path, 'rb') as f:
        files = {'image': f}
        print(f"[+] Sending CT image to: {url}")
        response = requests.post(url, files=files)
    mri_image_b64 = response.json()['mri_image']
    output_img = Image.open(io.BytesIO(base64.b64decode(mri_image_b64)))
    output_img.save("predicted_mri.png")
    print("[✓] MRI image saved as: predicted_mri.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'infer'], required=True, help="Choose mode: train or infer")
    parser.add_argument('--server', default='http://127.0.0.1:8000', help="Django server address")
    parser.add_argument('--username', default='hospital_alpha', help="Hospital user name (optional)")
    parser.add_argument('--ct', help="Path to CT image (only required in infer mode)")
    args = parser.parse_args()

    if args.mode == 'train':
        model_path = download_model(args.server)
        weights_path, loss, time_taken = mock_train_model(model_path)
        upload_weights(args.server,args.username, weights_path, loss, time_taken)
        os.remove(model_path)
        os.remove(weights_path)

    elif args.mode == 'infer':
        if not args.ct:
            print("[-] Please provide a path to a CT image using --ct")
        else:
            infer_ct_to_mri(args.server, args.ct)
