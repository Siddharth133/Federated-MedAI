"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from core.views import train_model, get_model, upload_weights, aggregate_weights, convert_ct_to_mri, get_metrics, get_training_progress

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/train/', train_model, name='train-model'),
    path('api/get_model/', get_model, name='get-model'),
    path('api/upload_weights/', upload_weights, name='upload-weights'),
    path('api/aggregate_weights/', aggregate_weights, name='aggregate-weights'),
    path('api/convert_ct_to_mri/', convert_ct_to_mri, name='convert-ct-to-mri'),
    path('api/get_metrics/', get_metrics, name='get-metrics'),
    path('api/get_training_progress/', get_training_progress, name='get_training_progress'),
]
