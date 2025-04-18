from django.urls import path
from . import views

urlpatterns = [
    path('api/convert_ct_to_mri/', views.convert_ct_to_mri, name='convert_ct_to_mri'),
    path('api/upload_weights/', views.upload_weights, name='upload_weights'),
    path('api/get_model/', views.get_model, name='get_model'),
    path('api/aggregate_weights/', views.aggregate_weights, name='aggregate_weights'),
    path('api/train/', views.train_model, name='train_model'),
    path('api/metrics/', views.get_metrics, name='get_metrics'),
] 