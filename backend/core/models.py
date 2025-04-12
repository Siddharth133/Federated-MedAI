# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class ModelVersion(models.Model):
    version_number = models.IntegerField()
    model_file = models.FileField(upload_to='models/')
    created_at = models.DateTimeField(auto_now_add=True)

class TrainingUpdate(models.Model):
    client = models.ForeignKey(User, on_delete=models.CASCADE)
    version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE)
    weights_file = models.FileField(upload_to='client_weights/')
    loss = models.FloatField()
    time_taken = models.FloatField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

class InferenceLog(models.Model):
    version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
