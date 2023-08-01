from django.urls import path
from .views import Index, Result

urlpatterns = [
    path('', Index, name = 'index'),
    path("result/", Result, name = 'result'),
]