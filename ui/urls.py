from django.contrib import admin
from django.urls import path, include
from .views import UIView, get_next_predict

urlpatterns = [

    path('', UIView.as_view(), name='ui'),
    #path('get-predict', get_next_predict, name='get_predict'),
]