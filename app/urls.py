"""URL patterns for ancient script recognition app."""
from django.urls import path
from . import views

app_name = 'app'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/analyze', views.analyze, name='analyze'),
    path('api/analyze-base64', views.analyze_base64, name='analyze_base64'),
    path('api/history', views.history, name='history'),
]
