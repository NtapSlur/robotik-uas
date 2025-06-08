from django.urls import path
from main.views import *

app_name = 'main'

urlpatterns = [
    path('', show_default, name='show_default'),
    path('stream/', show_stream, name='show_stream'),
    path('video_feed/', video_feed, name='video_feed'),
    path('crowd-data/', get_crowd_data, name='crowd_data'),
    path('upload/', show_upload, name='show_upload'),
    path('yolo-detect/', yolo_detect, name='yolo_detect'),
]