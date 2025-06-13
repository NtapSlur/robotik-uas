from django.urls import path
from main.views import *

app_name = 'main'

urlpatterns = [
    path('', show_default, name='show_default'),
    path('stream/', show_stream, name='show_stream'),
    path('video_feed_crowd/', video_feed_crowd, name='video_feed_crowd'),
    path('video_feed_crowd_db', video_feed_crowd_db, name='video_feed_crowd_db'),
    path('video_feed_raw/', video_feed_raw, name='video_feed_raw'),
    path('video_feed_face/', video_feed_face, name='video_feed_face'),
    path('crowd-data/', get_crowd_data, name='crowd_data'),
    path('upload/', show_upload, name='show_upload'),
    path('yolo-detect/', yolo_detect, name='yolo_detect'),
]