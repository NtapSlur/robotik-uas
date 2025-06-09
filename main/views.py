from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from datetime import datetime, timedelta
import random
import cv2
import numpy as np
import base64
from ultralytics import YOLO


crowd_log = []

model = YOLO("yolov8n.pt")

def show_default(request):
    return render(request, "main.html")

def show_stream(request):
    return render(request, "stream.html")

def show_upload(request):
    return render(request, "upload.html")

def yolo_detect(request):
    """
    Method untuk detection pada fitur Upload Foto
    """
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']

        # Read file into OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run YOLO detection
        results = model(img)

        # Draw bounding boxes on the original image
        annotated_frame = results[0].plot()

        # Encode the annotated image as base64
        _, buffer = cv2.imencode('.png', annotated_frame)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')

        count = 5 # Ini nanti jadi jumlah orang yang terdetect di foto
        name_list = ["Aku"] # Ini nanti jadi semua nama orang yang terdetect di foto

        return render(request, 'upload.html', {
            'result_image': result_image_base64,
            'count': count,
            'names': name_list
        })

    return render(request, 'upload.html')

def update_crowd_data():
    now = datetime.now().strftime('%H:%M')
    count = random.randint(5, 30)  # Nanti ganti ke logic asli

    if not crowd_log or crowd_log[-1]['time'] != now:
        crowd_log.append({'time': now, 'count': count})

    if len(crowd_log) > 24:
        crowd_log.pop(0)
    
    print(crowd_log)

def get_crowd_data(request):
    """
    Method untuk memperoleh data crowd untuk statistik dashboard, tapi bukan untuk face recognition
    """
    labels = [entry['time'] for entry in crowd_log]
    values = [entry['count'] for entry in crowd_log]
    return JsonResponse({'labels': labels, 'values': values})

def gen_frames():
    """
    Live streaming, ini bisa ubah ke drone dan sekaligus face detection disini
    """
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Create a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')