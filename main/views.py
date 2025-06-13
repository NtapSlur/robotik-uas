from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from datetime import datetime, timedelta
import random
import cv2
import numpy as np
import base64
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
import cv2
from sklearn.cluster import DBSCAN

crowd_log = []

global_person_count = 0

model = torch.hub.load('main/YOLO-CROWD', 'custom', path_or_model='main/yolo-crowd.pt', source='local')

cap = cv2.VideoCapture(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

model2 = InceptionResnetV1(pretrained="vggface2").to(device)
model2.load_state_dict(torch.load("main/facenet_finetuned.pth", map_location=device))
model2.eval()

def load_known_faces(known_faces_dir, model, transform, device):
    known_embeddings = {}
    model.eval()
    with torch.no_grad():
        for person in tqdm(os.listdir(known_faces_dir), desc="Loading known faces"):
            person_dir = os.path.join(known_faces_dir, person)
            if not os.path.isdir(person_dir):
                continue
            embeddings = []
            for img_name in os.listdir(person_dir):
                if not img_name.endswith(".jpg") and not img_name.endswith(".png"):
                    continue
                img_path = os.path.join(person_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)
                embedding = model(img).cpu().numpy()
                embeddings.append(embedding)
            if embeddings:
                known_embeddings[person] = np.mean(embeddings, axis=0)
    return known_embeddings

known_embeddings = load_known_faces("main/face_1fps", model2, transform, device)


def show_default(request):
    return render(request, "main.html")

def show_stream(request):
    return render(request, "stream.html")

def show_upload(request):
    return render(request, "upload.html")

def yolo_detect(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']

        # Read image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Convert image to RGB and PIL format
        resized = cv2.resize(img, (810, 640))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_img)

        # Initialize face detector (MTCNN)
        mtcnn = MTCNN(image_size=160, margin=0, device=device)

        # Detect faces
        boxes, _ = mtcnn.detect(pil_image)

        detected_names = set()

        # Create a copy of the original image for face detection annotations
        face_img = resized.copy()
        

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(pil_image.width, x2), min(pil_image.height, y2)
                
                face = pil_image.crop((x1, y1, x2, y2))
                face_tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = model2(face_tensor)
                    identity, dist = match_face(embedding, known_embeddings, threshold=0.8)

                detected_names.add(identity)

                # Draw box and label on face_img
                cv2.rectangle(face_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{identity} ({dist:.2f})"
                cv2.putText(face_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        # Run YOLO detection on PIL image
        results = model(pil_image)
        detections = results.xyxy[0]
        detections = detections[detections[:, 4] >= 0.3]
        jumlah_orang = detections.shape[0]

        names = model.names

        # Calculate centers for clustering
        centers = []
        for *box, conf, cls in detections.cpu().numpy():
            xmin, ymin, xmax, ymax = map(int, box)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            centers.append([cx, cy])

        centers = np.array(centers)

        # Use DBSCAN to find crowd areas
        labels = []
        n_clusters = 0
        if len(centers) > 0:
            clustering = DBSCAN(eps=50, min_samples=3).fit(centers)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"Jumlah kerumunan terdeteksi (cluster): {n_clusters}")

        # Create crowd detection image (separate from face detection)
        crowd_img = resized.copy()
        
        # Draw YOLO detections on crowd image
        for *box, conf, cls in detections.cpu().numpy():
            xmin, ymin, xmax, ymax = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"

            # Draw detection box and label
            cv2.rectangle(crowd_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(crowd_img, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Encode crowd image
        _, buffer = cv2.imencode('.jpg', crowd_img)
        result_crowd = base64.b64encode(buffer).decode('utf-8')

        # Create crowd clustering image
        crowd_cluster_img = resized.copy()
        
        # Draw cluster boxes if we have valid labels
        if len(labels) > 0 and len(centers) > 0:
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue  # Skip noise

                cluster_points = centers[labels == cluster_id]
                if len(cluster_points) > 0:
                    x_min, y_min = np.min(cluster_points, axis=0).astype(int)
                    x_max, y_max = np.max(cluster_points, axis=0).astype(int)

                    # Draw cluster box
                    cv2.rectangle(crowd_cluster_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                    cv2.putText(crowd_cluster_img, f"Crowd #{cluster_id}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Encode cluster image
        _, buffer = cv2.imencode('.jpg', crowd_cluster_img)
        result_crowd_db = base64.b64encode(buffer).decode('utf-8')

        # Encode face detection result image
        _, buffer = cv2.imencode('.png', face_img)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return render(request, 'upload.html', {
            'result_image': result_image_base64,
            'crowd_image': result_crowd,
            'crowd_image_db': result_crowd_db,
            'count': max(len(detected_names), jumlah_orang),
            'names': list(detected_names)
        })

    return render(request, 'upload.html')

# def yolo_detect(request):
#     """
#     Method untuk detection pada fitur Upload Foto
#     """
#     if request.method == 'POST' and request.FILES.get('image'):
#         uploaded_file = request.FILES['image']

#         # Read file into OpenCV image
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#         # Run YOLO detection
#         results = model(img)

#         # Draw bounding boxes on the original image
#         annotated_frame = results[0].plot()

#         # Encode the annotated image as base64
#         _, buffer = cv2.imencode('.png', annotated_frame)
#         result_image_base64 = base64.b64encode(buffer).decode('utf-8')

#         count = 5 # Ini nanti jadi jumlah orang yang terdetect di foto
#         name_list = ["Aku"] # Ini nanti jadi semua nama orang yang terdetect di foto

#         return render(request, 'upload.html', {
#             'result_image': result_image_base64,
#             'count': count,
#             'names': name_list
#         })

#     return render(request, 'upload.html')

def update_crowd_data():
    now = datetime.now().strftime('%H:%M')

    global global_person_count

    if not crowd_log or crowd_log[-1]['time'] != now:
        crowd_log.append({'time': now, 'count': global_person_count})

    if len(crowd_log) > 24:
        crowd_log.pop(0)
    
    print(crowd_log)
    global_person_count = 0

def get_crowd_data(request):
    """
    Method untuk memperoleh data crowd untuk statistik dashboard, tapi bukan untuk face recognition
    """
    labels = [entry['time'] for entry in crowd_log]
    values = [entry['count'] for entry in crowd_log]
    return JsonResponse({'labels': labels, 'values': values})

def gen_frames(db=False):
    """
    Live streaming, ini bisa ubah ke drone dan sekaligus face detection disini
    """
    # cap = cv2.VideoCapture(0)  # 0 for default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            resized = cv2.resize(frame, (810, 640))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Deteksi objek
            results = model(rgb)
            detections = results.xyxy[0]
            detections = detections[detections[:, 4] >= 0.3]
            global global_person_count
            
            if detections.shape[0] > global_person_count:
                global_person_count = detections.shape[0]

            names = model.names

            centers = []
            for *box, conf, cls in detections.cpu().numpy():
                xmin, ymin, xmax, ymax = map(int, box)
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                centers.append([cx, cy])

            centers = np.array(centers)

            # Gunakan DBSCAN untuk cari area kerumunan
            if len(centers) > 0:
                clustering = DBSCAN(eps=50, min_samples=3).fit(centers)
                labels = clustering.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                print(f"Jumlah kerumunan terdeteksi (cluster): {n_clusters}")
            else:
                labels = []
                n_clusters = 0

            if (not db):
                for *box, conf, cls in detections.cpu().numpy():
                    xmin, ymin, xmax, ymax = map(int, box)
                    label = f"{names[int(cls)]} {conf:.2f}"

                    # Gambar kotak deteksi dan label ke frame
                    cv2.rectangle(resized, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(resized, label, (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                #ini
                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue  # Lewati noise

                    cluster_points = centers[labels == cluster_id]
                    x_min, y_min = np.min(cluster_points, axis=0).astype(int)
                    x_max, y_max = np.max(cluster_points, axis=0).astype(int)

                    # Gambar kotak cluster (kerumunan)
                    cv2.rectangle(resized, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                    cv2.putText(resized, f"Crowd #{cluster_id}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Encode frame sebagai JPEG untuk streaming
            _, buffer = cv2.imencode('.jpg', resized)
            frame = buffer.tobytes()

            # Create a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def video_feed_crowd(request):
    return StreamingHttpResponse(gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')

def video_feed_crowd_db(request):
    return StreamingHttpResponse(gen_frames(True),
        content_type='multipart/x-mixed-replace; boundary=frame')

def gen_frame_raw() :
      # 0 for default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Create a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def video_feed_raw(request):
    return StreamingHttpResponse(gen_frame_raw(),
        content_type='multipart/x-mixed-replace; boundary=frame')

def match_face(embedding, known_embeddings, threshold=0.8):
    min_dist = float("inf")
    identity = "Unknown"
    embedding = embedding.cpu().numpy()
    for name, known_emb in known_embeddings.items():
        dist = np.linalg.norm(embedding - known_emb)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            identity = name
    return identity, min_dist

def gen_face_recognition_frames():

    mtcnn = MTCNN(image_size=160, margin=0, device=device)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            boxes, _ = mtcnn.detect(pil_image)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face = pil_image.crop((x1, y1, x2, y2))
                    face = transform(face).unsqueeze(0).to(device)
                    embedding = model2(face)
                    identity, dist = match_face(embedding, known_embeddings, threshold=0.8)
                    label = f"{identity} ({dist:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            resized = cv2.resize(frame, (810, 640))  # opsional
            _, buffer = cv2.imencode('.jpg', resized)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def video_feed_face(request):
    return StreamingHttpResponse(gen_face_recognition_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')

