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
global_hasil_area = [0,0,0,0]

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

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        resized = cv2.resize(img, (810, 640))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_img)

        mtcnn = MTCNN(image_size=160, margin=0, device=device)

        boxes, _ = mtcnn.detect(pil_image)

        detected_names = set()

        face_img = resized.copy()
        

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(pil_image.width, x2), min(pil_image.height, y2)
                
                face = pil_image.crop((x1, y1, x2, y2))
                face_tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = model2(face_tensor)
                    identity, dist = match_face(embedding, known_embeddings, threshold=0.8)

                detected_names.add(identity)

                cv2.rectangle(face_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{identity} ({dist:.2f})"
                cv2.putText(face_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        results = model(pil_image)
        detections = results.xyxy[0]
        detections = detections[detections[:, 4] >= 0.3]
        jumlah_orang = detections.shape[0]

        names = model.names

        centers = []
        for *box, conf, cls in detections.cpu().numpy():
            xmin, ymin, xmax, ymax = map(int, box)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            centers.append([cx, cy])

        centers = np.array(centers)

        labels = []
        n_clusters = 0
        if len(centers) > 0:
            clustering = DBSCAN(eps=50, min_samples=3).fit(centers)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"Jumlah kerumunan terdeteksi (cluster): {n_clusters}")

        crowd_img = resized.copy()
        
        for *box, conf, cls in detections.cpu().numpy():
            xmin, ymin, xmax, ymax = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"

            cv2.rectangle(crowd_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(crowd_img, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', crowd_img)
        result_crowd = base64.b64encode(buffer).decode('utf-8')

        crowd_cluster_img = resized.copy()

        if len(labels) > 0 and len(centers) > 0:
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue

                cluster_points = centers[labels == cluster_id]
                if len(cluster_points) > 0:
                    x_min, y_min = np.min(cluster_points, axis=0).astype(int)
                    x_max, y_max = np.max(cluster_points, axis=0).astype(int)

                    cv2.rectangle(crowd_cluster_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                    cv2.putText(crowd_cluster_img, f"Crowd #{cluster_id}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        _, buffer = cv2.imencode('.jpg', crowd_cluster_img)
        result_crowd_db = base64.b64encode(buffer).decode('utf-8')

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
    Live streaming, bisa ubah ke drone dan sekaligus face/object detection
    """
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            resized = cv2.resize(frame, (810, 640))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            global global_person_count

            if not db:
                height, width, _ = resized.shape
                h_half, w_half = height // 2, width // 2

                results = model(rgb)
                detections = results.xyxy[0]
                detections = detections[detections[:, 4] >= 0.3]
                names = model.names

                grid_counts = [0, 0, 0, 0]

                for *box, conf, cls in detections.cpu().numpy():
                    xmin, ymin, xmax, ymax = map(int, box)
                    label = f"{names[int(cls)]} {conf:.2f}"

                    cv2.rectangle(resized, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(resized, label, (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    cx = (xmin + xmax) // 2
                    cy = (ymin + ymax) // 2

                    if cx < w_half and cy < h_half:
                        grid_counts[0] += 1  # top-left
                    elif cx >= w_half and cy < h_half:
                        grid_counts[1] += 1  # top-right
                    elif cx < w_half and cy >= h_half:
                        grid_counts[2] += 1  # bottom-left
                    else:
                        grid_counts[3] += 1  # bottom-right

                cv2.line(resized, (w_half, 0), (w_half, height), (255, 255, 0), 2)
                cv2.line(resized, (0, h_half), (width, h_half), (255, 255, 0), 2)
                        
                if detections.shape[0] > global_person_count:
                    global_person_count = detections.shape[0]
                    global global_hasil_area
                    global_hasil_area = grid_counts
                    
                print(global_hasil_area)

            else:
                # Whole frame detection + DBSCAN clustering
                results = model(rgb)
                detections = results.xyxy[0]
                detections = detections[detections[:, 4] >= 0.3]

                if detections.shape[0] > global_person_count:
                    global_person_count = detections.shape[0]

                centers = []
                for *box, conf, cls in detections.cpu().numpy():
                    xmin, ymin, xmax, ymax = map(int, box)
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    centers.append([cx, cy])

                centers = np.array(centers)

                if len(centers) > 0:
                    clustering = DBSCAN(eps=50, min_samples=3).fit(centers)
                    labels = clustering.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    print(f"Jumlah kerumunan terdeteksi (cluster): {n_clusters}")
                else:
                    labels = []
                    n_clusters = 0

                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue
                    cluster_points = centers[labels == cluster_id]
                    x_min, y_min = np.min(cluster_points, axis=0).astype(int)
                    x_max, y_max = np.max(cluster_points, axis=0).astype(int)

                    cv2.rectangle(resized, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                    cv2.putText(resized, f"Crowd #{cluster_id}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', resized)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def video_feed_crowd(request):
    return StreamingHttpResponse(gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')

def video_feed_crowd_db(request):
    return StreamingHttpResponse(gen_frames(True),
        content_type='multipart/x-mixed-replace; boundary=frame')

def gen_frame_raw() :
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

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

            resized = cv2.resize(frame, (810, 640)) 
            _, buffer = cv2.imencode('.jpg', resized)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def video_feed_face(request):
    return StreamingHttpResponse(gen_face_recognition_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')

