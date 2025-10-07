from ultralytics import YOLO
import cv2
import numpy as np
import time
import copy
import os

# Start timer
start_time = time.time()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Input video
video_path = "ins/sunday_vid.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Output folder
out_dir = "outs"
os.makedirs(out_dir, exist_ok=True)

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriters
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_original = cv2.VideoWriter(f"{out_dir}/original.mp4", fourcc, fps, (width, height))
out_closed = cv2.VideoWriter(f"{out_dir}/closed.mp4", fourcc, fps, (width, height))
out_closed_yolo = cv2.VideoWriter(f"{out_dir}/closed_yolo.mp4", fourcc, fps, (width, height))
out_original_yolo = cv2.VideoWriter(f"{out_dir}/original_yolo.mp4", fourcc, fps, (width, height))

# Confidence tracking
confidence_sum = 0
frame_count = 0

# Preprocessing
def preprocess_steps(frame):
    original = frame.copy()
    blurred = cv2.GaussianBlur(original, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return original, closed

# Resize helper for visualization only
scale = 0.5
def resize(img):
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o no se puede obtener el frame.")
        break

    # Preprocess
    original, closed = preprocess_steps(frame)

    # YOLO detection on closed frame
    results_closed = model.track(closed, persist=True, classes=17)
    closed_yolo = results_closed[0].plot()

    # Deep copy for original
    results_original = copy.deepcopy(results_closed)
    results_original[0].orig_img = original
    original_yolo = results_original[0].plot()

    # Update stats
    for box in results_closed[0].boxes:
        confidence_sum += float(box.conf)
        frame_count += 1

    # Save each version
    out_original.write(original)
    out_closed.write(closed)
    out_closed_yolo.write(closed_yolo)
    out_original_yolo.write(original_yolo)

    # Visualization (optional)
    original_r = resize(original)
    closed_r = resize(closed)
    closed_yolo_r = resize(closed_yolo)
    original_yolo_r = resize(original_yolo)

    top_row = np.hstack((original_r, closed_r))
    bottom_row = np.hstack((closed_yolo_r, original_yolo_r))
    combined = np.vstack((top_row, bottom_row))

    cv2.imshow('Preprocesamiento + YOLO', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Average confidence
if frame_count > 0:
    avg_confidence = confidence_sum / frame_count
    print(f'Probabilidad promedio final: {avg_confidence:.5f}')

# Cleanup
cap.release()
out_original.release()
out_closed.release()
out_closed_yolo.release()
out_original_yolo.release()
cv2.destroyAllWindows()

end_time = time.time()
print(f'Tiempo de ejecución: {end_time - start_time:.2f} segundos')
print(f"Videos guardados en: {os.path.abspath(out_dir)}")
