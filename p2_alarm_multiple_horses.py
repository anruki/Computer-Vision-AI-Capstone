from ultralytics import YOLO
import cv2
import os

model = YOLO('yolov8n.pt')
video_path = 'p2/ins/horses_fence.mp4'
output_path = 'p2/outs/horses_fence_yolo.mp4'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error al cargar el frame.")
        break

    results = model.track(frame, persist=True, classes=17)

    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        color = (0, 0, 255) if center_x < 800 else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out.write(frame)
    cv2.imshow('Video Procesado', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f'Procesamiento completado. Video guardado en: {output_path}')
