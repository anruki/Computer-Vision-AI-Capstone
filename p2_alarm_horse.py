from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')
video_path = 'p2/ins/sunday_escapa.mp4'
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = 'p2/outs/sunday_escapa_yolo.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=17)

    alert_active = False  # Track if any detection triggers alert

    for result in results[0].boxes:
        label = model.names[int(result.cls)]
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        center_x = (x1 + x2) // 2

        if center_x < width // 2:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            alert_active = True  # Activate red overlay
            cv2.putText(frame, 'HORSE ESCAPED...', 
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        3)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Apply transparent red overlay if alert is active
    if alert_active:
        overlay = frame.copy()
        overlay[:] = (0, 0, 255)  # Red overlay
        alpha = 0.4                # Transparency factor (0.0-1.0)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    out.write(frame)
    cv2.imshow('Video Procesado', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print('Procesamiento de video completado.')
