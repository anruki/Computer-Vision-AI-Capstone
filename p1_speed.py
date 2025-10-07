from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

# --- Initialization ---
start_time = time.time()
model = YOLO("yolov8n.pt")

video_path = "p1/ins/sunday_vid.mov"
out_dir = "p1/outs"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_speed_yolo = cv2.VideoWriter(f"{out_dir}/speed_yolo.mp4", fourcc, fps, (width, height))

# --- Tracking & calibration variables ---
confidence_sum, frame_count = 0, 0
pos_prev, smooth_vec, smooth_speed = {}, {}, {}
prev_gray = None
alpha = 0.7
horse_height_m = 1.7
pix_a_m = 0.05
pix_a_m_smooth = pix_a_m

# --- Helper functions ---
def preprocess_steps(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def calcular_velocidad(cx, cy, pos_prev, obj_id, frame_rate, pix_a_m):
    if obj_id in pos_prev:
        prev_cx, prev_cy = pos_prev[obj_id]
        dx = cx - prev_cx
        dy = cy - prev_cy
        dist_px = np.sqrt(dx**2 + dy**2)
        dist_m = dist_px * pix_a_m
        vel = dist_m * frame_rate
        return vel, dx, dy
    return 0.0, 0, 0

def estimate_camera_motion(prev_gray, curr_gray):
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if prev_pts is None:
        return 0, 0
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]
    if len(good_prev) < 5:
        return 0, 0
    dxs = good_curr[:, 0] - good_prev[:, 0]
    dys = good_curr[:, 1] - good_prev[:, 1]
    return np.median(dxs), np.median(dys)

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    closed = preprocess_steps(frame)
    gray = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)

    cam_dx, cam_dy = (0, 0)
    if prev_gray is not None:
        cam_dx, cam_dy = estimate_camera_motion(prev_gray, gray)

    results = model.track(closed, persist=True, classes=17)
    results[0].orig_img = frame.copy()
    yolo_frame = results[0].plot()
    speed_frame = yolo_frame.copy()

    if results[0].boxes.id is not None:
        heights = []

        for box in results[0].boxes:
            obj_id = int(box.id.item()) if box.id is not None else None
            if obj_id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            box_height = y2 - y1
            heights.append(box_height)

            # --- Dynamic calibration based on average horse height ---
            if box_height > 0:
                pix_per_meter = box_height / horse_height_m
                new_pix_a_m = 1.0 / pix_per_meter
                pix_a_m_smooth = 0.95 * pix_a_m_smooth + 0.1 * new_pix_a_m

            vel, dx, dy = calcular_velocidad(cx, cy, pos_prev, obj_id, fps, pix_a_m_smooth)
            rel_dx, rel_dy = dx - cam_dx, dy - cam_dy

            # Smooth motion vector
            if obj_id in smooth_vec:
                sx, sy = smooth_vec[obj_id]
                rel_dx = alpha * sx + (1 - alpha) * rel_dx
                rel_dy = alpha * sy + (1 - alpha) * rel_dy
            smooth_vec[obj_id] = (rel_dx, rel_dy)

            # Smooth speed & reject spikes
            if obj_id in smooth_speed:
                prev_v = smooth_speed[obj_id]
                if abs(vel - prev_v) > 5:
                    vel = prev_v
                vel = alpha * prev_v + (1 - alpha) * vel
            smooth_speed[obj_id] = vel

            pos_prev[obj_id] = (cx, cy)
            confidence_sum += float(box.conf)
            frame_count += 1

            # --- Draw smoothed speed text ---
            valor = f"{vel:.2f}"
            unidad = "m/s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (80, 255, 80)

            (tw1, th1), _ = cv2.getTextSize(valor, font, 0.6, 2)
            (tw2, th2), _ = cv2.getTextSize(unidad, font, 0.6, 2)
            tx = x1 + (x2 - x1 - tw1) // 2 - 14
            ty_center = y1 + (y2 - y1) // 2

            cv2.putText(speed_frame, valor, (tx, ty_center - 5),
                        font, 0.6, color, 2, cv2.LINE_AA)
            tx2 = x1 + (x2 - x1 - tw2) // 2 - 10
            cv2.putText(speed_frame, unidad, (tx2, ty_center + th2 + 5),
                        font, 0.6, color, 2, cv2.LINE_AA)

            # --- Motion arrow ---
            mv_scale = 10
            end_x = int(cx + rel_dx * mv_scale)
            end_y = int(cy + rel_dy * mv_scale)
            cv2.arrowedLine(speed_frame, (cx, cy), (end_x, end_y),
                            (80, 255, 80), 2, tipLength=0.2)

    out_speed_yolo.write(speed_frame)
    cv2.imshow("Speed + Vector YOLO (Original Boxes)", cv2.resize(speed_frame, (0, 0), fx=0.5, fy=0.5))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    prev_gray = gray

# --- Summary ---
if frame_count > 0:
    avg_conf = confidence_sum / frame_count
    print(f"Probabilidad promedio final: {avg_conf:.5f}")

cap.release()
out_speed_yolo.release()
cv2.destroyAllWindows()

end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")
print(f"Video guardado en: {os.path.abspath(out_dir)}/speed_yolo.mp4")
