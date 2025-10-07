from ultralytics import YOLO
import cv2
import numpy as np

# Diccionario para seguimiento
objects_state = {}  # Diccionario: ID -> {inside_curr_frame, inside_prev_frame}
set_alert = False

# Polígono del área restringida
area_points = np.array([[350, 250], [1280, 250], [1280, 500], [350, 350]])

# Función para verificar si un punto está dentro del área
def point_in_area(point, area_points):
    return cv2.pointPolygonTest(area_points, point, False) >= 0

# Cargar modelo YOLOv8
model = YOLO('yolov8l.pt')

# Ruta del video de entrada
video_path = 'p4/ins/ponies.mp4'
cap = cv2.VideoCapture(video_path)

# Verificar si el video se cargó correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Obtener las características del video original (tamaño, fps, etc.)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Crear un objeto VideoWriter para guardar el video de salida con alta calidad
output_path = 'p4/outs/ponies_yolo.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Puedes usar 'X264' o 'H264' si está disponible para mayor calidad
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Procesamiento del video
while True:
    # Leer un frame
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error al cargar el frame.")
        break

    # Preprocesamiento del frame
    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_converted = cv2.convertScaleAbs(frame_blurred, alpha=1.2, beta=50)

    # Tracking con YOLOv8
    results = model.track(frame_converted, persist=True, classes=[15, 16, 17, 18, 19, 20, 21, 22, 23], conf=0.4)

    # Dibujar el área restringida (Opcional)
    # cv2.polylines(frame, [area_points], isClosed=True, color=(0, 0, 255), thickness=2)

    # Actualizar estados actuales para todos los objetos detectados
    for result in results[0].boxes:  # Acceder a las cajas detectadas
        # Obtener ID de seguimiento
        obj_id = int(result.id.item())

        # Coordenadas del cuadro delimitador
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        center_x = (x1 + x2) // 2
        bottom_center = (center_x, y2 - 15)  # Punto inferior central del BB

        # Comprobar si está dentro del área
        inside = point_in_area(bottom_center, area_points)

        # Si el ID no existe, inicializar su estado
        if obj_id not in objects_state:
            objects_state[obj_id] = {
                "inside_curr_frame": inside,
                "inside_prev_frame": False,  # Inicializar como False
            }
        else:
            # Actualizar el estado previo y actual
            objects_state[obj_id]["inside_prev_frame"] = objects_state[obj_id]["inside_curr_frame"]
            objects_state[obj_id]["inside_curr_frame"] = inside

        # Verificar cambio de estado
        if objects_state[obj_id]["inside_prev_frame"] and not objects_state[obj_id]["inside_curr_frame"]:
            set_alert = True

        # Dibujar rectángulo y estado
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {obj_id}: {"Inside" if inside else "Outside"}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    if set_alert:
        cv2.putText(frame, 'ALERTA!!', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Escribir el frame procesado al archivo de salida
    out.write(frame)

    # Mostrar el frame procesado
    cv2.imshow('Video Procesado', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
print('Procesamiento de video completado.')
