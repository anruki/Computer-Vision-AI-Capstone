from ultralytics import YOLO
import cv2

# Cargar modelo YOLOv8
model = YOLO('yolov8n.pt')

# Ruta del video de entrada
video_path = 'p3/ins/sunday_sale.mp4'

# Cargar el video
cap = cv2.VideoCapture(video_path)

# Obtener propiedades del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear un objeto VideoWriter para guardar el video de salida
output_path = 'p3/outs/sunday_person_yolo.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Verificar si el video se cargó correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Procesamiento del video
while True:
    # Leer un frame
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error al cargar el frame.")
        break

    # Tracking con YOLOv8
    results = model.track(frame, persist=True, classes=[0, 17])
    human = False

    # Procesar detecciones
    for result in results[0].boxes:  # Acceder a las cajas detectadas
        label = model.names[int(result.cls)]  # Obtener el nombre de la clase
        if label == 'person':
            # Coordenadas del cuadro delimitador
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            # Dibujar el cuadro y la etiqueta
            color = (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            human = True

        if label == 'horse':  # Filtrar solo caballos
            # Coordenadas del cuadro delimitador
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            # Determinar el color del cuadro según la posición
            center_x = (x1 + x2) // 2
            if human or center_x < width:  # Si hay una persona en el frame o el caballo está en el cercado
                color = (0, 255, 0)  # Verde
            else:
                color = (0, 0, 255)
                cv2.putText(frame, 'ALERTA!! HORSE ESCAPED!!', 
                            (10, 50),               # Posición del texto en la parte superior
                            cv2.FONT_HERSHEY_SIMPLEX,  # Fuente
                            1.5,                   # Tamaño grande
                            (255, 255, 255),       # Color blanco
                            3)                     # Grosor del texto
            # Dibujar el cuadro y la etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Escribir el frame procesado en el archivo de salida
    out.write(frame)

    # Mostrar el frame procesado
    cv2.imshow('Video Procesado', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()  # Cerrar el archivo de video de salida
cv2.destroyAllWindows()
print('Procesamiento de video completado.')
