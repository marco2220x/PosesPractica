import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b):
    a = np.array(a)  # Punto A (rodilla)
    b = np.array(b)  # Punto B (tobillo)

    ab = a - b  # Vector de rodilla a tobillo
    vertical = np.array([0, 1])  # Vector vertical de referencia
    
    # Producto punto entre los vectores
    cosine_angle = np.dot(ab, vertical) / (np.linalg.norm(ab) * np.linalg.norm(vertical))
    
    # Asegurar que el valor esté en el rango [-1,1]
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)  # Convertir a grados

# Umbrales
umbral_flexion = 160  # Grado mínimo para flexión
umbral_extension = 170  # Grado mínimo para contar un salto
umbral_codo = 0.02  # Diferencia mínima entre codo y hombro en Y para verificar si está arriba

# Variables de estado
en_salto = False  
contador_saltos = 0  
brazos_arriba = False  
cuerpo_completo_detectado = False  # Nuevo estado

# Función para detectar el salto con brazos arriba
def detectar_salto(angulo_izq, angulo_der, codo_izq_arriba, codo_der_arriba):
    global en_salto, contador_saltos, brazos_arriba

    # Verifica si los codos están arriba del hombro
    if codo_izq_arriba and codo_der_arriba:
        brazos_arriba = True
    else:
        brazos_arriba = False

    # Se detecta la flexión si ambas piernas se doblan
    if angulo_izq < umbral_flexion and angulo_der < umbral_flexion and brazos_arriba:
        en_salto = True  

    # Se cuenta el salto si ambas piernas se extienden después de flexión y brazos bajan
    if en_salto and angulo_izq > umbral_extension and angulo_der > umbral_extension and not brazos_arriba:
        contador_saltos += 1  
        en_salto = False  

    return contador_saltos

# Configuración de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
jump_count = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            image_height, image_width, _ = image.shape
            
            # Verificar si el cuerpo completo está visible (piernas, cadera, torso y cabeza)
            puntos_clave = [
                mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
            ]
            cuerpo_completo_detectado = all(0 <= landmarks[p].visibility <= 1 and landmarks[p].visibility > 0.6 for p in puntos_clave)

            if cuerpo_completo_detectado:
                # Coordenadas de piernas
                left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width, 
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height)
                left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width, 
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height)
                right_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height)
                right_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width, 
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height)

                # Cálculo de ángulos de piernas
                left_leg_angle = calculate_angle(left_knee, left_ankle)
                right_leg_angle = calculate_angle(right_knee, right_ankle)

                # Coordenadas de brazos
                left_elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y
                left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                right_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y
                right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

                # Verificación de codos arriba del hombro
                codo_izq_arriba = left_elbow_y < left_shoulder_y - umbral_codo
                codo_der_arriba = right_elbow_y < right_shoulder_y - umbral_codo

                # Llamada a la función de salto
                jump_count = detectar_salto(left_leg_angle, right_leg_angle, codo_izq_arriba, codo_der_arriba)

            # Mostrar siempre el contador de saltos
            cv2.putText(image, f'Saltos: {jump_count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            if not cuerpo_completo_detectado:
                cv2.putText(image, "Cuerpo incompleto detectado", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Dibujar los landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        else:
            # Mostrar el contador de saltos aunque no se detecte el cuerpo
            cv2.putText(image, f'Saltos: {jump_count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Cuerpo no detectado", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Salto de tijera', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

