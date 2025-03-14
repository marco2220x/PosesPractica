import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Colocar todo en una función 
def calculate_angle(a, b, c):
    a = np.array(a)  # Primer punto (hombro)
    b = np.array(b)  # Punto central (codo)
    c = np.array(c)  # Tercer punto (muñeca)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Variables de estado
pushup_count = 0
pushup_phase = "up"
ANGLE_MIN = 90  # Ángulo mínimo para considerar posición baja
ANGLE_MAX = 160  # Ángulo máximo para considerar posición alta
HORIZONTAL_THRESHOLD = 0.3  # Umbral para considerar posición horizontal

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Obtener coordenadas del brazo derecho
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Obtener coordenadas de la cadera
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Calcular ángulo del codo
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Verificar si el cuerpo está en posición horizontal
            shoulder_y = shoulder[1]
            hip_y = hip[1]
            is_horizontal = abs(shoulder_y - hip_y) < HORIZONTAL_THRESHOLD
            
            # Lógica para contar repeticiones solo si está en posición horizontal
            if is_horizontal:
                if angle > ANGLE_MAX and pushup_phase == "down":
                    pushup_count += 1
                    pushup_phase = "up"
                elif angle < ANGLE_MIN and pushup_phase == "up":
                    pushup_phase = "down"
            else:
                pushup_phase = "up"  # Reiniciar fase si no está en posición horizontal
            
            # Visualización
            cv2.putText(image, f"Conteo: {pushup_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
            cv2.putText(image, f"Posicion: {'Horizontal' if is_horizontal else 'No horizontal'}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Dibujar landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('PushUp Counter', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
