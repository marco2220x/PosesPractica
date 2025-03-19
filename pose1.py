import cv2
import mediapipe as mp
import numpy as np
import sys
import os  # Para manejar rutas de archivos

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_text(image, text, position, font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 165, 255), border_color=(255, 255, 255)):
    """Dibuja un texto con fondo y contorno para mejor visibilidad."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    cv2.rectangle(image, (x - 6, y - text_height - 6), (x + text_width + 6, y + 6), border_color, -1)
    cv2.rectangle(image, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def calculate_angle(a, b, c):
    a = np.array(a)  # Primer punto (hombro)
    b = np.array(b)  # Punto central (codo)
    c = np.array(c)  # Tercer punto (muñeca)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def contar_pushups(target_series, target_reps):
    # Variables de estado
    pushup_count = 0
    series_count = 0
    pushup_phase = "up"
    ANGLE_MIN = 90  # Ángulo mínimo para considerar posición baja
    ANGLE_MAX = 160  # Ángulo máximo para considerar posición alta
    HORIZONTAL_THRESHOLD = 0.3  # Umbral para considerar posición horizontal

    # Crear una carpeta para guardar las capturas de pantalla
    if not os.path.exists("capturas_flexiones"):
        os.makedirs("capturas_flexiones")

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
                        
                        # Dibujar todos los elementos antes de tomar la captura
                        draw_text(image, f"Flexiones: {pushup_count}/{target_reps}", (50, 50))
                        draw_text(image, f"Series: {series_count}/{target_series}", (50, 100))
                        draw_text(image, "Horizontal", (50, 150), font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 255, 0))
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        
                        # Tomar una captura de pantalla cuando se completa una repetición
                        cv2.imwrite(f"capturas_flexiones/flexion_{pushup_count}.jpg", image)
                        print(f"Captura de pantalla guardada: flexion_{pushup_count}.jpg")
                    elif angle < ANGLE_MIN and pushup_phase == "up":
                        pushup_phase = "down"
                else:
                    pushup_phase = "up"  # Reiniciar fase si no está en posición horizontal
                
                # Verificar si se completó una serie
                if pushup_count >= target_reps:
                    series_count += 1
                    pushup_count = 0  # Reiniciar contador de repeticiones
                    
                    # Verificar si se completaron todas las series
                    if series_count >= target_series:
                        break
                
                # Mostrar mensaje de posición horizontal
                if is_horizontal:
                    draw_text(image, "Horizontal", (50, 150), font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 255, 0))
                else:
                    draw_text(image, "No Horizontal", (50, 150), font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 0, 255))

                # Dibujar landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                # Si no se detectan landmarks, mostrar mensaje de error
                draw_text(image, "Cuerpo no detectado", (50, 150), font_scale=0.8, text_color=(255, 255, 255), bg_color=(255, 0, 0))

            # Mostrar siempre el conteo de flexiones y series
            draw_text(image, f"Flexiones: {pushup_count}/{target_reps}", (50, 50))
            draw_text(image, f"Series: {series_count}/{target_series}", (50, 100))

            cv2.imshow('PushUp Counter', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python pose1.py <series> <reps>")
        exit(1)
    
    series = int(sys.argv[1])
    reps = int(sys.argv[2])
    contar_pushups(series, reps)
