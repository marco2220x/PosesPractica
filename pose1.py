import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import pyttsx3
import threading
import time

from audio_manage import reproducir_audio_poses


# Configuración de texto a voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidad del habla
last_audio_time = 0
audio_cooldown = 2  # Segundos entre mensajes de audio para evitar saturación



def speak(text):
    """Función para convertir texto a voz con cooldown"""
    global last_audio_time
    current_time = time.time()
    
    if current_time - last_audio_time > audio_cooldown:
        last_audio_time = current_time
        # Ejecutar en un hilo para no bloquear la interfaz
        threading.Thread(target=lambda: engine.say(text)).start()
        threading.Thread(target=lambda: engine.runAndWait()).start()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

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
    reproducir_audio_poses("start")

    # Variables de estado
    pushup_count = 0
    series_count = 0
    pushup_phase = "up"
    ANGLE_MIN = 90  # Ángulo mínimo para considerar posición baja
    ANGLE_MAX = 160  # Ángulo máximo para considerar posición alta
    HORIZONTAL_THRESHOLD = 0.3  # Umbral para considerar posición horizontal
    series_completed = False
    workout_completed = False
    last_horizontal_state = None
    last_body_detected = True
    
    cv2.namedWindow('PushUp Counter', cv2.WINDOW_NORMAL)
    
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(static_image_mode=False) as pose, mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Aplicar efecto espejo (flip horizontal)
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Definir posición y tamaño del botón en la parte inferior izquierda
            button_x, button_w = 10, 150
            button_h = 50
            button_y = height - button_h - 10  # 10 píxeles desde el borde inferior

            overlay = frame.copy()
            cv2.rectangle(overlay, (button_x, button_y), (button_x + button_w, button_y + button_h), (0, 255, 255), -1)  # Amarillo
            alpha = 0.6  # Nivel de transparencia
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Sombra (ligeramente desplazada)
            shadow_offset = 5
            shadow_color = (0, 100, 200)
            cv2.rectangle(frame, 
              (button_x + shadow_offset, button_y + shadow_offset), 
              (button_x + button_w + shadow_offset, button_y + button_h + shadow_offset), 
              shadow_color, -1, cv2.LINE_AA)

            # Fondo del botón
            button_color = (0, 140, 255)  # Naranja fuerte
            cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), button_color, -1, cv2.LINE_AA)

            # Borde blanco
            cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), (255, 255, 255), 2, cv2.LINE_AA)

            # Texto centrado
            text = "Salir"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = button_x + (button_w - text_width) // 2
            text_y = button_y + (button_h + text_height) // 2 - 5
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Procesamiento de manos (convertir a RGB solo para el procesamiento)
            hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if hand_results.multi_hand_landmarks:
                 for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    # Verifica si es la mano derecha
                        label = handedness.classification[0].label
                        if label == 'Left':
                            index_finger_tip = hand_landmarks.landmark[8]
                            x_tip = int(index_finger_tip.x * width)
                            y_tip = int(index_finger_tip.y * height)

                            cv2.circle(frame, (x_tip, y_tip), 15, (0, 255, 0), -1)
                    # Detectar si dedo está sobre el botón
                            if button_x <= x_tip <= button_x + button_w and button_y <= y_tip <= button_y + button_h:
                                speak("Regresando al menú")
                                cap.release()
                                cv2.destroyAllWindows()
                                return True

            # Procesamiento de pose (convertir a RGB solo para el procesamiento)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Usar frame directamente para visualización (ya está en BGR)
            image = frame.copy()
            
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
                
                # Retroalimentación auditiva para posición horizontal
                if is_horizontal != last_horizontal_state:
                    if is_horizontal:
                        speak("Posición horizontal correcta")
                    else:
                        speak("Por favor, colócate en posición horizontal")
                    last_horizontal_state = is_horizontal
                
                # Retroalimentación cuando se detecta el cuerpo después de no detectarlo
                if not last_body_detected:
                    speak("Cuerpo detectado")
                    last_body_detected = True
                
                # Lógica para contar repeticiones solo si está en posición horizontal
                if is_horizontal and not workout_completed:
                    if angle > ANGLE_MAX and pushup_phase == "down":
                        pushup_phase = "up"
                        # Solo incrementar el contador si no hemos completado la serie
                        if not series_completed:
                            pushup_count += 1
                            reproducir_audio_poses("correct")
                            # Verificar si se completó la serie
                            if pushup_count == target_reps:
                                series_completed = True
                                reproducir_audio_poses("serie")
                                speak("¡Serie completada! Descansa antes de continuar")
                                draw_text(image, "Serie completada!", (image.shape[1]//2 - 100, 200), 
                                         font_scale=1.2, text_color=(0, 0, 0), bg_color=(0, 255, 0))
                                
                    elif angle < ANGLE_MIN and pushup_phase == "up":
                        pushup_phase = "down"
                        # Si la serie está completada y el usuario baja, comenzar nueva serie
                        if series_completed and pushup_phase == "down":
                            series_count += 1
                            pushup_count = 0
                            series_completed = False
                            
                            if series_count >= target_series:
                                workout_completed = True
                                reproducir_audio_poses("victory")
                                speak("¡Entrenamiento completado! Buen trabajo")
                else:
                    pushup_phase = "up"  # Reiniciar fase si no está en posición horizontal
                
                # Verificar si se completaron todas las series
                if series_count >= target_series:
                    draw_text(image, "Todas las series completadas!", (image.shape[1]//2 - 150, 200), 
                             font_scale=1.2, text_color=(0, 0, 0), bg_color=(0, 255, 0))
                
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
                if last_body_detected:
                    speak("Por favor, colócate frente a la cámara")
                    last_body_detected = False

            # Mostrar siempre el conteo de flexiones y series
            draw_text(image, f"Flexiones: {pushup_count}/{target_reps}", (50, 50))
            draw_text(image, f"Series: {series_count}/{target_series}", (50, 100))

            cv2.imshow('PushUp Counter', image)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC
                # Limpiar completamente antes de salir
                cap.release()
                cv2.destroyWindow('PushUp Counter')
                return True  # Salir de la función inmediatamente

    cap.release()
    cv2.destroyAllWindows()
    return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python pose1.py <series> <reps>")
        exit(1)
    
    series = int(sys.argv[1])
    reps = int(sys.argv[2])
    contar_pushups(series, reps)
