import cv2
import mediapipe as mp
import numpy as np
import sys
from math import acos, degrees
import pyttsx3
import threading
import time

# Configuración de texto a voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_audio_time = 0
audio_cooldown = 2  # Segundos

def speak(text):
    """Función para convertir texto a voz con cooldown"""
    global last_audio_time
    current_time = time.time()
    if current_time - last_audio_time > audio_cooldown:
        last_audio_time = current_time
        threading.Thread(target=lambda: engine.say(text)).start()
        threading.Thread(target=lambda: engine.runAndWait()).start()

def draw_text(image, text, position, font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 165, 255), border_color=(255, 255, 255)):
    """Dibuja un texto con fondo y contorno"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(image, (x - 6, y - text_height - 6), (x + text_width + 6, y + 6), border_color, -1)
    cv2.rectangle(image, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def countSquats(target_series, target_reps):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    # Variables de estado
    squat_count = 0
    series_count = 0
    squat_phase = "up"  # "up" o "down"
    ANGLE_UP = 160      # Ángulo para considerar posición de pie
    ANGLE_DOWN = 70     # Ángulo para considerar posición en cuclillas
    
    # Control de retroalimentación
    last_posture_state = None
    last_detection_time = 0
    DETECTION_REMINDER_INTERVAL = 5  # Segundos entre recordatorios
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Squat Counter', cv2.WINDOW_NORMAL)
    
    with mp_pose.Pose(static_image_mode=False) as pose, mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Efecto espejo para mejor experiencia de usuario
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            # --- Configuración del botón de salida ---
            button_x, button_w = 10, 150
            button_h = 50
            button_y = height - button_h - 10
            
            # Sombra del botón
            shadow_offset = 5
            cv2.rectangle(frame, 
              (button_x + shadow_offset, button_y + shadow_offset), 
              (button_x + button_w + shadow_offset, button_y + button_h + shadow_offset), 
              (0, 100, 200), -1, cv2.LINE_AA)

            # Botón principal
            button_color = (0, 140, 255)  # Naranja
            cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), button_color, -1, cv2.LINE_AA)
            cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), (255, 255, 255), 2, cv2.LINE_AA)

            # Texto del botón (corregido el acento)
            text = "Salir"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)
            text_x = button_x + (button_w - text_width) // 2
            text_y = button_y + (button_h + text_height) // 2 - 5
            cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # ----------------------------------------

            # Detección de manos para el botón de salida
            hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    if handedness.classification[0].label == 'Left':
                        index_finger_tip = hand_landmarks.landmark[8]
                        x_tip = int(index_finger_tip.x * width)
                        y_tip = int(index_finger_tip.y * height)
                        
                        cv2.circle(frame, (x_tip, y_tip), 15, (0, 255, 0), -1)
                        
                        if button_x <= x_tip <= button_x + button_w and button_y <= y_tip <= button_y + button_h:
                            speak("Regresando al menu")
                            cap.release()
                            cv2.destroyAllWindows()
                            return True

            # Procesamiento de la pose
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = frame.copy()
            current_time = time.time()
            current_posture_state = None
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                required_landmarks = [mp_pose.PoseLandmark.LEFT_HIP.value, 
                                     mp_pose.PoseLandmark.LEFT_KNEE.value, 
                                     mp_pose.PoseLandmark.LEFT_ANKLE.value]
                
                # Verificar visibilidad de los landmarks clave
                if all(landmarks[i].visibility > 0.5 for i in required_landmarks):
                    current_posture_state = "correct"
                    
                    # Obtener coordenadas de los puntos clave
                    hip = [int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height)]
                    knee = [int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height)]
                    ankle = [int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height)]
                    
                    # Calcular ángulo de la rodilla
                    p1 = np.array(hip)
                    p2 = np.array(knee)
                    p3 = np.array(ankle)
                    
                    l1 = np.linalg.norm(p2 - p3)  # Rodilla-tobillo
                    l2 = np.linalg.norm(p1 - p3)  # Cadera-tobillo
                    l3 = np.linalg.norm(p1 - p2)  # Cadera-rodilla
                    
                    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                    
                    # Lógica para contar sentadillas
                    if angle >= ANGLE_UP and squat_phase == "down":
                        squat_phase = "up"
                        squat_count += 1
                        speak("Bien hecho")
                        
                        # Verificar si se completó la serie
                        if squat_count == target_reps:
                            series_count += 1
                            squat_count = 0
                            speak(f"Serie {series_count} completada")
                            
                    elif angle <= ANGLE_DOWN and squat_phase == "up":
                        squat_phase = "down"
                    
                    # Dibujar landmarks y conexiones
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                    
                else:
                    current_posture_state = "incorrect"
                    draw_text(image, "Ajusta tu posicion", (50, 150),  # Corregido sin acento
                             font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 0, 255))
            else:
                current_posture_state = "not_detected"
                draw_text(image, "Cuerpo no detectado", (50, 150), 
                         font_scale=0.8, text_color=(255, 255, 255), bg_color=(255, 0, 0))

            # Control de retroalimentación por cambio de estado
            if current_posture_state != last_posture_state:
                if current_posture_state == "correct":
                    speak("Posicion correcta detectada")  # Corregido sin acento
                    last_detection_time = current_time
                elif current_posture_state == "not_detected":
                    speak("Por favor, colocaté frente a la camara")
                    last_detection_time = current_time
                elif current_posture_state == "incorrect":
                    speak("Ajusta tu posicion para continuar")
                    last_detection_time = current_time
                
                last_posture_state = current_posture_state
            else:
                # Recordatorio periódico si el cuerpo no es detectado
                if current_posture_state == "not_detected" and (current_time - last_detection_time) >= DETECTION_REMINDER_INTERVAL:
                    speak("Por favor, colocaté frente a la camara")
                    last_detection_time = current_time

            # Mostrar contadores
            draw_text(image, f"Sentadillas: {squat_count}/{target_reps}", (50, 50))
            draw_text(image, f"Series: {series_count}/{target_series}", (50, 100))
            
                        # Mostrar mensaje de finalización
            if series_count >= target_series:
                draw_text(image, "¡Entrenamiento completado!", (width//2 - 180, 200), 
                         font_scale=1.2, text_color=(0, 0, 0), bg_color=(0, 255, 0))
                speak("¡Entrenamiento completado! Buen trabajo")
                cv2.imshow('Squat Counter', image)
                cv2.waitKey(3000)  # Esperar 3 segundos antes de salir
                cap.release()
                cv2.destroyAllWindows()
                return True  # Permitir regreso al menú

            cv2.imshow('Squat Counter', image)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # Tecla ESC
                if series_count >= target_series:
                    speak("Regresando al menu")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                else:
                    draw_text(image, "Completa la rutina antes de salir", (50, height - 80), bg_color=(0, 0, 255))
                    speak("Aún no terminas la rutina")


    cap.release()
    cv2.destroyAllWindows()
    return False  # No necesitamos retornar True aquí ya que el botón de salida maneja el retorno

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python squat_counter.py <series> <reps>")
        exit(1)
    
    series = int(sys.argv[1])
    reps = int(sys.argv[2])
    countSquats(series, reps)
