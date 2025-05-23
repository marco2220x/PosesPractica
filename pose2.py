# Saltos de tijera
import cv2
import mediapipe as mp
import numpy as np
import sys
import pyttsx3
import threading
import time

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

def contar_saltos(target_series, target_reps):
    # Umbrales
    umbral_flexion = 160  # Grado mínimo para flexión
    umbral_extension = 170  # Grado mínimo para contar un salto
    umbral_codo = 0.02  # Diferencia mínima entre codo y hombro en Y para verificar si está arriba

    # Variables de estado
    en_salto = False  
    contador_saltos = 0  
    brazos_arriba = False  
    cuerpo_completo_detectado = False  # Nuevo estado
    series_completadas = 0

    # Configuración de MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    with mp_pose.Pose(static_image_mode=False) as pose, mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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

            # 💡 Agregamos aquí la detección del dedo índice
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


            image_height, image_width, _ = frame.shape

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
                    
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Verificar si el cuerpo completo está visible
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

                    # Lógica para contar saltos
                    if codo_izq_arriba and codo_der_arriba:
                        brazos_arriba = True
                    else:
                        brazos_arriba = False

                    if left_leg_angle < umbral_flexion and right_leg_angle < umbral_flexion and brazos_arriba:
                        en_salto = True  

                    if en_salto and left_leg_angle > umbral_extension and right_leg_angle > umbral_extension and not brazos_arriba:
                        contador_saltos += 1  
                        en_salto = False  

                    # Verificar si se completó una serie
                    if contador_saltos >= target_reps:
                        series_completadas += 1
                        contador_saltos = 0  # Reiniciar contador de saltos

                        # Verificar si se completaron todas las series
                        if series_completadas >= target_series:
                            break

                # Visualización
                draw_text(image, f"Saltos: {contador_saltos}/{target_reps}", (50, 50))
                draw_text(image, f"Series: {series_completadas}/{target_series}", (50, 100))

                if not cuerpo_completo_detectado:
                    draw_text(image, "Cuerpo incompleto detectado", (50, 150), font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 0, 255))

                # Dibujar los landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            else:
                draw_text(image, "Cuerpo no detectado", (50, 150), font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 0, 255))
                speak("Cuerpo no detectado")
            cv2.imshow('Saltos de Tijera', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python pose1.py <series> <reps>")
        exit(1)
    
    series = int(sys.argv[1])
    reps = int(sys.argv[2])
    contar_saltos(series, reps)
