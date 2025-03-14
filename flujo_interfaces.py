import cv2
import mediapipe as mp
import math
import time
import threading
import os

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Variables globales
current_screen = 1  # 1: Pantalla inicial, 2: Medición, 3: Mostrar número
saved_number = None  # Número basado en la distancia de las manos
selected_exercise = None  # 0: Sentadillas, 1: Lagartijas, 2: Saltos tijera
selection_start_time = None
BOX_COLOR = (254, 62, 105)
TEXT_COLOR = (63, 253, 213)
THIRD_COLOR = (61, 107, 255)
WHITE_COLOR = (255, 255, 255)

# Posiciones de los ejercicios en la pantalla
exercise_positions = [(20, 200), (20, 300), (20, 400)]


def is_finger_inside_area(finger_x, finger_y, area_x, area_y, area_w=150, area_h=50):
    """Verifica si la punta del dedo índice está dentro del área de selección."""
    return area_x <= finger_x <= area_x + area_w and area_y-50 <= finger_y <= area_y-50 + area_h


def show_text(frame, text, position=(50, 50), box_size=(250, 50), font_scale=0.7, box_color=BOX_COLOR,
              text_color=TEXT_COLOR):
    """Muestra texto en un recuadro con los colores definidos globalmente."""
    x, y = position
    w, h = box_size
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (x, y - h), (x + w, y), box_color, -1)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(frame, line, (x + 10, y - 10 + i * 25), font, font_scale, text_color, 2, cv2.LINE_AA)


def calculate_distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def detect_fist(landmarks):
    """Detecta si la mano está cerrada en puño basado en la proximidad de los dedos a la palma."""
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    fingers = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]

    distances = [calculate_distance(landmarks[f], wrist) for f in fingers]
    thumb_distance = calculate_distance(landmarks[mp_hands.HandLandmark.THUMB_TIP],
                                        landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP])

    return all(d < 0.08 for d in distances) and thumb_distance < 0.05

jump_counter_called = False

def main():
    global current_screen, saved_number, selected_exercise, selection_start_time

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Modo espejo
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hands_detected = 0
        fists_detected = 0
        wrist_positions = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hands_detected += 1
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener coordenadas del índice
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger_x, finger_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Selección de ejercicio con detección de permanencia
                if current_screen == 1:
                    for i, (ex_x, ex_y) in enumerate(exercise_positions):
                        if is_finger_inside_area(finger_x, finger_y, ex_x, ex_y):
                            if selected_exercise == i:
                                cv2.rectangle(frame, (exercise_positions[i][0], exercise_positions[i][1]-50), (exercise_positions[i][0] + 150, exercise_positions[i][1]-50 + 50), THIRD_COLOR, 10)
                                if time.time() - selection_start_time >= 3:  # 3 segundos
                                    current_screen = 2
                                    break
                            else:
                                selected_exercise = i
                                selection_start_time = time.time()

                # Detectar puño cerrado
                if detect_fist(hand_landmarks.landmark):
                    fists_detected += 1

                # Guardar posiciones de muñeca para calcular distancia
                wrist_positions.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])

            # Pantalla 2: Medición de repeticiones
            if current_screen == 2:
                if hands_detected == 2:
                    wrist_left, wrist_right = wrist_positions[:2]
                    hand_dist = calculate_distance(wrist_left, wrist_right)

                    # Normalizar distancia a rango de 1-30
                    min_dist, max_dist = 0.1, 0.6
                    normalized_dist = (hand_dist - min_dist) / (max_dist - min_dist)
                    saved_number = int(min(30, max(1, normalized_dist * 29 + 1)))

                    # Dibujar línea entre manos
                    cv2.line(frame,
                             (int(wrist_left.x * w), int(wrist_left.y * h)),
                             (int(wrist_right.x * w), int(wrist_right.y * h)),
                             THIRD_COLOR, 2)

                    # Mostrar repeticiones seleccionadas
                    show_text(frame, f"Repeticiones: {saved_number}", (50, 130), text_color=WHITE_COLOR,
                              box_color=THIRD_COLOR)

                    # Si ambas manos están en puño, pasar a la pantalla 3
                    if fists_detected == 2:
                        current_screen = 3
                        cap.release()
                        cv2.destroyAllWindows()
                        if selected_exercise ==0:
                            os.system('python pose3.py')
                        elif selected_exercise ==1:
                            os.system('python pose1.py')
                        elif selected_exercise == 2:
                            os.system('python pose2.py')

        # Mostrar pantalla actual
        if current_screen == 1:
            show_text(frame, "Selecciona tu ejercicio:", (10, 50), box_size=(300, 50))
            show_text(frame, "Usa una sola mano", (350, 50), box_size=(300, 50))
            show_text(frame, "Sentadillas", (20, 200), box_size=(150, 50), text_color=WHITE_COLOR, box_color=THIRD_COLOR)
            show_text(frame, "Lagartijas", (20, 300), box_size=(150, 50), text_color=WHITE_COLOR, box_color=THIRD_COLOR)
            show_text(frame, "Saltos tijera", (20, 400), box_size=(150, 50), text_color=WHITE_COLOR, box_color=THIRD_COLOR)

        elif current_screen == 2:
            show_text(frame, "Usa tus dos manos para elegir repeticiones", (50, 50), box_size=(510, 50))
            show_text(frame, "Cierra ambas manos para confirmar", (50, 450), box_size=(510, 50))

        elif current_screen == 3:
            show_text(frame, f"Pantalla 3: Objetivo: {saved_number} repeticiones", (50, 150))

        cv2.imshow('Interfaz Gestual', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' o 'Esc' para salir
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
