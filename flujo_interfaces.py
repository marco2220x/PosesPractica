import cv2
import mediapipe as mp
import math

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Variables globales
current_screen = 1  # 1: Pantalla inicial, 2: Medición, 3: Mostrar número
saved_number = None  # Número basado en la distancia de las manos
BOX_COLOR = (254, 62, 105)  # RGB(105,62,254)
TEXT_COLOR = (63, 253, 213)  # RGB(213,253,63)


def show_text(frame, text, position=(50, 50), box_size=(250, 50), font_scale=0.7):
    """Muestra texto en un recuadro con los colores definidos globalmente."""
    x, y = position
    w, h = box_size
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (x, y - h), (x + w, y), BOX_COLOR, -1)  # Fondo del recuadro
    for i, line in enumerate(text.split("\n")):
        cv2.putText(frame, line, (x + 10, y - 10 + i * 25), font, font_scale, TEXT_COLOR, 2, cv2.LINE_AA)


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
    thumb_distance = calculate_distance(landmarks[mp_hands.HandLandmark.THUMB_TIP], landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP])

    return all(d < 0.08 for d in distances) and thumb_distance < 0.05


def detect_hand_up(landmarks):
    """Detecta si la mano está levantada (por encima de la muñeca)."""
    return landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.WRIST].y


def main():
    global current_screen, saved_number

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Modo espejo (flip horizontal)
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hand_dist = None
        hands_detected = 0
        fists_detected = 0
        wrist_positions = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hands_detected += 1
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detectar si la mano está levantada en la Pantalla 1
                if current_screen == 1 and detect_hand_up(hand_landmarks.landmark):
                    current_screen = 2  # Avanzar a la Pantalla 2

                # Contar cuántas manos están en puño
                if detect_fist(hand_landmarks.landmark):
                    fists_detected += 1

                # Guardar posiciones de muñeca para calcular distancia
                wrist_positions.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST])

            # Pantalla 2: Medición de distancia entre manos
            if current_screen == 2:

                if hands_detected == 2:
                    wrist_left = wrist_positions[0]
                    wrist_right = wrist_positions[1]
                    hand_dist = calculate_distance(wrist_left, wrist_right)

                    # Normalizar distancia a rango de 1-30
                    min_dist, max_dist = 0.1, 0.6
                    normalized_dist = (hand_dist - min_dist) / (max_dist - min_dist)
                    saved_number = int(min(30, max(1, normalized_dist * 29 + 1)))

                    # Dibujar línea entre manos
                    h, w, _ = frame.shape
                    cv2.line(frame,
                             (int(wrist_left.x * w), int(wrist_left.y * h)),
                             (int(wrist_right.x * w), int(wrist_right.y * h)),
                             TEXT_COLOR, 2)

                    # Mostrar la distancia en un recuadro
                    show_text(frame, f"Repeticiones: {saved_number}", (50, 130))

                    # Si ambas manos están en puño, pasar a la pantalla 3
                    if fists_detected == 2:
                        current_screen = 3

        # Mostrar pantalla actual
        if current_screen == 1:
            show_text(frame, "Pantalla 1: Levanta la mano", (50, 50))
        elif current_screen == 3:
            show_text(frame, f"Pantalla 3: Objetivo: {saved_number} repeticiones", (50, 150))
        elif current_screen == 2:
            show_text(frame, "Usa tus dos manos para elegir repeticiones", (50, 50), box_size=(510, 50))
            show_text(frame, "Cierra ambas manos para confirmar", (50, 450), box_size=(510, 50))

        cv2.imshow('Interfaz Gestual', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' o 'Esc' para salir
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
