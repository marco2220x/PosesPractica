# ------------ Interfaz Principal Modificada (gesture_ui.py) ------------
import cv2
import mediapipe as mp
import math
import time
import os
import pose1

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Configuración de modos predefinidos
WORKOUT_MODES = {
    "facil": {"series": 2, "reps": 3},
    "medio": {"series": 3, "reps": 5}, 
    "avanzado": {"series": 4, "reps": 8}
}

# Variables de estado
current_screen = 1  # 1: Ejercicio, 2: Modo, 3: Ejecución
selected_exercise = None
selected_mode = None
selection_start_time = None

# Diseño de interfaz
COLORES = {
    "primario": (254, 62, 105),
    "secundario": (63, 253, 213),
    "terciario": (61, 107, 255),
    "texto": (255, 255, 255)
}

POSICIONES = {
    "ejercicios": [(20, 200), (20, 300), (20, 400)],
    "modos": [(20, 150), (20, 250), (20, 350)]
}

def mostrar_texto(frame, texto, posicion, tamaño=(250, 50), color_fondo=None, color_texto=None):
    color_fondo = color_fondo or COLORES["primario"]
    color_texto = color_texto or COLORES["texto"]
    x, y = posicion
    w, h = tamaño
    cv2.rectangle(frame, (x, y - h), (x + w, y), color_fondo, -1)
    for i, linea in enumerate(texto.split("\n")):
        cv2.putText(frame, linea, (x + 10, y - 10 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2, cv2.LINE_AA)

def detectar_puno(landmarks):
    """Detecta si la mano está cerrada en puño basado en la proximidad de los dedos a la palma."""
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    fingers = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]

jump_counter_called = False

def main():
    global current_screen, selected_exercise, selected_mode, selection_start_time

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ejercicios = ["Sentadillas", "Lagartijas", "Saltos"]
    modos = list(WORKOUT_MODES.keys())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                dedo_indice = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                px, py = int(dedo_indice.x * w), int(dedo_indice.y * h)

                # Lógica de selección
                if current_screen == 1:  # Selección de ejercicio
                    for i, (ex_x, ex_y) in enumerate(POSICIONES["ejercicios"]):
                        if ex_x <= px <= ex_x + 150 and ex_y-50 <= py <= ex_y:
                            if selected_exercise == i:
                                if time.time() - selection_start_time >= 2:
                                    current_screen = 2
                            else:
                                selected_exercise = i
                                selection_start_time = time.time()

                elif current_screen == 2:  # Selección de modo
                    for i, (modo_x, modo_y) in enumerate(POSICIONES["modos"]):
                        if modo_x <= px <= modo_x + 150 and modo_y-50 <= py <= modo_y:
                            if selected_mode == i:
                                if time.time() - selection_start_time >= 2:
                                    current_screen = 3
                                    # Iniciar ejercicio con parámetros predefinidos
                                    modo = modos[i]
                                    series = WORKOUT_MODES[modo]["series"]
                                    reps = WORKOUT_MODES[modo]["reps"]
                                    if selected_exercise == 0:
                                        print('ejercicio no disp.')
                                    elif selected_exercise == 1:
                                        cv2.destroyAllWindows()
                                        cap.release()
                                        pose1.contar_pushups(series, reps)
                                        # Reiniciar la interfaz después del ejercicio
                                        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                                        pose1.contar_pushups(series, reps)    
                                    elif selected_exercise == 2:
                                        print('ejercicio no disp.')
                 
                            else:
                                selected_mode = i
                                selection_start_time = time.time()

        # Renderizado de interfaz
        if current_screen == 1:
            mostrar_texto(frame, "Selecciona ejercicio:", (10, 50))
            for i, (x, y) in enumerate(POSICIONES["ejercicios"]):
                color = COLORES["terciario"] if selected_exercise == i else COLORES["primario"]
                mostrar_texto(frame, ejercicios[i], (x, y), color_fondo=color)

        elif current_screen == 2:
            mostrar_texto(frame, "Elige dificultad:", (10, 50))
            for i, (x, y) in enumerate(POSICIONES["modos"]):
                modo = modos[i]
                config = WORKOUT_MODES[modo]
                texto = f"{modo.capitalize()}\n{config['series']}x{config['reps']}"
                color = COLORES["terciario"] if selected_mode == i else COLORES["primario"]
                mostrar_texto(frame, texto, (x, y), color_fondo=color)

        cv2.imshow('Entrenador Virtual', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
