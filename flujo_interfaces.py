import cv2
import mediapipe as mp
import time
import numpy as np
import pose1, pose2, pose3
import platform
from audio_manage import reproducir_audio_menu

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Configuración de modos predefinidos
WORKOUT_MODES = {
    "facil": {"series": 2, "reps": 3, "desc": " "},
    "medio": {"series": 3, "reps": 5, "desc": " "}, 
    "avanzado": {"series": 4, "reps": 8, "desc": " "}
}

# Variables de estado
current_screen = 0  # 0: Bienvenida, 1: Ejercicio, 2: Modo, 3: Ejecución
selected_exercise = None
selected_mode = None
selection_start_time = None
back_button_selected = False
welcome_alpha = 1.0  # Transparencia del texto de bienvenida
welcome_detected = False  # Si se ha detectado el gesto de saludo
select_index_sound = 1
back_button_was_hovered = False

# Diseño de interfaz
COLORES = {
    "primario": (254, 62, 105),
    "secundario": (63, 253, 213),
    "terciario": (61, 107, 255),
    "texto": (255, 255, 255),
    "fondo": (50, 50, 50),
    "atras": (255, 100, 0),  # Naranja para el botón de regreso
    "bienvenida": (255, 215, 0)  # Dorado para el texto de bienvenida
}

# Configuración de diseño
ANCHO_RECUADRO = 300
ALTO_RECUADRO = 80
ALTO_BOTON_ATRAS = 50
MARGEN_SUPERIOR = 100
ESPACIADO = 20
MARGEN_IZQUIERDO = 20




def calcular_posiciones(num_opciones):
    """Calcula posiciones centradas verticalmente para las opciones"""
    posiciones = []
    for i in range(num_opciones):
        y = MARGEN_SUPERIOR + (ALTO_RECUADRO + ESPACIADO) * i
        posiciones.append((MARGEN_IZQUIERDO, y))
    return posiciones

# Posiciones para los elementos de la interfaz
POSICIONES = {
    "ejercicios": calcular_posiciones(3),
    "modos": calcular_posiciones(3),
    "atras": (MARGEN_IZQUIERDO, MARGEN_SUPERIOR + (ALTO_RECUADRO + ESPACIADO) * 3 + 20)
}

def mostrar_texto(frame, texto, posicion, tamaño=(ANCHO_RECUADRO, ALTO_RECUADRO), color_fondo=None, color_texto=None, transparencia=0.6):
    """Muestra texto en un recuadro semi-transparente"""
    color_fondo = color_fondo or COLORES["primario"]
    color_texto = color_texto or COLORES["texto"]
    x, y = posicion
    w, h = tamaño
    
    # Crear overlay para transparencia
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color_fondo, -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    # Aplicar transparencia
    cv2.addWeighted(overlay, transparencia, frame, 1 - transparencia, 0, frame)
    
    # Dibujar texto
    lineas = texto.split("\n")
    altura_total_texto = len(lineas) * 30
    y_texto = y + (h - altura_total_texto) // 2 + 25
    
    for i, linea in enumerate(lineas):
        cv2.putText(frame, linea, (x + 10, y_texto + i * 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2, cv2.LINE_AA)

def dibujar_boton_atras(frame, seleccionado=False):
    """Dibuja el botón de regreso al menú principal"""
    x, y = POSICIONES["atras"]
    color = COLORES["terciario"] if seleccionado else COLORES["atras"]
    mostrar_texto(frame, "<< REGRESAR", (x, y), (ANCHO_RECUADRO, ALTO_BOTON_ATRAS), color_fondo=color)

def dibujar_bienvenida(frame, alpha):
    """Dibuja la pantalla de bienvenida con efecto de desvanecimiento"""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Fondo semi-transparente
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    
    # Texto de bienvenida
    texto = "BIENVENIDO AL ENTRENADOR VIRTUAL"
    tamaño = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
    x = (w - tamaño[0]) // 2
    y = (h - tamaño[1]) // 2
    
    cv2.putText(overlay, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORES["bienvenida"], 4, cv2.LINE_AA)
    
    # Instrucciones
    instruccion = "Saluda con la mano para continuar"
    tamaño_inst = cv2.getTextSize(instruccion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    x_inst = (w - tamaño_inst[0]) // 2
    y_inst = y + tamaño[1] + 40
    
    cv2.putText(overlay, instruccion, (x_inst, y_inst), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Aplicar transparencia
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def detectar_saludo(hand_landmarks):
    """Detecta si el usuario está haciendo un gesto de saludo"""
    # Obtener landmarks de los dedos
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Verificar si los dedos están extendidos (saludo)
    fingers_extended = [
        index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,  # Dedo medio no extendido
        ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,  # Anular no extendido
        pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    ]
    
    # Saludo es pulgar extendido, índice y meñique extendidos, medio y anular cerrados
    return (fingers_extended[0] and not fingers_extended[1] and not fingers_extended[2] and fingers_extended[3])

def main():
    global current_screen, selected_exercise, selected_mode, selection_start_time, back_button_selected
    global welcome_alpha, welcome_detected
    global back_button_was_hovered, select_index_sound


    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    ejercicios = ["Sentadillas", "Lagartijas", "Saltos"]
    modos = list(WORKOUT_MODES.keys())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Procesar landmarks de manos
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Reiniciar estado del botón de regreso
        back_button_selected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Obtener posición del dedo índice
                dedo_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                px, py = int(dedo_indice.x * w), int(dedo_indice.y * h)
                
                # Dibujar círculo en el dedo índice
                cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)

                # Lógica para pantalla de bienvenida
                if current_screen == 0:
                    if detectar_saludo(hand_landmarks):
                        welcome_detected = True
                
                # Lógica de selección para otras pantallas
                elif current_screen == 1:  # Selección de ejercicio
                    for i, (ex_x, ex_y) in enumerate(POSICIONES["ejercicios"]):
                        if ex_x <= px <= ex_x + ANCHO_RECUADRO and ex_y <= py <= ex_y + ALTO_RECUADRO:
                            if selected_exercise == i:
                                if time.time() - selection_start_time >= 2:
                                    select_index_sound = reproducir_audio_menu("selected", 1)
                                    current_screen = 2
                            else:
                                selected_exercise = i
                                selection_start_time = time.time()
                                select_index_sound = reproducir_audio_menu("select", select_index_sound)

                elif current_screen == 2:  # Selección de modo
                    # Verificar selección de dificultad
                    for i, (modo_x, modo_y) in enumerate(POSICIONES["modos"]):
                        if modo_x <= px <= modo_x + ANCHO_RECUADRO and modo_y <= py <= modo_y + ALTO_RECUADRO:
                            if selected_mode == i:
                                if time.time() - selection_start_time >= 2:
                                    select_index_sound = reproducir_audio_menu("selected", 1)
                                    current_screen = 3
                                    # Iniciar ejercicio
                                    modo = modos[i]
                                    series = WORKOUT_MODES[modo]["series"]
                                    reps = WORKOUT_MODES[modo]["reps"]
                                    if selected_exercise == 0:
                                        cv2.destroyAllWindows()
                                        cap.release()
                                        should_return = pose3.countSquats(series, reps)
                                        if should_return:
                                            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                                            current_screen = 1  # ← Esto regresa al menú principal
                                            continue  # ← Esto reinicia el ciclo principal
                                        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                                    elif selected_exercise == 1:
                                        cv2.destroyWindow('Entrenador Virtual')
                                        cap.release()
                                        should_return = pose1.contar_pushups(series, reps)
                                        if should_return:
                                            # Volver a crear la ventana principal
                                            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                                            current_screen = 1  # Volver al menú principal
                                            continue
                                          
  
                                    elif selected_exercise == 2:
                                        cv2.destroyAllWindows()
                                        cap.release()
                                        should_return = pose2.contar_saltos(series, reps)
                                        if should_return:
                                            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                                            current_screen = 1  # Volver al menú principal
                                            continue  # Reiniciar el ciclo principal
                                        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                                        current_screen = 1
                            else:
                                selected_mode = i
                                selection_start_time = time.time()
                                select_index_sound = reproducir_audio_menu("select", select_index_sound)
                    
                    # Verificar selección del botón de regreso
                    back_x, back_y = POSICIONES["atras"]
                    is_hovering_back = back_x <= px <= back_x + ANCHO_RECUADRO and back_y <= py <= back_y + ALTO_BOTON_ATRAS

                    if is_hovering_back:
                        back_button_selected = True

                        # Solo reproducir si el usuario acaba de entrar al botón
                        if not back_button_was_hovered:
                            select_index_sound = reproducir_audio_menu("back_on", 1)
    
                        back_button_was_hovered = True

                        if time.time() - selection_start_time >= 2:
                            select_index_sound = reproducir_audio_menu("back", 1)
                            current_screen = 1
                            selected_mode = None
                    else:
                        if back_button_selected:
                            selection_start_time = time.time()
                            back_button_selected = False
    
                        # Marcar que ya no está sobre el botón
                        back_button_was_hovered = False

        # Animación de desvanecimiento si se detectó el saludo
        if current_screen == 0 and welcome_detected:
            welcome_alpha -= 0.02  # Ajusta la velocidad de desvanecimiento
            if welcome_alpha <= 0:
                current_screen = 1  # Pasar al menú principal
                welcome_alpha = 0

        # Renderizado de interfaz
        if current_screen == 0:
            dibujar_bienvenida(frame, welcome_alpha)
        elif current_screen == 1:
            mostrar_texto(frame, "SELECCIONA EJERCICIO", (MARGEN_IZQUIERDO, 30), (ANCHO_RECUADRO, 50), COLORES["secundario"])
            for i, (x, y) in enumerate(POSICIONES["ejercicios"]):
                color = COLORES["terciario"] if selected_exercise == i else COLORES["primario"]
                mostrar_texto(frame, ejercicios[i], (x, y), color_fondo=color)

        elif current_screen == 2:
            mostrar_texto(frame, "ELIGE DIFICULTAD", (MARGEN_IZQUIERDO, 30), (ANCHO_RECUADRO, 50), COLORES["secundario"])
            for i, (x, y) in enumerate(POSICIONES["modos"]):
                modo = modos[i]
                config = WORKOUT_MODES[modo]
                texto = f"{modo.upper()}\n{config['series']} series x {config['reps']} repeticiones\n{config['desc']}"
                color = COLORES["terciario"] if selected_mode == i else COLORES["primario"]
                mostrar_texto(frame, texto, (x, y), color_fondo=color)
            
            # Dibujar botón de regreso
            dibujar_boton_atras(frame, back_button_selected)

        cv2.imshow('Entrenador Virtual', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
