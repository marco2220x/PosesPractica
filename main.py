import cv2
import mediapipe as mp
import subprocess

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener posición de los dedos
                landmarks = hand_landmarks.landmark
                index_finger = landmarks[8].y
                middle_finger = landmarks[12].y
                ring_finger = landmarks[16].y
                pinky_finger = landmarks[20].y

                # Gesto: índice y medio arriba → Saltos de tijera
                if index_finger < landmarks[6].y and middle_finger < landmarks[10].y and ring_finger > landmarks[14].y and pinky_finger > landmarks[18].y:
                    print("Saltos de tijera seleccionado")
                    cap.release()
                    cv2.destroyAllWindows()
                    subprocess.run(["python", "/home/andoni/mediapipe/venv/PosesPractica/pose2.py"])  # Ejecuta pose2.py
                    break

                # Gesto: Puño cerrado → Flexiones
                if index_finger > landmarks[6].y and middle_finger > landmarks[10].y and ring_finger > landmarks[14].y and pinky_finger > landmarks[18].y:
                    print("Flexiones seleccionadas")
                    cap.release()
                    cv2.destroyAllWindows()
                    subprocess.run(["python", "/home/andoni/mediapipe/venv/PosesPractica/pose1.py"])  # Ejecuta pose1.py
                    break

        cv2.putText(frame, "Abre indice y medio -> Saltos", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Cierra la mano -> Flexiones", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Menu de ejercicios", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
            break

cap.release()
cv2.destroyAllWindows()
