import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)  # Usar la cámara en vivo
up = False
down = False
count = 0

with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Verificar que todos los puntos clave estén presentes
        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Puntos clave para la detección de sentadillas
                required_landmarks = [24, 26, 28]  # Cadera, rodilla, tobillo
                if all(landmarks[i].visibility > 0.5 for i in required_landmarks):
                    x1, y1 = int(landmarks[24].x * width), int(landmarks[24].y * height)
                    x2, y2 = int(landmarks[26].x * width), int(landmarks[26].y * height)
                    x3, y3 = int(landmarks[28].x * width), int(landmarks[28].y * height)

                    p1 = np.array([x1, y1])
                    p2 = np.array([x2, y2])
                    p3 = np.array([x3, y3])

                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)

                    # Calcular el ángulo
                    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                    if angle >= 160:
                        up = True
                    if up and not down and angle <= 70:
                        down = True
                    if up and down and angle >= 160:
                        count += 1
                        up = False
                        down = False

                    # Visualización
                    aux_image = np.zeros(frame.shape, np.uint8)
                    cv2.line(aux_image, (x1, y1), (x2, y2), (255, 255, 0), 20)
                    cv2.line(aux_image, (x2, y2), (x3, y3), (255, 255, 0), 20)
                    cv2.line(aux_image, (x1, y1), (x3, y3), (255, 255, 0), 5)
                    contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                    cv2.fillPoly(aux_image, pts=[contours], color=(128, 0, 250))

                    output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)

                    cv2.circle(output, (x1, y1), 6, (0, 255, 255), 4)
                    cv2.circle(output, (x2, y2), 6, (128, 0, 250), 4)
                    cv2.circle(output, (x3, y3), 6, (255, 191, 0), 4)
                    cv2.rectangle(output, (0, 0), (60, 60), (255, 255, 0), -1)
                    cv2.putText(output, str(int(angle)), (x2 + 30, y2), 1, 1.5, (128, 0, 250), 2)
                    cv2.putText(output, str(count), (10, 50), 1, 3.5, (128, 0, 250), 2)
                else:
                    output = frame.copy()

            except:
                output = frame.copy()
        else:
            output = frame.copy()

        cv2.imshow("output", output)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'ESC' para salir
            break

cap.release()
cv2.destroyAllWindows()
