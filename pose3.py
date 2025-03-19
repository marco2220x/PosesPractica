import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

def draw_text(image, text, position, font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 165, 255), border_color=(255, 255, 255)):
    """Dibuja un texto con fondo y contorno para mejor visibilidad."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    cv2.rectangle(image, (x - 6, y - text_height - 6), (x + text_width + 6, y + 6), border_color, -1)
    cv2.rectangle(image, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def count_squats():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
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

            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    required_landmarks = [24, 26, 28]
                    
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

                        angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))

                        if angle >= 160:
                            up = True
                        if up and not down and angle <= 70:
                            down = True
                        if up and down and angle >= 160:
                            count += 1
                            up = False
                            down = False

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
                        draw_text(frame, "Debes realizar el ejercicio en postura de frente", (50, 130), font_scale=0.5, text_color=(255, 255, 255), bg_color=(0, 0, 255))
                        output = frame.copy()

                except:
                    output = frame.copy()
            else:
                output = frame.copy()

            cv2.imshow("Contado sentadillas", output)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función
count_squats()
