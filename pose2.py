import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def distanceCalculate(p1, p2):
    """p1 y p2 en formato de tuplas (x1, y1) y (x2, y2)"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

jumpStart = 0
jumpCount = 0

# Webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convertimos la imagen a RGB (Mejora el procesamiento con MediaPipe)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convertimos de nuevo a BGR para OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        if results.pose_landmarks:
            # Obtener puntos clave de las caderas y rodillas
            leftHip = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width),
                       int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height))
            rightHip = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width),
                        int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height))
            leftKnee = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width),
                        int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height))
            rightKnee = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width),
                         int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height))

            # Dibujar puntos clave en la imagen
            cv2.circle(image, leftHip, 10, (0, 255, 255), -1)
            cv2.circle(image, rightHip, 10, (0, 255, 255), -1)
            cv2.circle(image, leftKnee, 10, (0, 255, 255), -1)
            cv2.circle(image, rightKnee, 10, (0, 255, 255), -1)

            # Calcular la distancia entre las caderas y las rodillas
            hipDistance = distanceCalculate(leftHip, rightHip)
            kneeDistance = distanceCalculate(leftKnee, rightKnee)

            # Detectar cuando las piernas están suficientemente abiertas
            if kneeDistance > 200:  # Umbral para considerar que las piernas están abiertas
                jumpStart = 1
            elif jumpStart and kneeDistance < 100:  # Cuando las piernas se cierran de nuevo
                jumpCount += 1
                jumpStart = 0

            print("Jump Count:", jumpCount)

            # Mostrar el contador de saltos en la imagen
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            image = cv2.putText(image, str(jumpCount), org, font, fontScale, color, thickness, cv2.LINE_AA)

            # Dibujar los landmarks completos
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Mostrar la imagen con los landmarks
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # Presionar 'Esc' para salir
            break

cap.release()
cv2.destroyAllWindows()


