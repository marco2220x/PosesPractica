import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calcula el ángulo entre tres puntos."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    bc = c - b

    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

jumpStart = False
jumpCount = 0

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Puntos de las piernas
            leftHip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * image_width,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * image_height)
            rightHip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height)
            leftKnee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height)
            rightKnee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height)
            leftAnkle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height)
            rightAnkle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height)

            # Puntos de los brazos
            leftShoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height)
            rightShoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height)
            leftElbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height)
            rightElbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height)
            leftWrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height)
            rightWrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height)

            # Calcular ángulos
            leftLegAngle = calculate_angle(leftHip, leftKnee, leftAnkle)
            rightLegAngle = calculate_angle(rightHip, rightKnee, rightAnkle)
            leftArmAngle = calculate_angle(leftShoulder, leftElbow, leftWrist)
            rightArmAngle = calculate_angle(rightShoulder, rightElbow, rightWrist)

            # Imprimir los ángulos en consola para depuración
            print(f"Pierna Izq: {leftLegAngle:.2f}, Pierna Der: {rightLegAngle:.2f}, "
                  f"Brazo Izq: {leftArmAngle:.2f}, Brazo Der: {rightArmAngle:.2f}")

            # Condiciones ajustadas para detectar el salto
            legs_open = leftLegAngle < 165 or rightLegAngle < 165
            arms_up = leftArmAngle > 145 or rightArmAngle > 145

            if legs_open and arms_up:
                jumpStart = True
            elif jumpStart and not legs_open and not arms_up:
                jumpCount += 1
                jumpStart = False
                print("Salto contado:", jumpCount)

            # Mostrar el contador en la pantalla
            cv2.rectangle(image, (20, 20), (200, 80), (0, 0, 0), -1)
            cv2.putText(image, f"Saltos: {jumpCount}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Dibujar los landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Jump Counter', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # Presionar 'Esc' para salir
            break

cap.release()
cv2.destroyAllWindows()

