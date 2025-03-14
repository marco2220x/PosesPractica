import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import tkinter as tk
from tkinter import Menu, Toplevel
import threading

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def start_squat_counter():
    squat_window = Toplevel()
    squat_window.title("Monitoreo de Sentadillas")
    squat_window.geometry("640x480")
    
    
    def squat_tracker():
        cap = cv2.VideoCapture(0)  # Usar la cámara en vivo
        up = False
        down = False
        count = 0

        activity_menu = Menu(menu_bar, tearoff=0)
        activity_menu.add_command(label="Lagartijas", command=lambda: print("Lagartijas seleccionadas"))
        activity_menu.add_command(label="Salto Tijera", command=lambda: print("Salto Tijera seleccionado"))
        activity_menu.add_command(label="Sentadillas", command=start_squat_counter)
        activity_menu.add_separator()
        activity_menu.add_command(label="Salir", command=squat_window.destroy)
        menu_bar.add_cascade(label="Monitorear Actividad", menu=activity_menu)
        squat_window.config(menu=menu_bar)

        with mp_pose.Pose(static_image_mode=False) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                height, width, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks is not None:
                    x1 = int(results.pose_landmarks.landmark[24].x * width)
                    y1 = int(results.pose_landmarks.landmark[24].y * height)

                    x2 = int(results.pose_landmarks.landmark[26].x * width)
                    y2 = int(results.pose_landmarks.landmark[26].y * height)

                    x3 = int(results.pose_landmarks.landmark[28].x * width)
                    y3 = int(results.pose_landmarks.landmark[28].y * height)

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
                    cv2.imshow("Monitoreo de Sentadillas", output)
                if cv2.waitKey(1) & 0xFF == 27 or not squat_window.winfo_exists():  # Salir si la ventana se cierra
                    break

        cap.release()
        cv2.destroyAllWindows()
    
    threading.Thread(target=squat_tracker, daemon=True).start()

# Crear la ventana principal
root = tk.Tk()
root.title("Aplicación con Menú")

# Crear un Frame
frame = tk.Frame(root, width=400, height=300)
frame.pack()

# Crear el menú principal
menu_bar = Menu(root)

# Crear el menú "Monitorear Actividad"
activity_menu = Menu(menu_bar, tearoff=0)
activity_menu.add_command(label="Lagartijas", command=lambda: print("Lagartijas seleccionadas"))
activity_menu.add_command(label="Salto Tijera", command=lambda: print("Salto Tijera seleccionado"))
activity_menu.add_command(label="Sentadillas", command=start_squat_counter)
activity_menu.add_separator()
activity_menu.add_command(label="Salir", command=root.quit)
menu_bar.add_cascade(label="Monitorear Actividad", menu=activity_menu)

# Configurar la ventana principal para usar el menú
root.config(menu=menu_bar)

# Iniciar el bucle principal de la aplicación
root.mainloop()
