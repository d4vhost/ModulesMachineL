import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageOps
import cv2
import mediapipe as mp
import threading
import os
import time
import queue

# --- Estilos (Mismos que Módulo 9) ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
COLOR_SUCCESS = "#34c759"
COLOR_ERROR = "#ff3b30"

FONT_TITLE = ("SF Pro Display", 18, "bold")
FONT_BODY = ("SF Pro Text", 11)
FONT_STATUS = ("SF Pro Text", 14, "bold")
FONT_BIG_STATUS = ("SF Pro Display", 22, "bold")
FONT_SMALL = ("SF Pro Text", 9)

# --- Configuración del Módulo 4 ---
IMAGE_FOLDER = "images/Modulo_4"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg')
DEFAULT_IMAGE_KEY = "NINGUNO"

# --- Cargar Imágenes de Referencia ---
os.makedirs(IMAGE_FOLDER, exist_ok=True)
GESTURE_IMAGES = {}
print(f"Escaneando carpeta '{IMAGE_FOLDER}' en busca de imágenes de gestos...")
try:
    # Nombres de archivo esperados que coinciden con las claves de gestos
    expected_files = {
        "PUÑO": "puño.jpg",
        "PALMA": "palma.jpg",
        "TIJERA": "tijera.jpg",
        "DEDO": "dedo.jpg",
        "NINGUNO": "ninguno.jpg"
    }
    
    # Comprobar si existen los archivos esperados
    for key, filename in expected_files.items():
        filepath = os.path.join(IMAGE_FOLDER, filename)
        if os.path.exists(filepath):
            GESTURE_IMAGES[key] = filepath
            print(f"  + Encontrada: '{filename}' (Clave: '{key}')")
        else:
            print(f"  - ADVERTENCIA: No se encontró '{filename}' para el gesto '{key}'.")
            GESTURE_IMAGES[key] = None # Marcar como faltante

    # Crear placeholder si 'ninguno.jpg' falta
    if GESTURE_IMAGES.get("NINGUNO") is None:
        placeholder_path = os.path.join(IMAGE_FOLDER, "ninguno.jpg")
        try:
            img = Image.new('RGB', (200, 200), color='lightgrey')
            draw = ImageDraw.Draw(img)
            draw.text((50, 90), "Imagen no encontrada", fill='black')
            img.save(placeholder_path)
            GESTURE_IMAGES["NINGUNO"] = placeholder_path
            print(f"  * Creado placeholder en '{placeholder_path}'")
        except Exception as e:
            print(f"No se pudo crear el placeholder: {e}")
            # Asignar None si falla la creación
            GESTURE_IMAGES["NINGUNO"] = None

    # Asignar placeholder a gestos sin imagen
    default_path = GESTURE_IMAGES["NINGUNO"]
    if default_path: # Solo si el placeholder existe o se creó
        for key in expected_files:
            if GESTURE_IMAGES.get(key) is None:
                GESTURE_IMAGES[key] = default_path
                print(f"  ! Usando placeholder para el gesto '{key}'.")
    else:
        print("ERROR: No se pudo cargar ni crear una imagen por defecto. La visualización puede fallar.")


except Exception as e:
    print(f"Error al escanear la carpeta {IMAGE_FOLDER}: {e}")

print(f"Carga de imágenes de gestos completa. {len(GESTURE_IMAGES)} gestos configurados.")


class GestureController:
    def __init__(self, app_queue):
        self.app_queue = app_queue
        self.cap = None
        self.running = False
        self.mp_hands = mp.solutions.hands
        # Aumentamos la confianza de detección para evitar falsos positivos
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, 
                                          min_tracking_confidence=0.7, 
                                          max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.last_gesture = DEFAULT_IMAGE_KEY
        self.last_gesture_time = time.time()
        self.debounce_time = 0.5 # Tiempo (seg) para estabilizar el gesto

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.app_queue.put(("error", "camera_error"))
            return
        self.running = True
        self.thread = threading.Thread(target=self.detect_gestures_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1.0)
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

    def classify_gesture(self, hand_landmarks):
        """
        Analiza los 21 puntos de la mano para clasificar el gesto.
        Esta es la lógica principal de detección.
        """
        try:
            landmarks = hand_landmarks.landmark
            
            # --- Lógica de Puntas de Dedos ---
            # Compara la punta (TIP) con la articulación media (PIP)
            # True si el dedo está extendido, False si está doblado
            thumb_extended = landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x > landmarks[self.mp_hands.HandLandmark.THUMB_IP].x
            index_extended = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            middle_extended = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            ring_extended = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
            pinky_extended = landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y < landmarks[self.mp_hands.HandLandmark.PINKY_PIP].y
            
            # --- Clasificación ---
            
            # Gesto: PUÑO (Todos los dedos doblados)
            if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                return "PUÑO"
                
            # Gesto: PALMA (Todos los dedos extendidos)
            if index_extended and middle_extended and ring_extended and pinky_extended: # Ignoramos el pulgar para más facilidad
                return "PALMA"
                
            # Gesto: DEDO (Solo índice extendido)
            if index_extended and not middle_extended and not ring_extended and not pinky_extended:
                return "DEDO"

            # Gesto: TIJERA (Índice y medio extendidos)
            if index_extended and middle_extended and not ring_extended and not pinky_extended:
                return "TIJERA"

            # Si no coincide con nada, es 'NINGUNO'
            return DEFAULT_IMAGE_KEY
            
        except Exception as e:
            print(f"Error en clasificación: {e}")
            return DEFAULT_IMAGE_KEY


    def detect_gestures_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.app_queue.put(("error", "frame_error"))
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            gesture_name = DEFAULT_IMAGE_KEY
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibuja los puntos de la mano en el frame
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Clasifica el gesto
                    gesture_name = self.classify_gesture(hand_landmarks)
            
            # --- Lógica de Debounce (Estabilización) ---
            current_time = time.time()
            if gesture_name != self.last_gesture:
                # Si el gesto es diferente, reinicia el timer
                self.last_gesture_time = current_time
                self.last_gesture = gesture_name
            
            # Solo envía la actualización si el gesto se ha mantenido
            # por el tiempo de self.debounce_time
            if (current_time - self.last_gesture_time > self.debounce_time):
                # Envia el gesto estabilizado a la GUI
                self.app_queue.put(("gesture", gesture_name))

            # Envia el frame de video a la GUI
            self.app_queue.put(("video_frame", frame))
            
            time.sleep(0.01) # Pequeña pausa


class SignRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Módulo 4: Reconocimiento de Señas")
        self.geometry("900x650") # Ancho aumentado para dos paneles
        self.config(bg=COLOR_BG)
        self.resizable(False, False)

        self.current_gesture = DEFAULT_IMAGE_KEY
        
        self.app_queue = queue.Queue()
        self.gesture_control = GestureController(self.app_queue)
        
        # --- Estilos ---
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_CARD, borderwidth=0)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat", borderwidth=0)
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Status.TLabel", font=FONT_STATUS, background=COLOR_BG)
        self.style.configure("BigStatus.Card.TLabel", font=FONT_BIG_STATUS, background=COLOR_CARD, foreground=COLOR_SUCCESS)
        
        # --- Layout Principal ---
        self.main_frame = ttk.Frame(self, padding=10)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        ttk.Label(self.main_frame, text="Módulo 4: Reconocimiento de Señas", style="Title.TLabel").pack(pady=(10, 10))

        # --- Panel de Contenido (Cámara + Resultado) ---
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # --- Panel de Cámara (Izquierda) ---
        self.camera_container = ttk.Frame(self.content_frame, style="Card.TFrame", width=540, height=405) # Relación 4:3
        self.camera_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 10))
        self.camera_container.pack_propagate(False)
        
        self.video_label = tk.Label(self.camera_container, text="Iniciando cámara...", bg=COLOR_CARD, fg=COLOR_FG, font=FONT_BODY)
        self.video_label.pack(expand=True)
        
        # --- Panel de Resultados (Derecha) ---
        self.result_container = ttk.Frame(self.content_frame, style="Card.TFrame", width=280)
        self.result_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.result_container.pack_propagate(False)
        
        ttk.Label(self.result_container, text="Gesto Detectado:", style="Card.TLabel", font=FONT_TITLE).pack(pady=20, padx=20)
        
        # Cargar la imagen por defecto
        self.img_tk_result = self.load_display_image(GESTURE_IMAGES[DEFAULT_IMAGE_KEY], size=(200, 200))
        
        self.image_display_label = tk.Label(self.result_container, image=self.img_tk_result, bg=COLOR_CARD)
        self.image_display_label.pack(pady=10)
        
        self.result_text_label = ttk.Label(self.result_container, 
                                            text=DEFAULT_IMAGE_KEY, 
                                            style="BigStatus.Card.TLabel")
        self.result_text_label.pack(pady=20, padx=20)

        # --- Barra de Estado (Abajo) ---
        self.status_frame = ttk.Frame(self, padding=10)
        self.status_frame.pack(fill="x", pady=10, padx=20, side="bottom")

        self.status_label = ttk.Label(self.status_frame, text="Estado: Iniciando...", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)
        
        # --- Iniciar procesos ---
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.gesture_control.start()
        self.process_queue()

    def load_display_image(self, filepath, size=(200, 150), radius=15):
        """Carga y redondea una imagen para mostrarla en la GUI."""
        if filepath is None or not os.path.exists(filepath):
            # Si el archivo no existe (ej. placeholder falló), crea una imagen de error
            img = Image.new('RGB', size, color=COLOR_ERROR)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Error\nImagen\nFaltante", fill='white', font=FONT_BODY)
        else:
            try:
                img = Image.open(filepath)
            except Exception as e:
                print(f"Error abriendo imagen {filepath}: {e}")
                img = Image.new('RGB', size, color=COLOR_ERROR)

        # Redondear esquinas
        try:
            img_fit = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
            mask = Image.new('L', size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle((0, 0) + size, radius, fill=255)
            img_fit.putalpha(mask)
            return ImageTk.PhotoImage(img_fit)
        except Exception as e:
            print(f"Error procesando imagen {filepath}: {e}")
            return ImageTk.PhotoImage(Image.new('RGB', size, color=COLOR_ERROR))

    def process_video_frame(self, img_pil, size=(540, 405), radius=15):
        """Procesa un objeto PIL Image para mostrarlo como frame de video."""
        try:
            img_fit = ImageOps.fit(img_pil, size, Image.Resampling.LANCZOS)
            mask = Image.new('L', size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle((0, 0) + size, radius, fill=255)
            img_fit.putalpha(mask)
            return ImageTk.PhotoImage(img_fit)
        except Exception as e:
            print(f"Error procesando frame de video: {e}")
            return ImageTk.PhotoImage(Image.new('RGB', size, color=COLOR_ERROR))

    def process_queue(self):
        """Procesa mensajes de la cámara y el controlador de gestos."""
        try:
            while True:
                msg = self.app_queue.get_nowait()
                
                if isinstance(msg, tuple):
                    msg_type = msg[0]
                    
                    if msg_type == "video_frame":
                        frame = msg[1]
                        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        self.imgtk_video = self.process_video_frame(img_pil, (540, 405), radius=15)
                        self.video_label.config(image=self.imgtk_video)
                    
                    elif msg_type == "gesture":
                        gesture_name = msg[1]
                        # Solo actualiza si el gesto ha cambiado
                        if gesture_name != self.current_gesture:
                            self.current_gesture = gesture_name
                            self.update_gesture_display(gesture_name)
                            
                    elif msg_type == "error":
                        error_type = msg[1]
                        if error_type == "camera_error":
                            self.video_label.config(text="ERROR: No se pudo abrir la cámara.", 
                                                    font=FONT_TITLE, fg=COLOR_ERROR)
                            self.update_status("Error de cámara", COLOR_ERROR)
                        elif error_type == "frame_error":
                            self.update_status("Error al leer frame de cámara", COLOR_ERROR)

        except queue.Empty:
            pass 
        
        self.after(16, self.process_queue) # Se ejecuta ~60 veces por segundo

    def update_gesture_display(self, gesture_name):
        """Actualiza el panel de resultados con el nuevo gesto."""
        
        # Actualizar texto de estado principal
        if gesture_name == DEFAULT_IMAGE_KEY:
            self.update_status("Mostrando mano...", COLOR_FG)
        else:
            self.update_status(f"¡Has hecho {gesture_name}!", COLOR_SUCCESS)
            
        # Actualizar el panel de resultados
        self.result_text_label.config(text=gesture_name)
        
        # Cargar y mostrar la imagen de referencia
        filepath = GESTURE_IMAGES.get(gesture_name, GESTURE_IMAGES[DEFAULT_IMAGE_KEY])
        self.img_tk_result = self.load_display_image(filepath, size=(200, 200))
        self.image_display_label.config(image=self.img_tk_result)

    def update_status(self, text, color="white"):
        """Actualiza la barra de estado inferior."""
        self.status_label.config(text=f"Estado: {text}", foreground=color)

    def on_closing(self):
        """Limpia los recursos al cerrar la app."""
        print("Cerrando la aplicación...")
        self.gesture_control.stop()
        self.destroy()

if __name__ == "__main__":
    app = SignRecognitionApp()
    app.mainloop()