import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw
import cv2
import mediapipe as mp
import threading
import os
import time
import queue
import numpy as np

# --- 1. CONFIGURACIÓN DE ESTILO ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
COLOR_SUCCESS = "#34c759"
COLOR_PLACEHOLDER = "#555"
COLOR_ERROR = "#ff3b30"

FONT_TITLE = ("SF Pro Display", 22, "bold")
FONT_BODY = ("SF Pro Text", 12)
FONT_BODY_BOLD = ("SF Pro Text", 12, "bold")
FONT_STATUS = ("SF Pro Text", 14, "bold")
FONT_PHONE_TITLE = ("SF Pro Text", 10, "bold")

# --- 2. RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR = os.path.join(BASE_DIR, "data", "gallery_images")
os.makedirs(GALLERY_DIR, exist_ok=True)

# --- 3. CLASE DE GESTOS (Sin cambios) ---
class GestureController:
    def __init__(self, app_queue):
        self.app_queue = app_queue
        self.cap = None
        self.running = False
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, 
                                          min_tracking_confidence=0.7, 
                                          max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.last_gesture = "NINGUNO"
        self.last_gesture_time = time.time()
        self.debounce_time = 0.3

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

    def classify_gesture(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        index_extended = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_extended = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_extended = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_extended = landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y < landmarks[self.mp_hands.HandLandmark.PINKY_PIP].y
        
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "PUÑO"
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return "PALMA"
        return "NINGUNO"

    def detect_gestures_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            gesture_name = "NINGUNO"
            hand_center_x = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    gesture_name = self.classify_gesture(hand_landmarks)
                    hand_center_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
            
            current_time = time.time()
            if gesture_name != self.last_gesture:
                if (current_time - self.last_gesture_time > self.debounce_time):
                    self.last_gesture = gesture_name
                    self.last_gesture_time = current_time
                    self.app_queue.put(("gesture", gesture_name)) # Ya no necesitamos la X

            self.app_queue.put(("video_frame", frame))
            time.sleep(0.01)

# --- 4. CLASE PRINCIPAL DE LA APP (RE-DISEÑADA) ---
class DualPhoneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # Ocultar la ventana raíz principal, solo usaremos Toplevels
        self.withdraw()
        
        # --- Variables de estado ---
        self.image_grabbed = False
        self.selected_pil_image = None # La imagen PIL original seleccionada
        self.selected_pil_image_rounded = None # La versión redondeada para mostrar
        self.transferred_image_tk = None # Para mantener la referencia en el teléfono 2
        self.wobble_phase = 0
        self.gallery_image_cache = {} # Cache para miniaturas

        # --- Crear las dos ventanas de "teléfono" ---
        self.phone1_window = self.create_phone_window("Teléfono 1: Galería", "400x700+50+50")
        self.phone2_window = self.create_phone_window("Teléfono 2: Receptor", "400x700+500+50")
        
        self.app_queue = queue.Queue()
        self.gesture_control = GestureController(self.app_queue)

        self.setup_styles()
        self.setup_phone1_ui() # UI de Galería y Vista Principal
        self.setup_phone2_ui() # UI del Receptor
        
        self.load_gallery_images()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.gesture_control.start()
        self.process_queue()

    def create_phone_window(self, title, geometry):
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry(geometry)
        window.configure(bg=COLOR_BG)
        window.resizable(False, False)
        
        # Simular "notch" o barra de estado
        status_bar = tk.Frame(window, bg=COLOR_BG, height=30)
        status_bar.pack(fill="x")
        ttk.Label(status_bar, text=title, font=FONT_PHONE_TITLE, background=COLOR_BG).pack(pady=5)
        
        window.protocol("WM_DELETE_WINDOW", self.on_closing)
        return window

    def setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_CARD, borderwidth=0)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Status.TLabel", font=FONT_STATUS, background=COLOR_BG, foreground=COLOR_ACCENT)
        self.style.configure("Gallery.TButton", background=COLOR_CARD, borderwidth=0, relief="flat")
        self.style.map("Gallery.TButton", background=[("active", COLOR_BG), ("pressed", COLOR_BG)])

    def setup_phone1_ui(self):
        # --- 1. VISTA DE GALERÍA (Apple-style) ---
        self.gallery_frame = ttk.Frame(self.phone1_window, padding=10)
        self.gallery_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(self.gallery_frame, text="Mi Galería", style="Title.TLabel", font=("SF Pro Display", 20, "bold")).pack(pady=10)
        
        canvas = tk.Canvas(self.gallery_frame, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.gallery_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, style="TFrame")

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- 2. VISTA PRINCIPAL (Imagen seleccionada) ---
        self.main_view_frame = ttk.Frame(self.phone1_window, padding=10)
        # Oculto al inicio
        
        back_button = ttk.Button(self.main_view_frame, text="< Galería", command=self.go_back_to_gallery)
        back_button.pack(anchor="nw", pady=(0, 10))

        # Contenedor para la cámara y la imagen (tamaño de teléfono)
        self.display_container = tk.Frame(self.main_view_frame, bg=COLOR_BG, width=380, height=580)
        self.display_container.pack(fill="both", expand=True, pady=5)
        self.display_container.pack_propagate(False)

        self.video_label = tk.Label(self.display_container, bg=COLOR_BG)
        self.video_label.place(x=0, y=0, width=380, height=580) # Llenar el contenedor

        self.image_label = tk.Label(self.display_container, bg=COLOR_BG)
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        self.status_label = ttk.Label(self.main_view_frame, text="Usa ✊ para 'agarrar' la imagen", style="Status.TLabel", anchor="center")
        self.status_label.pack(fill="x", side="bottom", pady=5)

    def setup_phone2_ui(self):
        # --- Teléfono 2: Solo un receptor ---
        self.receiver_frame = ttk.Frame(self.phone2_window, style="Card.TFrame", padding=10)
        self.receiver_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.receiver_image_label = tk.Label(self.receiver_frame, bg=COLOR_CARD)
        self.receiver_image_label.pack(expand=True, anchor="center")
        
        self.receiver_status_label = ttk.Label(self.receiver_frame, text="Esperando transferencia...",
                                                 font=FONT_BODY, background=COLOR_CARD, foreground=COLOR_PLACEHOLDER)
        self.receiver_status_label.place(relx=0.5, rely=0.5, anchor="center")

    def load_gallery_images(self):
        row, col = 0, 0
        img_size = 100 # Miniaturas más pequeñas para el teléfono
        
        for filename in os.listdir(GALLERY_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(GALLERY_DIR, filename)
                
                try:
                    img_pil = Image.open(filepath)
                    img_pil.thumbnail((img_size, img_size))
                    img_pil_rounded = self.add_rounded_corners(img_pil, 10)
                    img_tk = ImageTk.PhotoImage(img_pil_rounded)
                    
                    self.gallery_image_cache[filepath] = img_tk # Guardar en cache
                    
                    btn = ttk.Button(self.scrollable_frame, image=img_tk, style="Gallery.TButton",
                                     command=lambda p=filepath: self.select_image(p))
                    btn.image = img_tk # Mantener referencia
                    btn.grid(row=row, column=col, padx=5, pady=5)
                    
                    col += 1
                    if col > 2: # 3 columnas
                        col = 0
                        row += 1
                except Exception as e:
                    print(f"Error cargando imagen {filename}: {e}")
    
    def add_rounded_corners(self, img_pil, radius):
        mask = Image.new('L', img_pil.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0) + img_pil.size, radius, fill=255)
        output = ImageOps.fit(img_pil, img_pil.size, centering=(0.5, 0.5))
        output.putalpha(mask)
        final_img = Image.new("RGBA", img_pil.size, COLOR_CARD)
        final_img.paste(output, (0, 0), output)
        return final_img.convert("RGB") # Convertir a RGB

    def select_image(self, image_path):
        self.gallery_frame.pack_forget()
        self.main_view_frame.pack(fill=tk.BOTH, expand=True)
        
        self.selected_pil_image = Image.open(image_path) # Original
        self.selected_pil_image.thumbnail((360, 560)) # Ajustar a la vista principal
        
        self.selected_pil_image_rounded = self.add_rounded_corners(self.selected_pil_image, 20)
        
        img_tk = ImageTk.PhotoImage(self.selected_pil_image_rounded)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def go_back_to_gallery(self):
        self.main_view_frame.pack_forget()
        self.gallery_frame.pack(fill=tk.BOTH, expand=True)
        self.selected_pil_image = None
        self.selected_pil_image_rounded = None
        self.image_grabbed = False
        self.image_label.config(image=None)
        self.image_label.image = None
        self.status_label.config(text="Usa ✊ para 'agarrar' la imagen")
        
        # Limpiar el teléfono 2
        self.receiver_image_label.config(image=None)
        self.receiver_status_label.place(relx=0.5, rely=0.5, anchor="center")


    def process_queue(self):
        try:
            while True:
                msg = self.app_queue.get_nowait()
                
                if msg[0] == "video_frame":
                    frame = msg[1]
                    frame_resized = cv2.resize(frame, (380, 580)) # Ajustar al contenedor del teléfono 1
                    img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    self.video_label.configure(image=img_tk)
                    self.video_label.image = img_tk
                
                elif msg[0] == "gesture":
                    if self.selected_pil_image: # Solo si hay una imagen seleccionada
                        self.handle_gesture_on_image(msg[1])
                
                elif msg[0] == "error":
                    self.video_label.config(text=f"Error: {msg[1]}", font=FONT_TITLE, fg=COLOR_ERROR)
        
        except queue.Empty:
            pass
        
        if self.image_grabbed and self.selected_pil_image_rounded:
            self.apply_wobble_effect()
        
        self.after(16, self.process_queue) # ~60 FPS

    def handle_gesture_on_image(self, gesture):
        if gesture == "PUÑO":
            if not self.image_grabbed:
                print("¡Imagen Agarrada!")
                self.image_grabbed = True
                self.status_label.config(text="¡Agarrado! Usa 🖐️ para 'soltar'")
        
        elif gesture == "PALMA":
            if self.image_grabbed:
                print("¡Imagen Soltada! Transfiriendo...")
                self.image_grabbed = False
                
                # 1. Ocultar imagen del Teléfono 1
                self.image_label.config(image=None)
                self.image_label.image = None
                self.status_label.config(text="¡Transferido! Vuelve a la galería.")
                
                # 2. Mostrar imagen en Teléfono 2
                self.receiver_status_label.place_forget() # Ocultar texto "Esperando"
                
                # Usamos la imagen original (sin redondear) para la transferencia
                self.transferred_image_tk = ImageTk.PhotoImage(self.selected_pil_image)
                self.receiver_image_label.config(image=self.transferred_image_tk)
                self.receiver_image_label.image = self.transferred_image_tk

    def apply_wobble_effect(self):
        """El efecto 'Gelatinoso' (Bamboleo)."""
        if not self.selected_pil_image_rounded:
            return

        try:
            # ==================================================================
            # --- ARREGLO DEL ERROR ---
            # Convertir PIL a array de OpenCV
            # Usamos la imagen ya redondeada como base
            img_cv_rgba = np.array(self.selected_pil_image_rounded.convert('RGBA'))
            
            # Separar el canal alfa (bordes redondeados)
            b, g, r, a = cv2.split(img_cv_rgba)
            img_cv = cv2.merge((b, g, r)) # Imagen BGR
            alpha = cv2.merge((a, a, a)) # Canal alfa
            
            # --- FIN ARREGLO ---
            
            rows, cols = img_cv.shape[:2]
            map_x = np.zeros((rows, cols), dtype=np.float32)
            map_y = np.zeros((rows, cols), dtype=np.float32)
            
            self.wobble_phase += 0.5 # Velocidad de la onda
            
            for i in range(rows):
                for j in range(cols):
                    offset_x = int(10.0 * np.sin(2.0 * np.pi * i / 150 + self.wobble_phase))
                    offset_y = 0 
                    
                    map_x[i, j] = j + offset_x
                    map_y[i, j] = i + offset_y

            # Aplicar la distorsión
            distorted_img = cv2.remap(img_cv, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            
            # --- ARREGLO DEL ERROR 2 ---
            # Aplicar la máscara de bordes redondeados DE VUELTA
            # Esto evita que el fondo negro se vea
            distorted_img = np.where(alpha > 0, distorted_img, img_cv)
            # --- FIN ARREGLO 2 ---
            
            # Convertir de vuelta a PIL y luego a Tkinter
            img_pil_distorted = Image.fromarray(cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img_pil_distorted)
            
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

        except Exception as e:
            print(f"Error en efecto wobble: {e}")

    def on_closing(self):
        print("Cerrando la aplicación...")
        self.gesture_control.stop()
        self.destroy()

# --- 5. EJECUTAR LA APLICACIÓN ---
if __name__ == "__main__":
    app = DualPhoneApp()
    app.mainloop()