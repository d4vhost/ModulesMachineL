import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageOps
import cv2
import mediapipe as mp
import threading
import os
import time
import queue

COLOR_BG = "#f5f7fa"
COLOR_CARD = "#ffffff"
COLOR_CARD_SECONDARY = "#e3f2fd"
COLOR_FG = "#2c3e50"
COLOR_ACCENT = "#5c6bc0"
COLOR_ACCENT_LIGHT = "#9fa8da"
COLOR_SUCCESS = "#26a69a"
COLOR_ERROR = "#ef5350"
COLOR_WARNING = "#ffa726"
COLOR_INFO = "#42a5f5"
COLOR_BORDER = "#e0e0e0"
COLOR_SHADOW = "#d0d0d0"

FONT_TITLE = ("Segoe UI", 24, "bold")
FONT_SUBTITLE = ("Segoe UI", 16, "bold")
FONT_BODY = ("Segoe UI", 11)
FONT_STATUS = ("Segoe UI", 13, "bold")
FONT_BIG_STATUS = ("Segoe UI", 28, "bold")
FONT_SMALL = ("Segoe UI", 9)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "data", "reference_images")
DEFAULT_IMAGE_KEY = "NINGUNO"

# Cargar Imágenes
os.makedirs(IMAGE_FOLDER, exist_ok=True)
GESTURE_IMAGES = {}
print(f"Escaneando carpeta '{IMAGE_FOLDER}'...")

try:
    expected_files = {
        "PUÑO": "puño.jpg", "PALMA": "palma.jpg",
        "TIJERA": "tijera.jpg", "DEDO": "dedo.jpg",
        "NINGUNO": "ninguno.jpg"
    }
    
    for key, filename in expected_files.items():
        filepath = os.path.join(IMAGE_FOLDER, filename)
        if os.path.exists(filepath):
            GESTURE_IMAGES[key] = filepath
        else:
            GESTURE_IMAGES[key] = None

    if GESTURE_IMAGES.get("NINGUNO") is None:
        placeholder_path = os.path.join(IMAGE_FOLDER, "ninguno.jpg")
        try:
            img = Image.new('RGB', (200, 200), color='#e3f2fd')
            draw = ImageDraw.Draw(img)
            draw.text((50, 90), "Sin imagen", fill='#5c6bc0')
            img.save(placeholder_path)
            GESTURE_IMAGES["NINGUNO"] = placeholder_path
        except:
            GESTURE_IMAGES["NINGUNO"] = None

    default_path = GESTURE_IMAGES["NINGUNO"]
    if default_path:
        for key in expected_files:
            if GESTURE_IMAGES.get(key) is None:
                GESTURE_IMAGES[key] = default_path

except Exception as e:
    print(f"Error cargando imágenes: {e}")

class GestureController:
    def __init__(self, app_queue):
        self.app_queue = app_queue
        self.cap = None
        self.running = False
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.last_gesture = DEFAULT_IMAGE_KEY
        self.last_gesture_time = time.time()
        self.debounce_time = 0.5

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
        try:
            lm = hand_landmarks.landmark
            h = self.mp_hands.HandLandmark
            
            thumb = lm[h.THUMB_TIP].x > lm[h.THUMB_IP].x
            index = lm[h.INDEX_FINGER_TIP].y < lm[h.INDEX_FINGER_PIP].y
            middle = lm[h.MIDDLE_FINGER_TIP].y < lm[h.MIDDLE_FINGER_PIP].y
            ring = lm[h.RING_FINGER_TIP].y < lm[h.RING_FINGER_PIP].y
            pinky = lm[h.PINKY_TIP].y < lm[h.PINKY_PIP].y
            
            if not index and not middle and not ring and not pinky: return "PUÑO"
            if index and middle and ring and pinky: return "PALMA"
            if index and not middle and not ring and not pinky: return "DEDO"
            if index and middle and not ring and not pinky: return "TIJERA"
            return DEFAULT_IMAGE_KEY
        except:
            return DEFAULT_IMAGE_KEY

    def detect_gestures_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            gesture = DEFAULT_IMAGE_KEY
            
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hl, self.mp_hands.HAND_CONNECTIONS)
                    gesture = self.classify_gesture(hl)
            
            if gesture != self.last_gesture:
                self.last_gesture_time = time.time()
                self.last_gesture = gesture
            
            if (time.time() - self.last_gesture_time > self.debounce_time):
                self.app_queue.put(("gesture", gesture))

            self.app_queue.put(("video_frame", frame))
            time.sleep(0.01)

class SignRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reconocimiento de Señas - Módulo 4")
        
        w, h = 1100, 720
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.config(bg=COLOR_BG)
        self.resizable(False, False)

        self.current_gesture = DEFAULT_IMAGE_KEY
        self.app_queue = queue.Queue()
        self.gesture_control = GestureController(self.app_queue)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=COLOR_BG, foreground=COLOR_FG)
        style.configure("Subtitle.TLabel", font=FONT_SUBTITLE, background=COLOR_BG, foreground=COLOR_ACCENT)

        header = tk.Frame(self, bg=COLOR_ACCENT, height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="🤚 Reconocimiento de Señas", font=("Segoe UI", 28, "bold"), bg=COLOR_ACCENT, fg="white").pack(expand=True)

        main = ttk.Frame(self, padding=20)
        main.pack(expand=True, fill=tk.BOTH)
        
        content = ttk.Frame(main)
        content.pack(expand=True, fill=tk.BOTH)

        cam_wrap = ttk.Frame(content)
        cam_wrap.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 15))
        ttk.Label(cam_wrap, text="📹 Cámara en Vivo", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 10))
        
        # El contenedor se crea y empaqueta correctamente ahora
        self.camera_container = self._create_card_with_shadow(cam_wrap, width=620, height=465)
        
        self.video_label = tk.Label(self.camera_container, text="Iniciando...", bg=COLOR_CARD, fg=COLOR_FG)
        self.video_label.pack(expand=True, fill=tk.BOTH)

        res_wrap = ttk.Frame(content)
        res_wrap.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(res_wrap, text="✨ Gesto Detectado", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 10))
        
        self.result_container = self._create_card_with_shadow(res_wrap, width=340, height=465)
        
        res_inner = tk.Frame(self.result_container, bg=COLOR_CARD)
        res_inner.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        self.img_result = self.load_img(GESTURE_IMAGES[DEFAULT_IMAGE_KEY])
        self.lbl_img = tk.Label(res_inner, image=self.img_result, bg=COLOR_CARD)
        self.lbl_img.pack(pady=20)
        
        self.lbl_text = tk.Label(res_inner, text=DEFAULT_IMAGE_KEY, font=FONT_BIG_STATUS, bg=COLOR_CARD, fg=COLOR_SUCCESS)
        self.lbl_text.pack(pady=10)
        
        self.status_ind = tk.Frame(res_inner, bg=COLOR_WARNING, height=5)
        self.status_ind.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        # Footer
        self.lbl_status = tk.Label(self, text="Estado: Iniciando...", font=FONT_STATUS, bg=COLOR_CARD_SECONDARY, fg=COLOR_FG, height=2)
        self.lbl_status.pack(fill=tk.X, side=tk.BOTTOM)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.gesture_control.start()
        self.process_queue()

    def _create_card_with_shadow(self, parent, width, height):
        # 1. Crear sombra y EMPAQUETARLA 
        shadow = tk.Frame(parent, bg=COLOR_SHADOW, width=width+4, height=height+4)
        shadow.pack(fill=tk.BOTH, expand=True) 
        shadow.pack_propagate(False)
        
        card = tk.Frame(shadow, bg=COLOR_CARD, highlightbackground=COLOR_BORDER, highlightthickness=1)
        card.place(x=0, y=0, width=width, height=height) 
        
        return card

    def load_img(self, path):
        try:
            if path and os.path.exists(path):
                img = Image.open(path)
            else:
                img = Image.new('RGB', (220, 220), color=COLOR_ERROR)
            
            img = ImageOps.fit(img, (220, 220), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except:
            return ImageTk.PhotoImage(Image.new('RGB', (220, 220), color=COLOR_ERROR))

    def process_queue(self):
        try:
            while True:
                type, data = self.app_queue.get_nowait()
                if type == "video_frame":
                    img = Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
                    img = ImageOps.fit(img, (620, 465), Image.Resampling.LANCZOS)
                    self.imgtk = ImageTk.PhotoImage(img)
                    self.video_label.config(image=self.imgtk)
                elif type == "gesture":
                    self.update_ui(data)
                elif type == "error":
                    self.lbl_status.config(text=f"Error: {data}", fg=COLOR_ERROR)
        except queue.Empty:
            pass
        self.after(15, self.process_queue)

    def update_ui(self, gesture):
        if gesture == DEFAULT_IMAGE_KEY:
            self.lbl_status.config(text="Esperando gesto...", fg=COLOR_FG)
            self.status_ind.config(bg=COLOR_WARNING)
        else:
            self.lbl_status.config(text=f"¡Detectado: {gesture}!", fg=COLOR_SUCCESS)
            self.status_ind.config(bg=COLOR_SUCCESS)
            
        self.lbl_text.config(text=gesture)
        self.img_result = self.load_img(GESTURE_IMAGES.get(gesture))
        self.lbl_img.config(image=self.img_result)

    def on_close(self):
        self.gesture_control.stop()
        self.destroy()

if __name__ == "__main__":
    app = SignRecognitionApp()
    app.mainloop()