import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import face_recognition
import os
import numpy as np
import threading
import time

# --- CONFIGURACIÓN DE COLORES ---
COLOR_BG_MAIN = "#F0F2F5"      
COLOR_HEADER = "#2C3E50"   
COLOR_CARD = "#FFFFFF"    
COLOR_TEXT_HEAD = "#FFFFFF"     
COLOR_TEXT_BODY = "#34495E"     
COLOR_ACCENT = "#2980B9"   
COLOR_ACCENT_HOVER = "#1F618D"  
COLOR_SUCCESS = "#27AE60"     
COLOR_ERROR = "#C0392B"   
COLOR_WARNING = "#F39C12"
COLOR_ANIMAL = "#8E44AD"

FONT_HEADER = ("Segoe UI", 16, "bold") 
FONT_LABEL = ("Segoe UI", 10)
FONT_BTN = ("Segoe UI", 10, "bold")
FONT_STATUS = ("Segoe UI", 10, "bold")

class FaceRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Sistema de Detección Biométrica")
        
        # --- AJUSTE DE TAMAÑO Y CENTRADO ---
        window_width = 800
        window_height = 620  # Reducido para que se centre mejor
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Cálculo matemático para el centro exacto
        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))
        
        self.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
        
        self.configure(bg=COLOR_BG_MAIN)
        self.resizable(False, False)
        
        # --- ESTADO DEL SISTEMA ---
        self.view_mode = 'live'
        self.running = True
        self.model_failed = False
        
        # --- CARGAR MODELOS ---
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Nombres de archivo corregidos (sin .txt extra)
        self.path_proto = os.path.join(BASE_DIR, "model", "MobileNetSSD_deploy.prototxt")
        self.path_model = os.path.join(BASE_DIR, "model", "MobileNetSSD_deploy.caffemodel")
        
        self.net = None
        self.CLASSES = ["fondo", "avion", "bicicleta", "pajaro", "bote",
                        "botella", "bus", "auto", "gato", "silla", "vaca", "mesa",
                        "perro", "caballo", "moto", "persona", "planta", "oveja",
                        "sofa", "tren", "tv"]
        self.load_mobilenet()

        self.setup_styles()
        self.create_widgets()

        # --- INICIAR CÁMARA ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se detecta la cámara.")
            self.destroy()
            return
            
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_mobilenet(self):
        print(f"\n--- DIAGNÓSTICO DE MODELO ---")
        try:
            if os.path.exists(self.path_proto) and os.path.exists(self.path_model):
                size_proto = os.path.getsize(self.path_proto)
                size_model = os.path.getsize(self.path_model)
                
                if size_proto > 0 and size_model > 0:
                    self.net = cv2.dnn.readNetFromCaffe(self.path_proto, self.path_model)
                    print(">>> ÉXITO: Modelo MobileNet cargado correctamente.\n")
                else:
                    print(">>> ERROR: Archivos VACÍOS (0kb).\n")
            else:
                print(">>> ERROR: Archivos no encontrados en carpeta 'model'.\n")
        except Exception as e:
            print(f">>> EXCEPCIÓN: {e}\n")
            self.net = None

    def setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG_MAIN, foreground=COLOR_TEXT_BODY, font=FONT_LABEL)
        self.style.configure("Header.TFrame", background=COLOR_HEADER)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        self.style.configure("Header.TLabel", background=COLOR_HEADER, foreground=COLOR_TEXT_HEAD, font=FONT_HEADER)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT_BODY, font=FONT_LABEL)
        self.style.configure("Status.TLabel", background=COLOR_BG_MAIN, font=FONT_STATUS)
        self.style.configure("TButton", font=FONT_BTN, background=COLOR_ACCENT, foreground="#FFFFFF", borderwidth=0)
        self.style.map("TButton", background=[("active", COLOR_ACCENT_HOVER)])

    def create_widgets(self):
        header_frame = ttk.Frame(self, style="Header.TFrame", height=50)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        ttk.Label(header_frame, text="DETECCIÓN INTELIGENTE", style="Header.TLabel").pack(side=tk.LEFT, padx=20)

        main_content = ttk.Frame(self)
        main_content.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        video_card = ttk.Frame(main_content, style="Card.TFrame", padding=4)
        video_card.pack(pady=(0, 10))
        
        self.camera_frame = tk.Frame(video_card, bg="black", width=640, height=480)
        self.camera_frame.pack()
        self.camera_frame.pack_propagate(False)

        self.video_label = tk.Label(self.camera_frame, bg="black")
        self.video_label.place(x=0, y=0, width=640, height=480)
        
        controls_card = ttk.Frame(main_content, style="Card.TFrame", padding=10)
        controls_card.pack(fill=tk.X)
        
        self.btn_action = ttk.Button(controls_card, text="📂 CARGAR IMAGEN PARA ANALIZAR", cursor="hand2", command=self.toggle_mode)
        self.btn_action.pack(fill=tk.X, ipady=6)
        
        self.status_label = ttk.Label(main_content, text="Iniciando cámara...", style="Status.TLabel", anchor="center", foreground="#7F8C8D")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

    def toggle_mode(self):
        if self.view_mode == 'live':
            self.analyze_image_file()
        else:
            self.reset_to_camera()

    def reset_to_camera(self):
        self.view_mode = 'live'
        self.btn_action.config(text="📂 CARGAR IMAGEN PARA ANALIZAR", cursor="hand2")
        self.update_status("Cámara activa.", COLOR_TEXT_BODY)

    def process_frame_for_objects(self, image):
        detected_types = []
        
        # 1. PERSONAS (Face Recognition)
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.view_mode == 'live':
                small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb_small)
                face_locations = [(t*4, r*4, b*4, l*4) for (t, r, b, l) in locs]
            else:
                face_locations = face_recognition.face_locations(rgb_image)

            if face_locations:
                detected_types.append("PERSONA")
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    label = "PERSONA"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                    cv2.rectangle(image, (left, bottom - 25), (left + w + 10, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(image, label, (left + 5, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        except: pass

        # 2. ANIMALES (MobileNet)
        if self.net and not self.model_failed:
            try:
                (h_img, w_img) = image.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    # --- AJUSTE CRÍTICO: BAJAMOS LA CONFIANZA A 0.2 (20%) ---
                    # Esto permite detectar animales en poses difíciles (de lado, lejos, etc.)
                    if confidence > 0.2:  
                        idx = int(detections[0, 0, i, 1])
                        label = self.CLASSES[idx]
                        animales = ["pajaro", "gato", "vaca", "perro", "caballo", "oveja"]
                        
                        if label in animales:
                            detected_types.append("ANIMAL")
                            box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
                            (startX, startY, endX, endY) = box.astype("int")
                            
                            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 255), 2)
                            txt = f"ANIMAL ({label.upper()})"
                            (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                            cv2.rectangle(image, (startX, startY - 25), (startX + w + 10, startY), (255, 0, 255), cv2.FILLED)
                            cv2.putText(image, txt, (startX + 5, startY - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            except:
                self.model_failed = True

        return image, detected_types

    def video_loop(self):
        process_this_frame = True
        
        while self.running:
            if self.view_mode == 'static':
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            if process_this_frame:
                display_frame, detected = self.process_frame_for_objects(display_frame)
                
                if "PERSONA" in detected:
                    self.update_status("PERSONA DETECTADA", COLOR_SUCCESS)
                elif "ANIMAL" in detected:
                    self.update_status("ANIMAL DETECTADO", COLOR_ANIMAL)
                else:
                    self.update_status("BUSCANDO...", COLOR_WARNING)
            
            try:
                img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img)
                
                if hasattr(self, 'video_label') and self.view_mode == 'live':
                    self.video_label.configure(image=img_tk)
                    self.video_label.image = img_tk
            except: pass
            
            time.sleep(0.015)

    def analyze_image_file(self):
        self.view_mode = 'static'
        
        file_path = filedialog.askopenfilename(title="Seleccionar Imagen", filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
        if not file_path:
            self.reset_to_camera()
            return

        try:
            stream = open(file_path, "rb")
            bytes_data = bytearray(stream.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
            stream.close()

            if image is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen.")
                self.reset_to_camera()
                return
            
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            processed_image, detected = self.process_frame_for_objects(image.copy())

            # Ajuste de tamaño (Fit en 640x480)
            h, w = processed_image.shape[:2]
            ratio = min(640/w, 480/h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            processed_image = cv2.resize(processed_image, (new_w, new_h))
            
            final_display = np.zeros((480, 640, 3), dtype=np.uint8)
            y_offset = (480 - new_h) // 2
            x_offset = (640 - new_w) // 2
            final_display[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = processed_image

            img_rgb = cv2.cvtColor(final_display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

            self.btn_action.config(text="🎥 VOLVER A CÁMARA", cursor="hand2")
            self.update_status("Imagen analizada.", COLOR_TEXT_BODY)

        except Exception as e:
            print(f"Error: {e}")
            self.reset_to_camera()

    def update_status(self, text, color):
        try:
            self.after(0, lambda: self.status_label.config(text=text, foreground=color))
        except: pass

    def on_closing(self):
        self.running = False
        self.destroy()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()