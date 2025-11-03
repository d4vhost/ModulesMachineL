import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import face_recognition
import os
import numpy as np
import threading
import time

# --- 1. CONFIGURACIÓN DE ESTILO (Reutilizado de tu Módulo 9) ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
COLOR_SUCCESS = "#34c759"
COLOR_ERROR = "#ff3b30"

FONT_TITLE = ("SF Pro Display", 20, "bold")
FONT_BODY = ("SF Pro Text", 12)
FONT_BODY_BOLD = ("SF Pro Text", 12, "bold")
FONT_SMALL = ("SF Pro Text", 10)
### CAMBIO: Fuente más grande y negrita para el estado ###
FONT_STATUS = ("SF Pro Text", 13, "bold")

# --- 2. CLASE PRINCIPAL DE LA APLICACIÓN ---

class FaceRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # --- Configuración de la Ventana ---
        self.title("Módulo 3: Detección y Reconocimiento Facial")
        self.geometry("800x720")
        self.configure(bg=COLOR_BG)
        self.resizable(False, False)
        
        # --- Variables de Estado ---
        self.known_face_encodings = []
        self.known_face_names = []
        self.current_frame = None
        self.running = True

        ### CAMBIO: Variables para optimización ###
        self.process_this_frame = True
        self.last_face_locations = []
        self.last_face_names = []
        self.last_face_landmarks_list = []
        
        # --- Ruta de datos ---
        # ==================================================================
        # MODIFICACIÓN CLAVE: Usar rutas absolutas basadas en la ubicación del script
        # Obtenemos la ruta absoluta de la carpeta donde está app.py
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # Creamos la ruta a nuestra carpeta de datos
        self.RutaRostros = os.path.join(BASE_DIR, "data", "rostros_registrados")
        
        os.makedirs(self.RutaRostros, exist_ok=True)
        # ==================================================================
        
        # --- Estilos TTK (Minimalista Apple) ---
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_CARD, borderwidth=0, lightcolor=COLOR_CARD, darkcolor=COLOR_CARD)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        ### CAMBIO: Usar el nuevo FONT_STATUS ###
        self.style.configure("Status.TLabel", font=FONT_STATUS, background=COLOR_BG)
        
        self.style.configure("TEntry",
                             font=FONT_BODY,
                             fieldbackground=COLOR_CARD,
                             foreground=COLOR_FG,
                             insertcolor=COLOR_FG,
                             borderwidth=1,
                             relief="flat")
        self.style.map("TEntry",
                       bordercolor=[("focus", COLOR_ACCENT), ("!focus", COLOR_CARD)],
                       highlightcolor=[("focus", COLOR_ACCENT)])
        
        self.style.configure("TButton",
                             font=FONT_BODY_BOLD,
                             background=COLOR_ACCENT,
                             foreground=COLOR_FG,
                             borderwidth=0,
                             padding=(15, 10),
                             relief="flat")
        self.style.map("TButton",
                       background=[("active", "#0056b3"), ("pressed", "#0056b3")],
                       foreground=[("active", COLOR_FG), ("pressed", COLOR_FG)])

        # --- Creación de Widgets ---
        self.create_widgets()
        
        # --- Cargar datos ---
        self.load_known_faces()

        # --- Iniciar Cámara ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error de Cámara", "No se pudo abrir la cámara. La aplicación se cerrará.")
            self.destroy()
            return
            
        # --- Iniciar Bucle de Video ---
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        # --- Manejar Cierre ---
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(main_frame, text="Detección Facial en Vivo", style="Title.TLabel").pack(pady=(0, 20))
        
        self.camera_card = ttk.Frame(main_frame, style="Card.TFrame", width=640, height=480)
        self.camera_card.pack(pady=10)
        self.camera_card.pack_propagate(False)

        self.video_label = tk.Label(self.camera_card, bg=COLOR_CARD)
        self.video_label.place(x=0, y=0, width=640, height=480)
        
        controls_frame = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        controls_frame.pack(fill=tk.X, pady=(20, 0))

        controls_frame.columnconfigure(1, weight=1)

        ttk.Label(controls_frame, text="Nombre:", style="Card.TLabel", font=FONT_BODY_BOLD).grid(row=0, column=0, padx=(0, 10), sticky="w")
        
        self.name_entry = ttk.Entry(controls_frame, font=FONT_BODY, width=40)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=10)
        
        self.register_button = ttk.Button(controls_frame, text="Registrar Rostro", command=self.register_face)
        self.register_button.grid(row=0, column=2, padx=(10, 0))
        
        ### CAMBIO: Usar el estilo Status.TLabel y anchor 'center' ###
        self.status_label = ttk.Label(main_frame, text="Iniciando...", style="Status.TLabel", anchor="center")
        self.status_label.pack(fill=tk.X, pady=(15, 0))

    def video_loop(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.update_status("Error: No se puede leer el frame", COLOR_ERROR)
                    time.sleep(1)
                    continue
                
                self.current_frame = frame.copy() 
                frame = cv2.flip(frame, 1)
                
                ### CAMBIO: Inicio de la optimización ###
                # Solo procesamos 1 de cada 2 frames
                if self.process_this_frame:
                    # Achicamos el frame para procesar más rápido
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    # --- Detección y Reconocimiento ---
                    # Guardamos los resultados en variables de la clase
                    self.last_face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, self.last_face_locations)
                    self.last_face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, self.last_face_locations)
                    
                    self.last_face_names = []
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                        name = "Desconocido"

                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index]
                        self.last_face_names.append(name)
                
                # Invertimos el flag para el próximo frame
                self.process_this_frame = not self.process_this_frame
                ### CAMBIO: Fin de la optimización ###

                # --- Dibujar en el Frame (Esto se hace en TODOS los frames) ---
                # Usamos los ÚLTIMOS resultados guardados para que el video sea fluido
                
                # 1. Dibujar Puntos Faciales
                for face_landmarks in self.last_face_landmarks_list:
                    for facial_feature in face_landmarks.keys():
                        for pt in face_landmarks[facial_feature]:
                            x = pt[0] * 4
                            y = pt[1] * 4
                            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

                # 2. Dibujar Rectángulos y Nombres
                for (top, right, bottom, left), name in zip(self.last_face_locations, self.last_face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    color = COLOR_SUCCESS if name != "Desconocido" else COLOR_ERROR
                    color_bgr = tuple(int(color[i:i+2], 16) for i in (5, 3, 1))

                    cv2.rectangle(frame, (left, top), (right, bottom), color_bgr, 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color_bgr, cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)

                # --- Convertir Frame para Tkinter ---
                frame_resized = cv2.resize(frame, (640, 480)) 
                img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                self.video_label.configure(image=img_tk)
                self.video_label.image = img_tk
                
                ### CAMBIO: Reducir el sleep para un video más fluido ###
                time.sleep(0.01) # ~100 FPS (teóricos, limitado por la cámara)

        except Exception as e:
            if self.running:
                print(f"Error en el bucle de video: {e}")

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
        print("Cargando rostros conocidos...")
        for filename in os.listdir(self.RutaRostros):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(self.RutaRostros, filename)
                name = os.path.splitext(filename)[0]
                
                try:
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        encoding = encodings[0]
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)
                        print(f" - Cargado: {name}")
                    else:
                        print(f" - Advertencia: No se detectó rostro en {filename}")
                except Exception as e:
                    print(f" - Error al cargar {filename}: {e}")
        
        print(f"Carga completa. {len(self.known_face_names)} rostros conocidos.")
        if self.known_face_names:
            self.update_status(f"{len(self.known_face_names)} rostros cargados.", COLOR_SUCCESS)
        else:
            self.update_status("No hay rostros registrados. Usa el botón para añadir.", COLOR_ACCENT)

    def register_face(self):
        name = self.name_entry.get()
        if not name:
            self.update_status("Error: Debes ingresar un nombre.", COLOR_ERROR)
            return
            
        if self.current_frame is None:
            self.update_status("Error: Espera a que la cámara inicie.", COLOR_ERROR)
            return

        frame_to_register = self.current_frame.copy()
        face_locations = face_recognition.face_locations(frame_to_register, model="hog")
        
        if len(face_locations) == 0:
            self.update_status("Error: No se detecta ningún rostro.", COLOR_ERROR)
            return
        
        if len(face_locations) > 1:
            self.update_status("Error: Hay demasiados rostros. Asegúrate que solo haya uno.", COLOR_ERROR)
            return

        filename = f"{name}.jpg"
        save_path = os.path.join(self.RutaRostros, filename)
        
        try:
            cv2.imwrite(save_path, frame_to_register)
            self.update_status(f"¡Éxito! Rostro de '{name}' guardado.", COLOR_SUCCESS)
            self.name_entry.delete(0, tk.END)
            
            threading.Thread(target=self.load_known_faces, daemon=True).start()
            
        except Exception as e:
            self.update_status(f"Error al guardar imagen: {e}", COLOR_ERROR)

    def update_status(self, text, color_hex):
        if hasattr(self, 'status_label'):
            self.status_label.config(text=text, foreground=color_hex)
        else:
            print(f"Estado (pre-GUI): {text}")


    def on_closing(self):
        print("Cerrando aplicación...")
        self.running = False
        if hasattr(self, 'video_thread'):
            self.video_thread.join(timeout=1.0)
        if hasattr(self, 'cap'):
            self.cap.release()
        self.destroy()

# (Función add_rounded_corners omitida por brevedad, no se usa)

# --- 4. EJECUTAR LA APLICACIÓN ---

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()