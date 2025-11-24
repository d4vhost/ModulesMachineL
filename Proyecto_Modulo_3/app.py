import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import face_recognition
import os
import numpy as np
import threading
import time

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

# Tipografías ajustadas
FONT_HEADER = ("Segoe UI", 16, "bold") 
FONT_LABEL = ("Segoe UI", 10)
FONT_ENTRY = ("Segoe UI", 11)
FONT_BTN = ("Segoe UI", 10, "bold")
FONT_STATUS = ("Segoe UI", 10, "bold")

class FaceRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Sistema de Control Biométrico")

        window_width = 780
        window_height = 680 
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))
        
        self.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
        self.configure(bg=COLOR_BG_MAIN)
        self.resizable(False, False)
        
        # Variables 
        self.known_face_encodings = []
        self.known_face_names = []
        self.current_frame = None
        self.running = True
        self.is_registering = False 

        # Variables Video
        self.process_this_frame = True
        self.last_face_locations = []
        self.last_face_names = []
        
        # Rutas
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.RutaRostros = os.path.join(BASE_DIR, "data", "rostros_registrados")
        os.makedirs(self.RutaRostros, exist_ok=True)

        self.setup_styles()
        self.create_widgets()
        self.load_known_faces_initial()

        # Cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se detecta la cámara.")
            self.destroy()
            return
            
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        
        self.style.configure(".", background=COLOR_BG_MAIN, foreground=COLOR_TEXT_BODY, font=FONT_LABEL)
        self.style.configure("Header.TFrame", background=COLOR_HEADER)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        
        self.style.configure("Header.TLabel", background=COLOR_HEADER, foreground=COLOR_TEXT_HEAD, font=FONT_HEADER)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT_BODY, font=FONT_LABEL)
        self.style.configure("Status.TLabel", background=COLOR_BG_MAIN, font=FONT_STATUS)
        
        self.style.configure("TEntry", fieldbackground="#F8F9F9", borderwidth=1, relief="solid")
        self.style.map("TEntry", bordercolor=[("focus", COLOR_ACCENT), ("!focus", "#BDC3C7")])
        
        self.style.configure("TButton", font=FONT_BTN, background=COLOR_ACCENT, foreground="#FFFFFF", borderwidth=0)
        self.style.map("TButton", background=[("active", COLOR_ACCENT_HOVER)])

    def create_widgets(self):
        header_frame = ttk.Frame(self, style="Header.TFrame", height=50)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        ttk.Label(header_frame, text="REGISTRO BIOMÉTRICO", style="Header.TLabel").pack(side=tk.LEFT, padx=20)

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
        
        controls_card.columnconfigure(1, weight=1)

        ttk.Label(controls_card, text="Nombre:", style="Card.TLabel").grid(row=0, column=0, padx=(0, 10), sticky="w")
        
        self.name_entry = ttk.Entry(controls_card, font=FONT_ENTRY, width=30)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=5, ipady=3)
        
        self.register_button = ttk.Button(controls_card, text="REGISTRAR", cursor="hand2", command=self.register_face)
        self.register_button.grid(row=0, column=2, padx=(10, 0), ipadx=10, ipady=5)
        
        self.status_label = ttk.Label(main_content, text="Sistema listo.", style="Status.TLabel", anchor="center", foreground="#7F8C8D")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

    def video_loop(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue
                
                if self.is_registering:
                     time.sleep(0.02)
                     continue

                self.current_frame = frame.copy() 
                frame = cv2.flip(frame, 1)
                
                if self.process_this_frame:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    self.last_face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, self.last_face_locations)
                    
                    self.last_face_names = []
                    current_db_encodings = list(self.known_face_encodings)
                    current_db_names = list(self.known_face_names)

                    for face_encoding in face_encodings:
                        name = "Desconocido"
                        if current_db_encodings:
                            matches = face_recognition.compare_faces(current_db_encodings, face_encoding, tolerance=0.5)
                            face_distances = face_recognition.face_distance(current_db_encodings, face_encoding)
                            
                            if len(face_distances) > 0:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    name = current_db_names[best_match_index]
                        self.last_face_names.append(name)
                
                self.process_this_frame = not self.process_this_frame

                for (top, right, bottom, left), name in zip(self.last_face_locations, self.last_face_names):
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    
                    color = (0, 0, 255) if name == "Desconocido" else (0, 255, 0) 
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame, name.upper(), (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 480))
                img_pil = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                if hasattr(self, 'video_label'):
                    self.video_label.configure(image=img_tk)
                    self.video_label.image = img_tk
                
                time.sleep(0.015)

        except Exception:
            pass

    def load_known_faces_initial(self):
        def _load():
            loaded_encodings = []
            loaded_names = []
            if os.path.exists(self.RutaRostros):
                for filename in os.listdir(self.RutaRostros):
                    if filename.lower().endswith((".jpg", ".png")):
                        try:
                            img = face_recognition.load_image_file(os.path.join(self.RutaRostros, filename))
                            encs = face_recognition.face_encodings(img)
                            if encs:
                                loaded_encodings.append(encs[0])
                                loaded_names.append(os.path.splitext(filename)[0])
                        except: pass
            
            self.known_face_encodings = loaded_encodings
            self.known_face_names = loaded_names
            msg = f"Base de datos: {len(loaded_names)} usuarios." if loaded_names else "Base de datos vacía."
            self.after(0, lambda: self.update_status(msg, COLOR_TEXT_BODY))

        threading.Thread(target=_load, daemon=True).start()

    def register_face(self):
        name = self.name_entry.get().strip()
        if not name:
            self.update_status("Escriba un nombre.", COLOR_WARNING)
            return
            
        # Validación básica de nombre duplicado (Texto)
        if name.lower() in [n.lower() for n in self.known_face_names]:
            self.update_status(f"⛔ Nombre '{name}' ya existe.", COLOR_ERROR)
            return

        if self.current_frame is None or self.is_registering:
            return

        frame_copy = self.current_frame.copy()
        self.is_registering = True
        self.update_status("⏳ Verificando biometría...", COLOR_ACCENT)
        self.name_entry.config(state='disabled') 
        self.register_button.config(state='disabled')

        threading.Thread(target=self._register_worker, args=(frame_copy, name), daemon=True).start()

    def _register_worker(self, frame, name):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            
            if not boxes:
                self.after(0, lambda: self._finish_register("⚠️ No se detectó rostro.", COLOR_WARNING))
                return
            
            if len(boxes) > 1:
                self.after(0, lambda: self._finish_register("⚠️ Solo una persona a la vez.", COLOR_WARNING))
                return

            new_encoding = face_recognition.face_encodings(rgb, boxes)[0]
            
            # --- VALIDACIÓN ROSTRO DUPLICADO ---
            # Comparamos el nuevo rostro con TODOS los rostros existentes
            if len(self.known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_face_encodings, new_encoding)

                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.5:
                    existing_name = self.known_face_names[best_match_index]

                    error_msg = f"⛔ Rostro ya registrado como: {existing_name}"
                    self.after(0, lambda: messagebox.showerror("Identidad Duplicada", 
                        f"¡Acción Bloqueada!\n\nEste rostro ya pertenece al usuario: '{existing_name}'.\n\nSi desea cambiar el nombre, primero elimine al usuario anterior desde la carpeta 'data'."))
                    self.after(0, lambda: self._finish_register(error_msg, COLOR_ERROR))
                    return

            # Si pasa la validación, guardamos
            path = os.path.join(self.RutaRostros, f"{name}.jpg")
            cv2.imwrite(path, frame)
            
            self.known_face_encodings.append(new_encoding)
            self.known_face_names.append(name)
            
            self.after(0, lambda: self._finish_register(f"Usuario '{name}' registrado.", COLOR_SUCCESS, True))

        except Exception as e:
            self.after(0, lambda: self._finish_register(f"Error: {e}", COLOR_ERROR))

    def _finish_register(self, msg, color, success=False):
        self.is_registering = False
        self.update_status(msg, color)
        self.name_entry.config(state='normal')
        self.register_button.config(state='normal')
        if success:
            self.name_entry.delete(0, tk.END)

    def update_status(self, text, color):
        if hasattr(self, 'status_label'):
            self.status_label.config(text=text, foreground=color)

    def on_closing(self):
        self.running = False
        self.destroy()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()