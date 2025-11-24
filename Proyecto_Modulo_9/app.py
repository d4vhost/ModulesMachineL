import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw
import cv2
import mediapipe as mp
import threading
import time
import queue
import numpy as np
import socket
import pickle
import struct
import math
import random
from datetime import datetime

COLOR_BG = "#0a0e27"
COLOR_PANEL = "#1a1f3a"
COLOR_PANEL_LIGHT = "#252d4a"
COLOR_TEXT = "#e8eaf6"
COLOR_TEXT_DIM = "#9fa8c9"
COLOR_ACCENT = "#00d4ff"
COLOR_ACCENT_2 = "#6366f1"
COLOR_SUCCESS = "#10b981"
COLOR_CHECKER_1 = "#2a2f4a"
COLOR_CHECKER_2 = "#1e2338"

FONT_TITLE = ("SF Pro Display", 18, "bold")
FONT_SUBTITLE = ("SF Pro Display", 12)
FONT_BODY = ("SF Pro Text", 11)
FONT_STATUS = ("SF Pro Text", 10, "bold")
FONT_HAMBURGER = ("SF Pro Display", 24)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR = os.path.join(BASE_DIR, "data", "gallery_images")
RECEIVED_DIR = os.path.join(BASE_DIR, "data", "gallery_images_receptor")
ICON_PATH = os.path.join(BASE_DIR, "data", "icon", "estacion.png")

os.makedirs(GALLERY_DIR, exist_ok=True)
os.makedirs(RECEIVED_DIR, exist_ok=True)

# IP de Receptor
# Radmin VPN IP del destino:
RECEIVER_IP = '26.18.184.16' 
PORT = 9999
MAX_IMG_SIZE = (500, 500)

def center_window(root, width, height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f'{width}x{height}+{x}+{y}')

def create_checkered_bg(width, height, size=20):
    img = Image.new("RGB", (width, height), COLOR_CHECKER_1)
    draw = ImageDraw.Draw(img)
    for y in range(0, height, size):
        for x in range(0, width, size):
            if (x // size + y // size) % 2 == 1:
                draw.rectangle((x, y, x + size, y + size), fill=COLOR_CHECKER_2)
    return img

def resize_image_smart(img, max_size):
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def load_icon_image(size=(100, 100)):
    try:
        if os.path.exists(ICON_PATH):
            icon = Image.open(ICON_PATH).convert("RGBA")
            data = np.array(icon)
            white_mask = (data[:,:,0] > 230) & (data[:,:,1] > 230) & (data[:,:,2] > 230)
            data[white_mask] = [0, 0, 0, 0]
            icon = Image.fromarray(data, 'RGBA')
            icon = icon.resize(size, Image.Resampling.LANCZOS)

            colored_data = np.array(icon)
            for i in range(colored_data.shape[0]):
                for j in range(colored_data.shape[1]):
                    if colored_data[i, j, 3] > 0:
                        colored_data[i, j, 0] = min(255, int(colored_data[i, j, 0] * 0.2 + 0))
                        colored_data[i, j, 1] = min(255, int(colored_data[i, j, 1] * 0.7 + 212))
                        colored_data[i, j, 2] = min(255, int(colored_data[i, j, 2] * 0.9 + 255))
            
            icon = Image.fromarray(colored_data, 'RGBA')
            return ImageTk.PhotoImage(icon)
        else:
            return None
    except Exception as e:
        print(f"[Error] Al cargar icono: {e}")
        return None

class App:
    def __init__(self, mode):
        self.mode = mode
        self.root = tk.Tk()
        self.root.title(f"HarmonyOS - {mode.upper()}")
        self.root.configure(bg=COLOR_BG)

        self.current_image = None
        self.wobble_phase = 0
        self.glow_phase = 0

        self.is_grabbing = False    
        self.is_preparing = False   
        self.sender_done = False    
        
        self.rx_buffer = None       
        self.rx_state = "WAITING"  
        self.rx_reveal_progress = 0.0
        
        self.gallery_visible = False
        self.history_visible = False

        self.custom_icon = load_icon_image(size=(100, 100))
        self.cloud_particles = []
        for _ in range(30):
            self.cloud_particles.append({
                'x': 200 + random.randint(-80, 80), 'y': 250 + random.randint(-80, 80),
                'r': random.randint(15, 45), 'dx': random.uniform(-0.8, 0.8), 'dy': random.uniform(-0.8, 0.8),
                'color': random.choice([COLOR_ACCENT, "#4cc9f0", "#4895ef"]) 
            })

        # GESTOS Y RED
        self.queue = queue.Queue()
        self.running = True
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1)
        self.last_gesture = "NINGUNO"
        self.last_gesture_time = time.time()
        
        if self.mode == "sender":
            self.start_camera()
        else:
            self.start_server()

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.process_loop()
        self.root.mainloop()

    # CÁMARA
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self.camera_loop, daemon=True).start()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def camera_loop(self):
        print("[Cámara] Iniciando lectura de webcam...")
        frame_count = 0
        while self.running:
            if self.cap and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret:
                        frame_count += 1
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.hands.process(rgb)
                        
                        gesture = "NINGUNO"
                        if results.multi_hand_landmarks:
                            gesture = self.classify_gesture(results.multi_hand_landmarks[0])

                            if frame_count % 30 == 0:
                                print(f"[OJO] Mano detectada. Gesto actual: {gesture}")
                        else:
                            if frame_count % 60 == 0:
                                print("[BUSCANDO] No veo ninguna mano...")

                        if gesture != self.last_gesture:
                            if time.time() - self.last_gesture_time > 0.15:
                                print(f"--> ¡CAMBIO DE GESTO DETECTADO!: De {self.last_gesture} a {gesture}")
                                self.last_gesture = gesture
                                self.last_gesture_time = time.time()
                                self.queue.put(("gesture", gesture))
                    else:
                        print("[Error] No se pudo leer el frame de la cámara.")
                        time.sleep(0.1)
                except Exception as e:
                    print(f"[Error Crítico] En bucle de cámara: {e}")
                    time.sleep(0.1)
            else:
                print("[Aviso] Cámara no iniciada o desconectada.")
                time.sleep(0.5)
            time.sleep(0.03)

    def classify_gesture(self, landmarks):
        count = 0
        for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
            if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
                count += 1
        if count >= 3: return "PALMA"
        if count == 0: return "PUÑO"
        return "NINGUNO"

    # RED 
    def start_server(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_sock.bind(('0.0.0.0', PORT))
            self.server_sock.listen(1)
            print(f"[Sistema] Receptor listo en puerto {PORT}")
            threading.Thread(target=self.server_loop, daemon=True).start()
        except Exception as e:
            print(f"[Error] Server: {e}")

    def server_loop(self):
        while self.running:
            try:
                conn, addr = self.server_sock.accept()
                self.handle_client(conn)
            except: pass

    def handle_client(self, conn):
        try:
            raw_size = self.recvall(conn, struct.calcsize("Q"))
            if not raw_size: return
            size = struct.unpack("Q", raw_size)[0]
            data = self.recvall(conn, size)
            if data:
                image = pickle.loads(data)
                self.queue.put(("received_image", image))
        except: pass
        finally: conn.close()

    def recvall(self, sock, n):
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet: return None
            data += packet
        return data

    def send_image(self):
        def _send():
            try:
                print(f"[Emisor] Intentando conectar a {RECEIVER_IP}:{PORT}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5) 
                sock.connect((RECEIVER_IP, PORT))
                data = pickle.dumps(self.current_image)
                sock.sendall(struct.pack("Q", len(data)) + data)
                sock.close()
                print("[Emisor] Imagen enviada con éxito.")
            except Exception as e:
                print(f"[Emisor] Error al conectar o enviar: {e}")
        threading.Thread(target=_send, daemon=True).start()

    # UI
    def setup_ui(self):
        if self.mode == "sender":
            w, h = 700, 650
            center_window(self.root, w, h)
            self.root.columnconfigure(0, weight=1)
            self.root.columnconfigure(1, weight=0, minsize=0)
            self.root.rowconfigure(0, weight=1)
            self.root.resizable(False, False)

            self.panel_view = tk.Frame(self.root, bg=COLOR_PANEL)
            self.panel_view.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=15, pady=15)
            
            header = tk.Frame(self.panel_view, bg=COLOR_PANEL, height=70)
            header.pack(fill=tk.X, padx=20, pady=(20, 10))
            header.pack_propagate(False)
            
            self.btn_hamburger = tk.Button(header, text="☰", font=FONT_HAMBURGER, bg=COLOR_PANEL, fg=COLOR_ACCENT, bd=0, cursor="hand2", command=self.toggle_gallery)
            self.btn_hamburger.pack(side="right", padx=10)
            tk.Label(header, text="📤 EMISOR", font=FONT_TITLE, bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(anchor="w")
            tk.Label(header, text=f"Destino: {RECEIVER_IP}", font=("SF Pro Text", 10), bg=COLOR_PANEL, fg=COLOR_TEXT_DIM).pack(anchor="w", pady=(2,0))

            self.img_container = tk.Frame(self.panel_view, bg=COLOR_PANEL_LIGHT, highlightbackground=COLOR_ACCENT, highlightthickness=2)
            self.img_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
            self.img_container.pack_propagate(False)
            
            self.lbl_image = tk.Label(self.img_container, bg=COLOR_PANEL_LIGHT)
            self.lbl_image.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)
            
            self.lbl_icon = tk.Label(self.img_container, bg=COLOR_PANEL_LIGHT, text="📡", font=("Segoe UI Emoji", 40), fg=COLOR_ACCENT)
            if self.custom_icon:
                self.lbl_icon.configure(image=self.custom_icon, text="")
                self.lbl_icon.image = self.custom_icon

            self.panel_gallery = tk.Frame(self.root, bg=COLOR_PANEL_LIGHT, width=280)
            self.setup_gallery_ui(self.panel_gallery, GALLERY_DIR, self.select_sender_image, "🖼️ Galería")
            self.update_viewport(None)

        else:
            w, h = 550, 700
            center_window(self.root, w, h)
            self.root.columnconfigure(0, weight=1)
            self.root.columnconfigure(1, weight=0, minsize=0)
            self.root.rowconfigure(0, weight=1)
            self.root.resizable(False, False)

            self.frame_rx = tk.Frame(self.root, bg=COLOR_PANEL)
            self.frame_rx.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=15, pady=15)
            
            header = tk.Frame(self.frame_rx, bg=COLOR_PANEL, height=90)
            header.pack(fill=tk.X, padx=25, pady=(25, 15))
            header.pack_propagate(False)
            
            self.btn_hamburger = tk.Button(header, text="☰", font=FONT_HAMBURGER, bg=COLOR_PANEL, fg=COLOR_ACCENT, bd=0, cursor="hand2", command=self.toggle_history)
            self.btn_hamburger.pack(side="right", padx=10)
            tk.Label(header, text="📥 RECEPTOR", font=FONT_TITLE, bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(anchor="w")
            self.lbl_status = tk.Label(header, text="Esperando conexión...", font=FONT_STATUS, bg=COLOR_PANEL, fg=COLOR_TEXT_DIM)
            self.lbl_status.pack(anchor="w", pady=(8,0))
            
            view_area = tk.Frame(self.frame_rx, bg=COLOR_PANEL_LIGHT, highlightbackground=COLOR_ACCENT_2, highlightthickness=2)
            view_area.pack(fill=tk.BOTH, expand=True, padx=25, pady=(10, 25))
            
            self.container = tk.Frame(view_area, bg=COLOR_PANEL_LIGHT)
            self.container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.lbl_rx_icon = tk.Label(self.container, bg=COLOR_PANEL_LIGHT, text="📡", font=("Segoe UI Emoji", 40), fg=COLOR_ACCENT)
            if self.custom_icon:
                self.lbl_rx_icon.configure(image=self.custom_icon, text="")
                self.lbl_rx_icon.image = self.custom_icon
                
            self.cv_cloud = tk.Canvas(self.container, bg=COLOR_PANEL_LIGHT, highlightthickness=0)
            self.lbl_rx_img = tk.Label(self.container, bg=COLOR_PANEL_LIGHT)
            self.lbl_rx_img.place(relx=0.5, rely=0.5, anchor="center")

            self.panel_history = tk.Frame(self.root, bg=COLOR_PANEL_LIGHT, width=220)
            self.setup_gallery_ui(self.panel_history, RECEIVED_DIR, None, "📚 Historial")

    def setup_gallery_ui(self, parent, path, callback, title):
        parent.pack_propagate(False)
        header = tk.Frame(parent, bg=COLOR_PANEL_LIGHT, height=70)
        header.pack(fill=tk.X, padx=15, pady=(20, 10))
        header.pack_propagate(False)
        
        cmd = self.toggle_gallery if self.mode == "sender" else self.toggle_history
        tk.Button(header, text="✕", font=("SF Pro Display", 20, "bold"), bg=COLOR_PANEL_LIGHT, fg=COLOR_ACCENT, bd=0, cursor="hand2", command=cmd).pack(side="right", padx=5)
        tk.Label(header, text=title, font=FONT_TITLE, bg=COLOR_PANEL_LIGHT, fg=COLOR_TEXT).pack(anchor="w")
        
        canvas = tk.Canvas(parent, bg=COLOR_PANEL_LIGHT, highlightthickness=0)
        scroll = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.frm_thumbs = tk.Frame(canvas, bg=COLOR_PANEL_LIGHT)
        
        self.frm_thumbs.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.frm_thumbs, anchor="nw", width=parent.winfo_reqwidth()-20)
        canvas.configure(yscrollcommand=scroll.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.load_gallery_content(path, callback)

    def load_gallery_content(self, directory, callback):
        for widget in self.frm_thumbs.winfo_children(): widget.destroy()
        if not os.path.exists(directory): return
        
        files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], reverse=True)
        row, col = 0, 0
        for f in files:
            try:
                path = os.path.join(directory, f)
                img = Image.open(path)
                img = ImageOps.fit(img, (80, 80), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                btn = tk.Button(self.frm_thumbs, image=photo, bg=COLOR_PANEL, bd=0, cursor="hand2", command=lambda p=path: callback(p)) if callback else tk.Label(self.frm_thumbs, image=photo, bg=COLOR_PANEL, bd=2)
                btn.image = photo
                btn.grid(row=row, column=col, padx=8, pady=8)
                col += 1
                if col > 1: col=0; row+=1
            except: pass

    def select_sender_image(self, path):
        img = Image.open(path)
        self.current_image = resize_image_smart(img, MAX_IMG_SIZE)
        self.is_grabbing = False
        self.is_preparing = False
        self.sender_done = False
        self.start_camera() 
        self.update_viewport(self.current_image)
        print("[Emisor] Imagen seleccionada.")

    def update_viewport(self, pil_img):
        if self.mode == "sender":
            w = self.img_container.winfo_width() or 600
            h = self.img_container.winfo_height() or 500
            if pil_img is None:
                tk_img = ImageTk.PhotoImage(create_checkered_bg(w, h))
            else:
                img_copy = pil_img.copy()
                img_copy.thumbnail((w, h), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(img_copy)
            self.lbl_image.configure(image=tk_img)
            self.lbl_image.image = tk_img

    def process_loop(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "gesture":
                    self.handle_gesture(msg[1])
                elif msg[0] == "received_image":
                    self.rx_buffer = resize_image_smart(msg[1], MAX_IMG_SIZE)
                    self.rx_state = "READY_TO_REVEAL"
                    self.lbl_status.config(text="✓ Recibido - Haz gestos", fg=COLOR_SUCCESS)
                    self.start_camera()
        except queue.Empty: pass

        self.animate_ui()
        self.root.after(30, self.process_loop)

    def animate_ui(self):
        # EMISOR 
        if self.mode == "sender" and self.current_image:
            if self.sender_done:
                self.lbl_icon.place_forget()
                self.update_viewport(self.current_image)
            elif self.is_grabbing:
                self.lbl_icon.place_forget()
                liq = self.apply_liquid(self.current_image, 1.5, 0.85, True)
                self.update_viewport(liq)
            elif self.is_preparing:
                self.lbl_icon.place(relx=0.5, y=30, anchor="n")
                self.lbl_icon.lift()
                self.update_viewport(self.current_image)
            else:
                self.lbl_icon.place_forget()

        # RECEPTOR
        if self.mode == "receiver":
            if self.cv_cloud.winfo_ismapped():
                self.cv_cloud.delete("all")
                for p in self.cloud_particles:
                    p['x'] += p['dx']; p['y'] += p['dy']
                    if p['x'] < 120 or p['x'] > 280: p['dx'] *= -1
                    if p['y'] < 180 or p['y'] > 320: p['dy'] *= -1
                    self.cv_cloud.create_oval(p['x']-p['r'], p['y']-p['r'], p['x']+p['r'], p['y']+p['r'], fill=p['color'], outline="")

            if self.rx_state == "REVEALING" and self.rx_reveal_progress < 1.0:
                self.rx_reveal_progress += 0.03
                sc = 0.1 + (0.9 * self.rx_reveal_progress)
                dist = 1.0 - self.rx_reveal_progress
                img_anim = self.apply_liquid(self.rx_buffer, dist, sc, True)
                tk_img = ImageTk.PhotoImage(img_anim)
                self.lbl_rx_img.configure(image=tk_img)
                self.lbl_rx_img.image = tk_img
                if self.rx_reveal_progress >= 1.0:
                    self.rx_state = "DONE"
                    self.save_received_image()

    def apply_liquid(self, img, intensity, scale, animate):
        if not img: return None
        arr = np.array(img)
        rows, cols = arr.shape[:2]
        if animate: self.wobble_phase += 0.5
        amp_x, Y, X = 10.0 * intensity, *np.indices((rows, cols))
        map_x = X + amp_x * np.sin(Y/30.0 + self.wobble_phase)
        map_y = Y + amp_x * np.cos(X/30.0 + self.wobble_phase)
        dist = cv2.remap(arr, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderValue=(26, 31, 58))
        res = Image.fromarray(dist)
        if scale < 1.0:
            new_w, new_h = int(cols*scale), int(rows*scale)
            if new_w > 0 and new_h > 0:
                res = res.resize((new_w, new_h), Image.Resampling.LANCZOS)
                bg = Image.new("RGB", (cols, rows), COLOR_PANEL_LIGHT)
                bg.paste(res, ((cols-new_w)//2, (rows-new_h)//2))
                return bg
        return res

    def handle_gesture(self, gesture):
        # EMISOR
        if self.mode == "sender" and self.current_image and not self.sender_done:
            if gesture == "PALMA":
                if not self.is_grabbing:
                    self.is_preparing = True
            elif gesture == "PUÑO":
                if self.is_preparing:
                    self.is_grabbing = True
                    self.is_preparing = False
                    self.sender_done = True
                    print("[Emisor] Enviando imagen...")
                    self.send_image()

        # RECEPTOR
        if self.mode == "receiver" and self.rx_buffer and self.rx_state != "DONE":
            if gesture == "PUÑO":
                self.lbl_rx_icon.place(relx=0.5, y=60, anchor="n")
                self.cv_cloud.place(relx=0.5, rely=0.5, anchor="center", width=400, height=500)
                self.lbl_rx_icon.lift()
                self.lbl_status.config(text="Esperando a recibir imagen", fg=COLOR_TEXT_DIM)
            
            elif gesture == "PALMA":
                self.lbl_rx_icon.place_forget()
                self.cv_cloud.place_forget()
                self.lbl_status.config(text="Recibiendo imagen...", fg=COLOR_ACCENT)
                self.rx_state = "REVEALING"
            
            else:
                if self.rx_state != "REVEALING":
                    self.lbl_rx_icon.place_forget()
                    self.cv_cloud.place_forget()

    def save_received_image(self):
        self.lbl_status.config(text="✓ Guardado", fg=COLOR_SUCCESS)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RECEIVED_DIR, f"rx_{ts}.jpg")
        self.rx_buffer.save(path)
        self.load_gallery_content(RECEIVED_DIR, None) 
        self.root.after(3000, self.reset_receiver)

    def reset_receiver(self):
        self.rx_buffer = None
        self.rx_state = "WAITING"
        self.rx_reveal_progress = 0.0
        self.lbl_rx_img.configure(image='')
        self.stop_camera()
        self.lbl_status.config(text="Esperando conexión...", fg=COLOR_TEXT_DIM)

    def toggle_gallery(self):
        self.gallery_visible = not self.gallery_visible
        if self.gallery_visible:
            self.panel_view.grid(row=0, column=0, sticky="nsew")
            self.panel_gallery.grid(row=0, column=1, sticky="nsew")
            self.btn_hamburger.config(text="✕")
        else:
            self.panel_gallery.grid_forget()
            self.panel_view.grid(row=0, column=0, columnspan=2, sticky="nsew")
            self.btn_hamburger.config(text="☰")

    def toggle_history(self):
        self.history_visible = not self.history_visible
        if self.history_visible:
            self.frame_rx.grid(row=0, column=0, sticky="nsew")
            self.panel_history.grid(row=0, column=1, sticky="nsew")
            self.btn_hamburger.config(text="✕")
        else:
            self.panel_history.grid_forget()
            self.frame_rx.grid(row=0, column=0, columnspan=2, sticky="nsew")
            self.btn_hamburger.config(text="☰")

    def close(self):
        self.running = False
        self.stop_camera()
        if hasattr(self, 'server_sock'): self.server_sock.close()
        self.root.destroy()

# CONFIGURACIÓN DE EJECUCIÓN
if __name__ == "__main__":
    # AQUÍ DEFINIMOS QUE SOMOS EL EMISOR SIEMPRE
    App("sender")