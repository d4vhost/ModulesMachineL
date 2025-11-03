import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFilter
import cv2
import mediapipe as mp
import threading
import os
import time
import queue
import numpy as np
import socket
import pickle
import struct

# --- CONFIGURACIÓN DE ESTILO MEJORADA ---
COLOR_BG = "#000000"
COLOR_CARD = "#1c1c1e"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#ff6b00"  # Naranja Huawei
COLOR_SUCCESS = "#34c759"
COLOR_PLACEHOLDER = "#8e8e93"
COLOR_ERROR = "#ff3b30"
COLOR_GLASS = "#2c2c2e"

FONT_TITLE = ("SF Pro Display", 24, "bold")
FONT_SUBTITLE = ("SF Pro Display", 16)
FONT_BODY = ("SF Pro Text", 11)
FONT_STATUS = ("SF Pro Text", 13, "bold")
FONT_PHONE_TITLE = ("SF Pro Text", 10, "bold")

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR = os.path.join(BASE_DIR, "data", "gallery_images")
os.makedirs(GALLERY_DIR, exist_ok=True)

# --- CONFIGURACIÓN DE RED ---
HOST = '127.0.0.1'
PORT = 9999

# --- CONTROLADOR DE GESTOS ---
class GestureController:
    def __init__(self, app_queue):
        self.app_queue = app_queue
        self.cap = None
        self.running = False
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_gesture = "NINGUNO"
        self.last_gesture_time = time.time()
        self.debounce_time = 0.5
        self.hand_position = None

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.app_queue.put(("error", "No se pudo abrir la cámara"))
            return
        self.running = True
        self.thread = threading.Thread(target=self.detect_gestures_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def classify_gesture(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        
        # Detectar dedos extendidos
        fingers_extended = []
        
        # Pulgar (requiere lógica especial)
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        fingers_extended.append(thumb_tip.x < thumb_ip.x if thumb_tip.x < 0.5 else thumb_tip.x > thumb_ip.x)
        
        # Resto de dedos
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_pips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers_extended.append(landmarks[tip].y < landmarks[pip].y)
        
        # Clasificar gestos
        if sum(fingers_extended) == 0:
            return "PUÑO"
        elif sum(fingers_extended) >= 4:
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
            hand_pos = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks con estilo mejorado
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(255, 107, 0), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    gesture_name = self.classify_gesture(hand_landmarks)
                    
                    # Obtener posición de la mano (centro de la palma)
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    hand_pos = {
                        'x': (wrist.x + middle_mcp.x) / 2,
                        'y': (wrist.y + middle_mcp.y) / 2
                    }
            
            current_time = time.time()
            if gesture_name != self.last_gesture:
                if (current_time - self.last_gesture_time > self.debounce_time):
                    self.last_gesture = gesture_name
                    self.last_gesture_time = current_time
                    self.app_queue.put(("gesture", gesture_name, hand_pos))
            
            self.hand_position = hand_pos
            self.app_queue.put(("video_frame", frame))
            time.sleep(0.01)

# --- SERVIDOR DE TRANSFERENCIA ---
class TransferServer:
    def __init__(self, app_queue):
        self.app_queue = app_queue
        self.server_socket = None
        self.running = False

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((HOST, PORT))
            self.server_socket.listen(1)
            self.running = True
            self.thread = threading.Thread(target=self.accept_connections, daemon=True)
            self.thread.start()
            print(f"✓ Servidor iniciado en {HOST}:{PORT}")
        except Exception as e:
            print(f"Error iniciando servidor: {e}")
            self.app_queue.put(("error", f"Error de servidor: {e}"))

    def accept_connections(self):
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client_socket, addr = self.server_socket.accept()
                print(f"Conexión desde {addr}")
                self.handle_client(client_socket)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error aceptando conexión: {e}")

    def handle_client(self, client_socket):
        try:
            # Recibir tamaño del mensaje
            data = b""
            payload_size = struct.calcsize("Q")
            
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                data += packet
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            
            # Recibir imagen
            while len(data) < msg_size:
                data += client_socket.recv(4096)
            
            frame_data = data[:msg_size]
            image = pickle.loads(frame_data)
            
            self.app_queue.put(("received_image", image))
            print("✓ Imagen recibida")
            
        except Exception as e:
            print(f"Error recibiendo imagen: {e}")
        finally:
            client_socket.close()

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()

# --- CLIENTE DE TRANSFERENCIA ---
class TransferClient:
    @staticmethod
    def send_image(image_pil):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))
            
            # Serializar imagen
            image_bytes = pickle.dumps(image_pil)
            message_size = struct.pack("Q", len(image_bytes))
            
            client_socket.sendall(message_size + image_bytes)
            client_socket.close()
            print("✓ Imagen enviada")
            return True
        except Exception as e:
            print(f"Error enviando imagen: {e}")
            return False

# --- APLICACIÓN PRINCIPAL ---
class HarmonyOSApp:
    def __init__(self, mode="sender"):
        self.mode = mode  # "sender" o "receiver"
        self.root = tk.Tk()
        self.root.title(f"HarmonyOS {'Emisor' if mode == 'sender' else 'Receptor'}")
        self.root.configure(bg=COLOR_BG)
        
        # Variables de estado
        self.image_grabbed = False
        self.selected_pil_image = None
        self.wobble_phase = 0
        self.gallery_images = []
        self.transfer_animation_phase = 0
        
        self.app_queue = queue.Queue()
        
        if mode == "sender":
            self.setup_sender_ui()
            self.gesture_control = GestureController(self.app_queue)
            self.gesture_control.start()
        else:
            self.setup_receiver_ui()
            self.server = TransferServer(self.app_queue)
            self.server.start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_queue()

    def setup_sender_ui(self):
        self.root.geometry("1200x750")
        
        # Contenedor principal con dos paneles
        main_container = tk.Frame(self.root, bg=COLOR_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # --- PANEL IZQUIERDO: Cámara y Controles ---
        left_panel = tk.Frame(main_container, bg=COLOR_CARD, width=700, height=700)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Header
        header = tk.Frame(left_panel, bg=COLOR_CARD, height=60)
        header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(header, text="HarmonyOS SuperDevice", font=FONT_TITLE,
                bg=COLOR_CARD, fg=COLOR_FG).pack(anchor="w")
        tk.Label(header, text="Transferencia por gestos", font=FONT_BODY,
                bg=COLOR_CARD, fg=COLOR_PLACEHOLDER).pack(anchor="w")
        
        # Contenedor de video/imagen
        self.camera_container = tk.Frame(left_panel, bg=COLOR_BG, width=660, height=495)
        self.camera_container.pack(padx=20, pady=10)
        self.camera_container.pack_propagate(False)
        
        self.video_label = tk.Label(self.camera_container, bg=COLOR_BG)
        self.video_label.place(x=0, y=0, width=660, height=495)
        
        self.image_overlay = tk.Label(self.camera_container, bg=COLOR_BG)
        self.image_overlay.place(relx=0.5, rely=0.5, anchor="center")
        
        # Panel de estado con efecto glass
        status_panel = tk.Frame(left_panel, bg=COLOR_GLASS, height=80)
        status_panel.pack(fill=tk.X, padx=20, pady=(10, 20))
        
        self.gesture_indicator = tk.Label(status_panel, text="👋", font=("SF Pro Display", 32),
                                         bg=COLOR_GLASS, fg=COLOR_ACCENT)
        self.gesture_indicator.pack(side=tk.LEFT, padx=20)
        
        status_text_frame = tk.Frame(status_panel, bg=COLOR_GLASS)
        status_text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(status_text_frame, text="Selecciona una imagen de la galería",
                                     font=FONT_STATUS, bg=COLOR_GLASS, fg=COLOR_FG, anchor="w")
        self.status_label.pack(anchor="w", pady=(15, 5))
        
        self.instruction_label = tk.Label(status_text_frame, text="✊ Agarrar  •  🖐️ Soltar",
                                         font=FONT_BODY, bg=COLOR_GLASS, fg=COLOR_PLACEHOLDER, anchor="w")
        self.instruction_label.pack(anchor="w")
        
        # --- PANEL DERECHO: Galería ---
        right_panel = tk.Frame(main_container, bg=COLOR_CARD, width=450, height=700)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Header de galería
        gallery_header = tk.Frame(right_panel, bg=COLOR_CARD, height=60)
        gallery_header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(gallery_header, text="Galería", font=FONT_SUBTITLE,
                bg=COLOR_CARD, fg=COLOR_FG).pack(anchor="w")
        
        # Scroll de galería
        canvas = tk.Canvas(right_panel, bg=COLOR_CARD, highlightthickness=0)
        scrollbar = tk.Scrollbar(right_panel, orient="vertical", command=canvas.yview,
                                bg=COLOR_CARD, troughcolor=COLOR_BG)
        self.gallery_frame = tk.Frame(canvas, bg=COLOR_CARD)
        
        self.gallery_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.gallery_frame, anchor="nw", width=410)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=(20, 0))
        scrollbar.pack(side="right", fill="y", padx=(0, 20))
        
        self.load_gallery()

    def setup_receiver_ui(self):
        self.root.geometry("500x750")
        
        # Contenedor principal
        main_container = tk.Frame(self.root, bg=COLOR_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Panel receptor
        receiver_panel = tk.Frame(main_container, bg=COLOR_CARD)
        receiver_panel.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = tk.Frame(receiver_panel, bg=COLOR_CARD, height=80)
        header.pack(fill=tk.X, padx=30, pady=(30, 20))
        
        tk.Label(header, text="Dispositivo Receptor", font=FONT_TITLE,
                bg=COLOR_CARD, fg=COLOR_FG).pack(anchor="w")
        tk.Label(header, text="Esperando transferencia...", font=FONT_BODY,
                bg=COLOR_CARD, fg=COLOR_PLACEHOLDER).pack(anchor="w", pady=(5, 0))
        
        # Área de recepción
        self.receiver_area = tk.Frame(receiver_panel, bg=COLOR_BG, width=440, height=550)
        self.receiver_area.pack(padx=30, pady=20)
        self.receiver_area.pack_propagate(False)
        
        # Placeholder animado
        self.receiver_placeholder = tk.Frame(self.receiver_area, bg=COLOR_BG)
        self.receiver_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        
        self.placeholder_icon = tk.Label(self.receiver_placeholder, text="📱", font=("SF Pro Display", 64),
                                        bg=COLOR_BG, fg=COLOR_PLACEHOLDER)
        self.placeholder_icon.pack()
        
        tk.Label(self.receiver_placeholder, text="Listo para recibir", font=FONT_SUBTITLE,
                bg=COLOR_BG, fg=COLOR_PLACEHOLDER).pack(pady=(10, 0))
        
        self.receiver_image_label = tk.Label(self.receiver_area, bg=COLOR_BG)
        self.receiver_image_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.animate_receiver_placeholder()

    def load_gallery(self):
        row, col = 0, 0
        img_size = 120
        
        for filename in sorted(os.listdir(GALLERY_DIR)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(GALLERY_DIR, filename)
                
                try:
                    img_pil = Image.open(filepath)
                    img_pil.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
                    
                    # Crear thumbnail con bordes redondeados y sombra
                    img_rounded = self.create_rounded_thumbnail(img_pil, 15)
                    img_tk = ImageTk.PhotoImage(img_rounded)
                    
                    # Contenedor para efecto hover
                    btn_container = tk.Frame(self.gallery_frame, bg=COLOR_CARD, 
                                            highlightthickness=2, highlightbackground=COLOR_BG)
                    btn_container.grid(row=row, column=col, padx=8, pady=8)
                    
                    btn = tk.Label(btn_container, image=img_tk, bg=COLOR_CARD, cursor="hand2")
                    btn.image = img_tk
                    btn.pack()
                    
                    # Bind events
                    btn.bind("<Button-1>", lambda e, p=filepath: self.select_image(p))
                    btn.bind("<Enter>", lambda e, c=btn_container: c.configure(highlightbackground=COLOR_ACCENT))
                    btn.bind("<Leave>", lambda e, c=btn_container: c.configure(highlightbackground=COLOR_BG))
                    
                    col += 1
                    if col > 2:
                        col = 0
                        row += 1
                        
                except Exception as e:
                    print(f"Error cargando {filename}: {e}")

    def create_rounded_thumbnail(self, img_pil, radius):
        # Crear máscara con bordes redondeados
        mask = Image.new('L', img_pil.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0) + img_pil.size, radius, fill=255)
        
        # Aplicar máscara
        output = ImageOps.fit(img_pil, img_pil.size, centering=(0.5, 0.5))
        output.putalpha(mask)
        
        # Crear imagen final con fondo
        final = Image.new("RGBA", img_pil.size, COLOR_CARD)
        final.paste(output, (0, 0), output)
        
        return final.convert("RGB")

    def select_image(self, path):
        try:
            self.selected_pil_image = Image.open(path)
            # Mantener relación de aspecto
            self.selected_pil_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Crear versión con bordes redondeados
            img_rounded = self.create_rounded_thumbnail(self.selected_pil_image, 20)
            img_tk = ImageTk.PhotoImage(img_rounded)
            
            self.image_overlay.config(image=img_tk)
            self.image_overlay.image = img_tk
            
            self.status_label.config(text="Imagen seleccionada - Usa gestos para transferir")
            self.gesture_indicator.config(text="👌", fg=COLOR_SUCCESS)
            
        except Exception as e:
            print(f"Error seleccionando imagen: {e}")

    def process_queue(self):
        try:
            while True:
                msg = self.app_queue.get_nowait()
                
                if msg[0] == "video_frame":
                    frame = msg[1]
                    frame_resized = cv2.resize(frame, (660, 495))
                    img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    self.video_label.configure(image=img_tk)
                    self.video_label.image = img_tk
                
                elif msg[0] == "gesture":
                    self.handle_gesture(msg[1], msg[2] if len(msg) > 2 else None)
                
                elif msg[0] == "received_image":
                    self.display_received_image(msg[1])
                
                elif msg[0] == "error":
                    self.status_label.config(text=f"Error: {msg[1]}", fg=COLOR_ERROR)
        
        except queue.Empty:
            pass
        
        # Aplicar efecto wobble si la imagen está agarrada
        if self.mode == "sender" and self.image_grabbed and self.selected_pil_image:
            self.apply_wobble_effect()
        
        self.root.after(16, self.process_queue)

    def handle_gesture(self, gesture, hand_pos):
        if not self.selected_pil_image:
            return
        
        if gesture == "PUÑO" and not self.image_grabbed:
            self.image_grabbed = True
            self.status_label.config(text="¡Imagen agarrada! Mueve tu mano para soltar")
            self.gesture_indicator.config(text="✊", fg=COLOR_ACCENT)
            
        elif gesture == "PALMA" and self.image_grabbed:
            self.image_grabbed = False
            self.status_label.config(text="Transfiriendo imagen...")
            self.gesture_indicator.config(text="📤", fg=COLOR_SUCCESS)
            
            # Iniciar transferencia en hilo separado
            threading.Thread(target=self.transfer_image, daemon=True).start()

    def transfer_image(self):
        if TransferClient.send_image(self.selected_pil_image):
            self.root.after(0, lambda: self.status_label.config(
                text="✓ Imagen transferida exitosamente", fg=COLOR_SUCCESS))
            self.root.after(0, lambda: self.gesture_indicator.config(text="✅"))
            
            # Resetear después de 2 segundos
            self.root.after(2000, self.reset_state)
        else:
            self.root.after(0, lambda: self.status_label.config(
                text="✗ Error en la transferencia", fg=COLOR_ERROR))

    def reset_state(self):
        self.image_overlay.config(image=None)
        self.image_overlay.image = None
        self.selected_pil_image = None
        self.status_label.config(text="Selecciona otra imagen", fg=COLOR_FG)
        self.gesture_indicator.config(text="👋", fg=COLOR_ACCENT)

    def apply_wobble_effect(self):
        if not self.selected_pil_image:
            return
        
        try:
            img_array = np.array(self.selected_pil_image)
            rows, cols = img_array.shape[:2]
            
            # Crear mapas de distorsión
            map_x = np.zeros((rows, cols), dtype=np.float32)
            map_y = np.zeros((rows, cols), dtype=np.float32)
            
            self.wobble_phase += 0.3
            
            for i in range(rows):
                for j in range(cols):
                    offset_x = int(8.0 * np.sin(2.0 * np.pi * i / 120 + self.wobble_phase))
                    offset_y = int(5.0 * np.cos(2.0 * np.pi * j / 120 + self.wobble_phase))
                    
                    map_x[i, j] = j + offset_x
                    map_y[i, j] = i + offset_y
            
            # Aplicar distorsión
            distorted = cv2.remap(img_array, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            img_pil = Image.fromarray(distorted)
            
            # Aplicar bordes redondeados
            img_rounded = self.create_rounded_thumbnail(img_pil, 20)
            img_tk = ImageTk.PhotoImage(img_rounded)
            
            self.image_overlay.config(image=img_tk)
            self.image_overlay.image = img_tk
            
        except Exception as e:
            print(f"Error en wobble: {e}")

    def display_received_image(self, image_pil):
        try:
            # Ocultar placeholder
            self.receiver_placeholder.place_forget()
            
            # Mostrar imagen con animación
            image_pil.thumbnail((420, 530), Image.Resampling.LANCZOS)
            img_rounded = self.create_rounded_thumbnail(image_pil, 20)
            
            # Efecto de aparición con blur
            img_tk = ImageTk.PhotoImage(img_rounded)
            self.receiver_image_label.config(image=img_tk)
            self.receiver_image_label.image = img_tk
            
            print("✓ Imagen mostrada en receptor")
            
        except Exception as e:
            print(f"Error mostrando imagen: {e}")

    def animate_receiver_placeholder(self):
        if self.mode == "receiver":
            # Animación de pulso para el placeholder
            current_color = self.placeholder_icon.cget("fg")
            new_color = COLOR_ACCENT if current_color == COLOR_PLACEHOLDER else COLOR_PLACEHOLDER
            self.placeholder_icon.config(fg=new_color)
            self.root.after(1000, self.animate_receiver_placeholder)

    def on_closing(self):
        print("Cerrando aplicación...")
        if self.mode == "sender":
            self.gesture_control.stop()
        else:
            self.server.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# --- EJECUTAR ---
if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("HarmonyOS SuperDevice - Transferencia por Gestos")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "receiver":
        print("\n🔵 Iniciando en modo RECEPTOR...")
        app = HarmonyOSApp(mode="receiver")
    else:
        print("\n🟠 Iniciando en modo EMISOR...")
        print("\nPara abrir el receptor, ejecuta:")
        print("python app.py receiver")
        app = HarmonyOSApp(mode="sender")
    
    app.run()