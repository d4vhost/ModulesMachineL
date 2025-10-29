import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageOps
import cv2
import mediapipe as mp
import threading
import socket
import struct
import os
from zeroconf import ServiceInfo, ServiceBrowser, Zeroconf, ServiceListener
import time
import queue

COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
COLOR_GRAB = "#34c759"
COLOR_THROW = "#ff9500"
COLOR_ERROR = "#ff3b30"

FONT_TITLE = ("SF Pro Display", 18, "bold")
FONT_BODY = ("SF Pro Text", 11)
FONT_STATUS = ("SF Pro Text", 12, "bold")
FONT_SMALL = ("SF Pro Text", 9)

APP_PORT = 12346
SERVICE_TYPE = "_imagetransfer._tcp.local."
SERVICE_NAME = f"Device_{os.environ.get('COMPUTERNAME', 'PythonDevice')}"
BUFFER_SIZE = 4096
IMAGE_FOLDER = "images/9"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

os.makedirs(IMAGE_FOLDER, exist_ok=True) 
TEST_IMAGES = {}
print(f"Escaneando carpeta '{IMAGE_FOLDER}' en busca de imágenes...")
try:
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            filepath = os.path.join(IMAGE_FOLDER, filename)
            name_key = os.path.splitext(filename)[0]
            TEST_IMAGES[name_key] = filepath
            print(f"  + Encontrada: '{filename}' (Clave: '{name_key}')")
except Exception as e:
    print(f"Error al escanear la carpeta {IMAGE_FOLDER}: {e}")

if not TEST_IMAGES:
    print(f"ADVERTENCIA: No se encontraron imágenes soportadas en '{IMAGE_FOLDER}'.")
    placeholder_path = os.path.join(IMAGE_FOLDER, "placeholder.png")
    if not os.path.exists(placeholder_path):
        try:
            Image.new('RGB', (200, 150), color='lightgrey').save(placeholder_path)
        except Exception as e:
            print(f"No se pudo crear el placeholder: {e}")
    TEST_IMAGES = {"Error: Carpeta Vacía": placeholder_path}
else:
    print(f"Carga completa. {len(TEST_IMAGES)} imágenes listas.")


class NetworkManager:
    def __init__(self, app_queue):
        self.zeroconf = Zeroconf()
        self.listener = None
        self.server_thread = None
        self.discovered_devices = {}
        self.my_ip = self.get_local_ip()
        self.app_queue = app_queue 

    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try: 
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception: 
            ip = "127.0.0.1"
        finally: 
            s.close()
        return ip

    def start(self):
        self.server_thread = threading.Thread(target=self.run_file_server, daemon=True)
        self.server_thread.start()
        service_info = ServiceInfo(
            SERVICE_TYPE, f"{SERVICE_NAME}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(self.my_ip)], port=APP_PORT,
            properties={'name': SERVICE_NAME}
        )
        self.zeroconf.register_service(service_info)
        print(f"Servicio registrado: {SERVICE_NAME} en {self.my_ip}:{APP_PORT}")
        self.listener = DeviceListener(self)
        ServiceBrowser(self.zeroconf, SERVICE_TYPE, self.listener)

    def stop(self):
        self.zeroconf.unregister_all_services()
        self.zeroconf.close()

    def run_file_server(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind(("", APP_PORT))
                server_socket.listen(1)
                print(f"Servidor de archivos escuchando en el puerto {APP_PORT}...")
                while True:
                    conn, addr = server_socket.accept()
                    print(f"Conexión de archivo entrante desde {addr[0]}")
                    
                    filename_len = struct.unpack("<I", conn.recv(4))[0]
                    filename = conn.recv(filename_len).decode('utf-8')
                    filesize = struct.unpack("<Q", conn.recv(8))[0]

                    save_path = f"recibido_{filename}"
                    received_so_far = 0
                    with open(save_path, 'wb') as f:
                        while received_so_far < filesize:
                            bytes_read = conn.recv(min(BUFFER_SIZE, filesize - received_so_far))
                            if not bytes_read: break
                            f.write(bytes_read)
                            received_so_far += len(bytes_read)
                    conn.close()
                    print(f"Archivo '{save_path}' recibido con éxito.")
                    self.app_queue.put(("file_received", os.path.abspath(save_path), addr[0]))
        except Exception as e:
            print(f"Error en el servidor de archivos: {e}")

class DeviceListener(ServiceListener):
    def __init__(self, manager): 
        self.manager = manager
    def remove_service(self, zc, type, name):
        print(f"Servicio removido: {name}")
        if name in self.manager.discovered_devices:
            del self.manager.discovered_devices[name]
            self.manager.app_queue.put("device_update")
    def add_service(self, zc, type, name):
        info = zc.get_service_info(type, name)
        if info:
            ip = socket.inet_ntoa(info.addresses[0])
            if ip != self.manager.my_ip:
                print(f"Servicio agregado: {name} en {ip}")
                self.manager.discovered_devices[name] = ip
                self.manager.app_queue.put("device_update")
    def update_service(self, zc, type, name): 
        self.add_service(zc, type, name)


class GestureController:
    def __init__(self, app_queue):
        self.app_queue = app_queue
        self.cap = None
        self.running = False
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_state = "OPEN"
        self.last_gesture_time = time.time()
        self.palm_x, self.palm_y = 0, 0 
        self.last_palm_x, self.last_palm_y = 0, 0
        self.smoothing_factor = 0.6

    def start(self):
        self.cap = cv2.VideoCapture(0) 
        if not self.cap.isOpened():
            self.app_queue.put("camera_error")
            return
        self.running = True
        self.thread = threading.Thread(target=self.detect_gestures, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1.0)
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

    def detect_gestures(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            gesture = "NONE"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    h, w, c = frame.shape
                    cx = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * w)
                    cy = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * h)
                    
                    self.palm_x = int(self.last_palm_x * (1 - self.smoothing_factor) + cx * self.smoothing_factor)
                    self.palm_y = int(self.last_palm_y * (1 - self.smoothing_factor) + cy * self.smoothing_factor)
                    self.last_palm_x, self.last_palm_y = self.palm_x, self.palm_y

                    try:
                        index_tip_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                        index_mcp_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                        if (index_tip_y > index_mcp_y): 
                            gesture = "FIST" 
                        else: 
                            gesture = "OPEN" 
                    except Exception: 
                        pass
            
            self.app_queue.put(("video_frame", frame, (self.palm_x, self.palm_y)))
            
            current_time = time.time()
            if gesture != self.gesture_state and (current_time - self.last_gesture_time > 1.0): 
                if gesture == "FIST":
                    self.app_queue.put("gesture_grab")
                    self.gesture_state = "FIST"
                elif gesture == "OPEN":
                    self.app_queue.put("gesture_throw")
                    self.gesture_state = "OPEN"
                self.last_gesture_time = current_time
            
            time.sleep(0.01)


class GestureTransferApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transferencia Mágica v4")
        self.geometry("850x750")
        self.config(bg=COLOR_BG)
        self.resizable(False, False)

        self.selected_image_key = list(TEST_IMAGES.keys())[0]
        self.image_is_grabbed = False
        self.animation_in_progress = False
        
        self.app_queue = queue.Queue()
        self.network = NetworkManager(self.app_queue)
        self.gesture_control = GestureController(self.app_queue)
        
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_CARD, borderwidth=0)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat", borderwidth=0)
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Status.TLabel", font=FONT_STATUS, background=COLOR_BG)
        self.style.configure("Small.TLabel", font=FONT_SMALL, background=COLOR_CARD, foreground="#b0b0b0")
        
        self.style.configure("TButton", font=FONT_BODY, background=COLOR_ACCENT, foreground=COLOR_FG, borderwidth=0, padding=(10, 5))
        self.style.map("TButton",
            background=[("active", COLOR_FG), ("pressed", COLOR_FG), ("!disabled", COLOR_ACCENT)],
            foreground=[("active", COLOR_ACCENT), ("pressed", COLOR_ACCENT), ("!disabled", COLOR_FG)]
        )
        
        self.main_frame = ttk.Frame(self, padding=10)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        ttk.Label(self.main_frame, text="Módulo 9: Transferencia por Gestos", style="Title.TLabel").pack(pady=(10, 10))

        self.camera_container = ttk.Frame(self.main_frame, style="Card.TFrame", width=640, height=480)
        self.camera_container.pack(pady=10, padx=20)
        self.camera_container.pack_propagate(False)
        
        self.video_label = tk.Label(self.camera_container, text="Iniciando cámara...", bg=COLOR_CARD, fg=COLOR_FG, font=FONT_BODY, width=640, height=480)
        self.video_label.place(x=0, y=0)
        
        self.floating_image_label = tk.Label(self.main_frame, bg=COLOR_BG)

        self.image_frame = ttk.Frame(self.main_frame, style="Card.TFrame")
        self.image_frame.pack(pady=10, padx=20, fill="x")
        
        self.img_tk = self.load_display_image(TEST_IMAGES[self.selected_image_key])
        self.image_display_label = tk.Label(self.image_frame, image=self.img_tk, bg=COLOR_CARD)
        self.image_display_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        self.image_info_frame = ttk.Frame(self.image_frame, style="Card.TFrame")
        self.image_info_frame.pack(side=tk.LEFT, padx=20, pady=20, expand=True, fill="x")
        
        self.image_name_label = ttk.Label(self.image_info_frame, text=self.selected_image_key, font=FONT_TITLE, style="Card.TLabel")
        self.image_name_label.config(foreground=COLOR_FG)
        self.image_name_label.pack(anchor="w")
        
        self.image_file_label = ttk.Label(self.image_info_frame, text=TEST_IMAGES[self.selected_image_key], style="Small.TLabel")
        self.image_file_label.pack(anchor="w", pady=(5, 20))

        self.next_image_button = ttk.Button(self.image_info_frame, text="Siguiente Imagen", command=self.next_image)
        self.next_image_button.pack(anchor="w", side="left")
        
        if len(TEST_IMAGES) <= 1:
            self.next_image_button.pack_forget()

        self.status_frame = ttk.Frame(self, padding=10)
        self.status_frame.pack(fill="x", pady=10, padx=20, side="bottom")

        self.status_label = ttk.Label(self.status_frame, text="Estado: Buscando dispositivos...", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)
        
        self.devices_label = ttk.Label(self.status_frame, text="Dispositivos: Ninguno", style="Status.TLabel", foreground=COLOR_ACCENT)
        self.devices_label.pack(side=tk.RIGHT)
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.network.start()
        self.gesture_control.start()
        self.process_queue() 

    def create_rounded_image(self, pil_image, size, radius):
        img = ImageOps.fit(pil_image, size, Image.Resampling.LANCZOS)
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0) + size, radius, fill=255)
        img.putalpha(mask)
        return ImageTk.PhotoImage(img)

    def load_display_image(self, filepath, size=(200, 150), radius=15):
        try: img = Image.open(filepath)
        except Exception: img = Image.new('RGB', size, color=COLOR_CARD)
        return self.create_rounded_image(img, size, radius)

    def load_floating_image(self, filepath, size=(120, 90), radius=10):
        try: img = Image.open(filepath)
        except Exception: img = Image.new('RGB', size, color=COLOR_CARD)
        return self.create_rounded_image(img, size, radius)

    def next_image(self):
        if self.image_is_grabbed or self.animation_in_progress:
            self.update_status("Espera a que termine la animación", COLOR_ERROR)
            return
            
        current_index = list(TEST_IMAGES.keys()).index(self.selected_image_key)
        next_index = (current_index + 1) % len(TEST_IMAGES)
        self.selected_image_key = list(TEST_IMAGES.keys())[next_index]
        
        filepath = TEST_IMAGES[self.selected_image_key]
        self.img_tk = self.load_display_image(filepath)
        self.image_display_label.config(image=self.img_tk)
        self.image_name_label.config(text=self.selected_image_key)
        self.image_file_label.config(text=filepath)
        self.update_status(f"Viendo '{self.selected_image_key}'", COLOR_FG)

    def process_queue(self):
        try:
            while True:
                msg = self.app_queue.get_nowait()
                
                if isinstance(msg, tuple) and msg[0] == "video_frame":
                    frame, palm_pos = msg[1], msg[2]
                    
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.imgtk_video = self.create_rounded_image(img_pil, (640, 480), radius=15)
                    self.video_label.config(image=self.imgtk_video)
                    
                    if self.image_is_grabbed and not self.animation_in_progress:
                        img_w, img_h = self.floating_image_label.winfo_width(), self.floating_image_label.winfo_height()
                        
                        base_x = self.main_frame.winfo_x() + self.camera_container.winfo_x() + self.video_label.winfo_x()
                        base_y = self.main_frame.winfo_y() + self.camera_container.winfo_y() + self.video_label.winfo_y()
                        
                        x_pos = base_x + palm_pos[0] - (img_w // 2)
                        y_pos = base_y + palm_pos[1] - (img_h // 2)
                        
                        self.floating_image_label.place(x=x_pos, y=y_pos)
                    
                elif msg == "camera_error":
                    self.video_label.config(text="ERROR: No se pudo abrir la cámara.", font=FONT_TITLE, fg=COLOR_ERROR)
                
                elif msg == "device_update":
                    names = ", ".join(self.network.discovered_devices.keys())
                    if not names: names = "Ninguno"
                    self.devices_label.config(text=f"Dispositivos: {names}")
                
                elif msg == "gesture_grab":
                    if not self.image_is_grabbed and not self.animation_in_progress:
                        self.image_is_grabbed = True
                        self.animation_in_progress = True
                        self.update_status(f"¡Agarraste '{self.selected_image_key}'!", COLOR_GRAB)
                        palm_pos_rel = (self.gesture_control.palm_x, self.gesture_control.palm_y)
                        self.animate_grab(palm_pos_rel)
                
                elif msg == "gesture_throw":
                    if self.image_is_grabbed and not self.animation_in_progress:
                        self.image_is_grabbed = False
                        self.animation_in_progress = True
                        if self.network.discovered_devices:
                            self.update_status(f"¡Lanzando '{self.selected_image_key}'!", COLOR_THROW)
                            self.animate_throw_send()
                            self.send_selected_image() 
                        else:
                            self.update_status(f"¡Soltaste '{self.selected_image_key}'!", COLOR_ERROR)
                            self.animate_throw_back_to_shelf()
                
                elif isinstance(msg, tuple) and msg[0] == "file_received":
                    filepath, sender_ip = msg[1], msg[2]
                    self.show_received_image(filepath, sender_ip)
                
                elif isinstance(msg, str) and msg.startswith("Conectando"):
                    self.update_status(msg, COLOR_ACCENT)
                elif isinstance(msg, str) and msg.startswith("¡'"):
                    self.update_status(msg, COLOR_GRAB)
                elif isinstance(msg, str) and msg.startswith("Error"):
                    self.update_status(msg, COLOR_ERROR)

        except queue.Empty:
            pass 
        self.after(16, self.process_queue)

    def ease_out_cubic(self, t):
        return 1 - pow(1 - t, 4)

    def run_animation(self, image_label, start_x, start_y, end_x, end_y, steps=30, on_complete=None):
        dx = end_x - start_x
        dy = end_y - start_y

        def _animate_step(current_step):
            if current_step > steps:
                if on_complete:
                    on_complete()
                return

            t = current_step / steps
            eased_t = self.ease_out_cubic(t)

            new_x = start_x + (dx * eased_t)
            new_y = start_y + (dy * eased_t)
            
            image_label.place(x=new_x, y=new_y)
            self.after(16, _animate_step, current_step + 1)

        _animate_step(0)

    def animate_grab(self, end_pos_relative):
        self.update_idletasks()
        
        start_x = self.main_frame.winfo_x() + self.image_frame.winfo_x() + self.image_display_label.winfo_x()
        start_y = self.main_frame.winfo_y() + self.image_frame.winfo_y() + self.image_display_label.winfo_y()

        self.floating_img_tk = self.load_floating_image(TEST_IMAGES[self.selected_image_key])
        self.floating_image_label.config(image=self.floating_img_tk, bg=COLOR_BG)
        
        img_w, img_h = self.floating_img_tk.width(), self.floating_img_tk.height()
        
        base_x = self.main_frame.winfo_x() + self.camera_container.winfo_x() + self.video_label.winfo_x()
        base_y = self.main_frame.winfo_y() + self.camera_container.winfo_y() + self.video_label.winfo_y()
        
        end_x = base_x + end_pos_relative[0] - (img_w // 2)
        end_y = base_y + end_pos_relative[1] - (img_h // 2)

        self.image_display_label.pack_forget()
        self.image_info_frame.pack_forget()
        self.floating_image_label.place(x=start_x, y=start_y)
        self.floating_image_label.lift()

        def on_grab_complete():
            self.animation_in_progress = False

        self.run_animation(self.floating_image_label, start_x, start_y, end_x, end_y, steps=25, on_complete=on_grab_complete)

    def animate_throw_send(self):
        self.update_idletasks()
        start_x = self.floating_image_label.winfo_x()
        start_y = self.floating_image_label.winfo_y()
        
        end_x = self.winfo_width() + 100
        end_y = start_y - 200

        def on_send_complete():
            self.floating_image_label.place_forget()
            self.image_display_label.pack(side=tk.LEFT, padx=20, pady=20)
            self.image_info_frame.pack(side=tk.LEFT, padx=20, pady=20, expand=True, fill="x")
            self.next_image()
            self.animation_in_progress = False

        self.run_animation(self.floating_image_label, start_x, start_y, end_x, end_y, steps=30, on_complete=on_send_complete)

    def animate_throw_back_to_shelf(self):
        self.update_idletasks()
        start_x = self.floating_image_label.winfo_x()
        start_y = self.floating_image_label.winfo_y()
        
        end_x = self.main_frame.winfo_x() + self.image_frame.winfo_x() + 20
        end_y = self.main_frame.winfo_y() + self.image_frame.winfo_y() + 20
        
        self.image_display_label.pack_forget()
        self.image_info_frame.pack_forget()

        def on_return_complete():
            self.floating_image_label.place_forget()
            self.image_display_label.pack(side=tk.LEFT, padx=20, pady=20)
            self.image_info_frame.pack(side=tk.LEFT, padx=20, pady=20, expand=True, fill="x")
            self.animation_in_progress = False

        self.run_animation(self.floating_image_label, start_x, start_y, end_x, end_y, steps=35, on_complete=on_return_complete)

    def send_selected_image(self):
        if not self.network.discovered_devices:
            self.update_status("Error: No hay dispositivos", COLOR_ERROR)
            return
        target_name = list(self.network.discovered_devices.keys())[0]
        target_ip = self.network.discovered_devices[target_name]
        filepath = TEST_IMAGES[self.selected_image_key]
        if not os.path.exists(filepath) or "placeholder" in filepath:
            self.update_status("Error: No hay imagen real para enviar", COLOR_ERROR)
            self.animation_in_progress = False
            return
        threading.Thread(target=self.send_file_logic, args=(target_ip, filepath, target_name), daemon=True).start()

    def send_file_logic(self, host_ip, filepath, target_name):
        try:
            self.app_queue.put(f"Conectando a {target_name}...")
            filesize = os.path.getsize(filepath)
            filename = os.path.basename(filepath)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((host_ip, APP_PORT))
                client_socket.sendall(struct.pack("<I", len(filename.encode('utf-8'))))
                client_socket.sendall(filename.encode('utf-8'))
                client_socket.sendall(struct.pack("<Q", filesize))
                with open(filepath, 'rb') as f:
                    while True:
                        bytes_read = f.read(BUFFER_SIZE)
                        if not bytes_read: break
                        client_socket.sendall(bytes_read)
            self.app_queue.put(f"¡'{filename}' enviado a {target_name}!")
        except Exception as e:
            self.app_queue.put(f"Error al enviar a {target_name}: {e}")

    def show_received_image(self, filepath, sender_ip):
        recv_win = tk.Toplevel(self)
        recv_win.title("¡Imagen Recibida!")
        recv_win.geometry("400x450")
        recv_win.config(bg=COLOR_CARD)
        recv_win.resizable(False, False)

        style = ttk.Style(recv_win)
        style.theme_use("clam")
        style.configure(".", background=COLOR_CARD, foreground=COLOR_FG, borderwidth=0)
        style.configure("Recv.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        style.configure("Recv.Title.TLabel", background=COLOR_CARD, foreground=COLOR_ACCENT, font=("SF Pro Display", 16, "bold"))
        style.configure("Recv.TButton", font=FONT_BODY, background=COLOR_ACCENT, foreground=COLOR_FG, borderwidth=0, padding=(10, 5))
        style.map("Recv.TButton",
            background=[("active", COLOR_FG), ("pressed", COLOR_FG), ("!disabled", COLOR_ACCENT)],
            foreground=[("active", COLOR_ACCENT), ("pressed", COLOR_ACCENT), ("!disabled", COLOR_FG)]
        )
        
        ttk.Label(recv_win, text=f"Recibida de: {sender_ip}", style="Recv.Title.TLabel").pack(pady=(20, 5))
        ttk.Label(recv_win, text=f"Guardada como: {os.path.basename(filepath)}", style="Recv.TLabel").pack(pady=5, padx=20, fill="x")
        
        try:
            img_received_tk = self.load_display_image(filepath, size=(300, 250), radius=15)
            label_img = tk.Label(recv_win, image=img_received_tk, bg=COLOR_CARD)
            label_img.image = img_received_tk
            label_img.pack(pady=10, padx=10)
        except Exception as e:
            ttk.Label(recv_win, text=f"Error al mostrar imagen: {e}", style="Recv.TLabel", foreground=COLOR_ERROR).pack()
        
        ttk.Button(recv_win, text="Cerrar", command=recv_win.destroy, style="Recv.TButton").pack(pady=20, padx=20, fill="x")
        
        recv_win.transient(self)
        recv_win.grab_set()
        self.wait_window(recv_win)

    def update_status(self, text, color="black"):
        self.status_label.config(text=text, foreground=color)

    def on_closing(self):
        print("Cerrando la aplicación...")
        self.gesture_control.stop()
        self.network.stop()
        self.destroy()

if __name__ == "__main__":
    app = GestureTransferApp()
    app.mainloop()