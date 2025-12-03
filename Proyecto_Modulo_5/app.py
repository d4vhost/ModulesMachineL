import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw
import cv2
import easyocr
import threading
import queue
import time
import numpy as np
from collections import deque

COLOR_BG = "#f5f7fa"          
COLOR_CARD = "#ffffff"        
COLOR_FG = "#2c3e50"          
COLOR_ACCENT = "#3b82f6"     
COLOR_SUCCESS = "#10b981"    
COLOR_ERROR = "#ef4444"       
COLOR_WARNING = "#f59e0b"     
COLOR_BORDER = "#e5e7eb"      
COLOR_SHADOW = "#d1d5db"      

FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_SUBTITLE = ("Segoe UI", 10)
FONT_BODY = ("Segoe UI", 11)
FONT_STATUS = ("Segoe UI", 13, "bold")

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
VIDEO_FPS = 30

PROCESS_EVERY_N_FRAMES = 6
PROCESS_SCALE_FACTOR = 0.5
DISPLAY_SCALE = 0.8

MIN_CONFIDENCE = 0.25
MIN_PLATE_CHARS = 4
MAX_TRACKING_AGE = 1.0
IOU_THRESHOLD = 0.25

class PlateTracker:
    """Maneja el seguimiento de placas detectadas."""
    
    def __init__(self):
        self.tracked_plates = {}
        self.next_id = 0
    
    def _calculate_iou(self, box1, box2):
        """Calcula IoU entre dos bounding boxes."""
        def bbox_to_rect(bbox):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            return [min(xs), min(ys), max(xs), max(ys)]
        
        r1 = bbox_to_rect(box1)
        r2 = bbox_to_rect(box2)
        
        x_left = max(r1[0], r2[0])
        y_top = max(r1[1], r2[1])
        x_right = min(r1[2], r2[2])
        y_bottom = min(r1[3], r2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
        area2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections, current_time):
        """Actualiza tracking con nuevas detecciones."""
        to_remove = [pid for pid, data in self.tracked_plates.items() 
                     if current_time - data['last_seen'] > MAX_TRACKING_AGE]
        for pid in to_remove:
            del self.tracked_plates[pid]
        
        for bbox, text, conf in detections:
            best_match = None
            best_iou = IOU_THRESHOLD
            
            for pid, data in self.tracked_plates.items():
                iou = self._calculate_iou(bbox, data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = pid
            
            if best_match:
                old = self.tracked_plates[best_match]
                new_conf = old['confidence'] * 0.6 + conf * 0.4
                new_text = text if conf > old['best_conf'] else old['text']
                best_conf = max(conf, old['best_conf'])
                
                self.tracked_plates[best_match] = {
                    'bbox': bbox,
                    'text': new_text,
                    'confidence': new_conf,
                    'best_conf': best_conf,
                    'last_seen': current_time,
                    'detections': old['detections'] + 1
                }
            else:
                self.tracked_plates[self.next_id] = {
                    'bbox': bbox,
                    'text': text,
                    'confidence': conf,
                    'best_conf': conf,
                    'last_seen': current_time,
                    'detections': 1
                }
                self.next_id += 1
        
        return [(d['bbox'], d['text'], d['confidence']) 
                for d in self.tracked_plates.values()
                if d['detections'] >= 2]
    
    def get_all_active(self, current_time):
        """Obtiene todas las placas activas."""
        return [(d['bbox'], d['text'], d['confidence']) 
                for d in self.tracked_plates.values()
                if current_time - d['last_seen'] < MAX_TRACKING_AGE]


class LicensePlateRecognizer:
    def __init__(self, app_queue):
        self.app_queue = app_queue
        self.cap = None
        self.running = False
        self.processing = False
        
        self.tracker = PlateTracker()
        self.frame_counter = 0
        self.process_queue = queue.Queue(maxsize=1)
        
        self.reader = None
        self.reader_ready = False
        threading.Thread(target=self._init_easyocr, daemon=True).start()
        
        self.allow_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'

    def _init_easyocr(self):
        """Inicializa EasyOCR."""
        try:
            print("Cargando EasyOCR...")
            self.reader = easyocr.Reader(
                ['es', 'en'], 
                gpu=True, 
                verbose=False,
                quantize=True
            )
            self.reader_ready = True
            print("EasyOCR listo (GPU)")
            self.app_queue.put(("status", "EasyOCR listo", COLOR_SUCCESS))
        except Exception as e:
            try:
                print(f"GPU no disponible: {e}")
                self.reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)
                self.reader_ready = True
                print("EasyOCR listo (CPU)")
                self.app_queue.put(("status", "EasyOCR listo (CPU)", COLOR_WARNING))
            except Exception as e2:
                print(f"Error: {e2}")
                self.app_queue.put(("error", "easyocr_error"))

    def start(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.app_queue.put(("error", "camera_error"))
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            self.running = True
            
            threading.Thread(target=self._capture_loop, daemon=True).start()
            threading.Thread(target=self._processing_loop, daemon=True).start()
            
        except Exception as e:
            self.app_queue.put(("error", f"Error: {e}"))

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print("Detenido")

    def _capture_loop(self):
        """Loop de captura"""
        while self.running:
            if not self.cap or not self.cap.isOpened():
                break
                
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self.frame_counter += 1
            
            current_time = time.time()
            active_plates = self.tracker.get_all_active(current_time)
            
            self.app_queue.put(("video_frame", frame, active_plates))
            
            if (self.frame_counter % PROCESS_EVERY_N_FRAMES == 0 and 
                self.reader_ready and not self.processing):
                
                while not self.process_queue.empty():
                    try:
                        self.process_queue.get_nowait()
                    except:
                        break
                
                try:
                    self.process_queue.put_nowait((frame.copy(), current_time))
                except:
                    pass
            
            time.sleep(0.001)

    def _processing_loop(self):
        """Loop de procesamiento OCR"""
        while self.running:
            try:
                frame, timestamp = self.process_queue.get(timeout=1.0)
                
                if not self.reader_ready:
                    continue
                
                self.processing = True
                
                h, w = frame.shape[:2]
                small_w = int(w * PROCESS_SCALE_FACTOR)
                small_h = int(h * PROCESS_SCALE_FACTOR)
                small_frame = cv2.resize(frame, (small_w, small_h), 
                                        interpolation=cv2.INTER_LINEAR)
                
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                
                results = self.reader.readtext(
                    sharpened,
                    detail=1,
                    allowlist=self.allow_list,
                    paragraph=False,
                    batch_size=1,
                    text_threshold=0.6,
                    low_text=0.3,
                    link_threshold=0.3
                )
                
                scale_inv = 1.0 / PROCESS_SCALE_FACTOR
                detections = []
                
                for bbox, text, prob in results:
                    if prob >= MIN_CONFIDENCE:
                        text_clean = "".join(text.split()).upper()
                        
                        if (len(text_clean) >= MIN_PLATE_CHARS and 
                            any(c.isdigit() for c in text_clean)):
                            
                            scaled_bbox = [[int(p[0]*scale_inv), int(p[1]*scale_inv)] 
                                          for p in bbox]
                            detections.append((scaled_bbox, text_clean, prob))
                
                if detections:
                    tracked = self.tracker.update(detections, timestamp)
                    if tracked:
                        texts = [t for _, t, _ in tracked]
                        self.app_queue.put(("status", 
                                          f"{len(tracked)} placa(s): {', '.join(texts[:2])}", 
                                          COLOR_SUCCESS))
                
                self.processing = False
                
            except queue.Empty:
                self.processing = False
            except Exception as e:
                print(f"Error OCR: {e}")
                self.processing = False
                time.sleep(0.1)


class PlateRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🚗 Reconocimiento de Placas Vehiculares")
        self.geometry("920x750")
        self.config(bg=COLOR_BG)
        self.resizable(False, False)
        
        self.center_window()

        self.app_queue = queue.Queue()
        self.recognizer = LicensePlateRecognizer(self.app_queue)
        
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, 
                           fieldbackground=COLOR_CARD, borderwidth=0)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Subtitle.TLabel", font=FONT_SUBTITLE, background=COLOR_BG, foreground=COLOR_ACCENT)
        self.style.configure("Status.TLabel", font=FONT_STATUS, background=COLOR_CARD)
        
        self.create_ui()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.recognizer.start()
        self.process_queue()

    def center_window(self):
        """Centra la ventana en la pantalla"""
        self.update_idletasks()
        width = 920
        height = 750
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def create_ui(self):
        """Crea la interfaz mejorada"""
        self.main_frame = ttk.Frame(self, padding=20, style="TFrame")
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        header_frame = ttk.Frame(self.main_frame, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, 
                               text="🚗 Reconocimiento de Placas", 
                               style="Title.TLabel")
        title_label.pack()
        
        info = (f"Resolución: {CAMERA_WIDTH}×{CAMERA_HEIGHT} • "
                f"FPS: {VIDEO_FPS} • Procesamiento: 1/{PROCESS_EVERY_N_FRAMES}")
        subtitle_label = ttk.Label(header_frame, text=info, style="Subtitle.TLabel")
        subtitle_label.pack(pady=(5, 0))

        video_card = tk.Frame(self.main_frame, bg=COLOR_CARD, 
                             highlightbackground=COLOR_BORDER,
                             highlightthickness=1)
        video_card.pack(expand=True, pady=(0, 20))

        shadow = tk.Frame(self.main_frame, bg=COLOR_SHADOW, height=2)
        shadow.place(in_=video_card, relx=0.02, rely=1, relwidth=0.96)
        
        self.camera_container = tk.Frame(video_card, bg=COLOR_CARD,
                                        width=820, height=470)
        self.camera_container.pack(padx=10, pady=10)
        self.camera_container.pack_propagate(False)
        
        self.video_label = tk.Label(self.camera_container, 
                                    text="⏳ Iniciando cámara...", 
                                    bg=COLOR_CARD, fg=COLOR_FG, font=FONT_BODY)
        self.video_label.pack(expand=True)
        
        status_card = tk.Frame(self.main_frame, bg=COLOR_CARD,
                              highlightbackground=COLOR_BORDER,
                              highlightthickness=1)
        status_card.pack(fill="x")
        
        status_inner = ttk.Frame(status_card, style="Card.TFrame", padding=15)
        status_inner.pack(fill="x")

        status_left = ttk.Frame(status_inner, style="Card.TFrame")
        status_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.status_indicator = tk.Canvas(status_left, width=12, height=12, 
                                         bg=COLOR_CARD, highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 10))
        self.status_dot = self.status_indicator.create_oval(2, 2, 10, 10, 
                                                           fill=COLOR_WARNING, 
                                                           outline="")
        
        self.status_label = ttk.Label(status_left, 
                                     text="Estado: Iniciando sistema...", 
                                     style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)
        
        fps_frame = tk.Frame(status_inner, bg=COLOR_ACCENT, 
                            highlightbackground=COLOR_ACCENT,
                            highlightthickness=0)
        fps_frame.pack(side=tk.RIGHT)
        
        self.fps_label = tk.Label(fps_frame, text="FPS: --", 
                                 font=("Segoe UI", 10, "bold"), 
                                 fg="#ffffff", bg=COLOR_ACCENT,
                                 padx=12, pady=4)
        self.fps_label.pack()
        
        # CORRECCIÓN: Inicializar last_frame_time correctamente
        self.last_frame_time = time.time()
        self.fps_history = deque(maxlen=10)

    def draw_results(self, frame, results):
        """Dibuja detecciones con estilo moderno"""
        for bbox, text, conf in results:
            conf_pct = int(conf * 100)
            
            if conf_pct >= 70:
                color = (16, 185, 129)  
                thickness = 3
            elif conf_pct >= 50:
                color = (59, 130, 246)  
                thickness = 2
            else:
                color = (245, 158, 11)  
                thickness = 2
            
            pts = np.array(bbox, dtype=np.int32)

            cv2.polylines(frame, [pts], True, (209, 213, 219), thickness+2)
            cv2.polylines(frame, [pts], True, color, thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            top_left = tuple(pts[0])
            
            (tw, th), bl = cv2.getTextSize(text, font, 0.8, 2)
            ty = max(top_left[1] - 12, th + 22)

            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (top_left[0]-8, ty-th-bl-8),
                         (top_left[0]+tw+8, ty+bl+8),
                         color, -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

            cv2.rectangle(frame,
                         (top_left[0]-8, ty-th-bl-8),
                         (top_left[0]+tw+8, ty+bl+8),
                         color, 2)

            cv2.putText(frame, text, (top_left[0], ty), 
                       font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            badge_text = f"{conf_pct}%"
            (bw, bh), _ = cv2.getTextSize(badge_text, font, 0.4, 1)
            badge_x = top_left[0] + tw - bw + 4
            badge_y = ty + 20
            
            cv2.rectangle(frame,
                         (badge_x-4, badge_y-bh-4),
                         (badge_x+bw+4, badge_y+4),
                         (255, 255, 255), -1)
            cv2.putText(frame, badge_text, (badge_x, badge_y),
                       font, 0.4, color, 1, cv2.LINE_AA)
        
        return frame

    def process_queue(self):
        """Procesa cola de mensajes"""
        processed = 0
        
        try:
            while processed < 5:
                msg = self.app_queue.get_nowait()
                processed += 1
                
                if msg[0] == "video_frame":
                    frame_bgr, results = msg[1], msg[2]
                    
                    frame_with_results = self.draw_results(frame_bgr, results)
                    
                    display_h = int(frame_bgr.shape[0] * DISPLAY_SCALE)
                    display_w = int(frame_bgr.shape[1] * DISPLAY_SCALE)
                    display_frame = cv2.resize(frame_with_results, 
                                              (display_w, display_h),
                                              interpolation=cv2.INTER_LINEAR)
                    
                    img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_fit = ImageOps.fit(img_pil, (800, 450), Image.Resampling.BILINEAR)
                    
                    mask = Image.new('L', (800, 450), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.rounded_rectangle((0, 0, 800, 450), 8, fill=255)
                    img_fit.putalpha(mask)
                    
                    imgtk = ImageTk.PhotoImage(img_fit)
                    self.video_label.config(image=imgtk)
                    self.video_label.image = imgtk
                    
                    current_time = time.time()
                    time_diff = current_time - self.last_frame_time
                    
                    # Solo calcular FPS si hay diferencia de tiempo significativa
                    if time_diff > 0.001:  # Mínimo 1ms de diferencia
                        fps = 1.0 / time_diff
                        self.fps_history.append(fps)
                        self.last_frame_time = current_time
                    
                    # Mostrar FPS promedio si hay datos
                    if len(self.fps_history) > 0:
                        avg_fps = sum(self.fps_history) / len(self.fps_history)
                        self.fps_label.config(text=f"FPS: {int(avg_fps)}")
                    
                elif msg[0] == "status":
                    self.status_label.config(text=f"Estado: {msg[1]}")
                    color = msg[2]
                    self.status_indicator.itemconfig(self.status_dot, fill=color)
                    
                elif msg[0] == "error":
                    self.handle_error(msg[1])

        except queue.Empty:
            pass
        
        self.after(10, self.process_queue)

    def handle_error(self, error_type):
        errors = {
            "camera_error": "Error: No se pudo acceder a la cámara",
            "easyocr_error": "Error: Fallo al cargar EasyOCR",
        }
        msg = errors.get(error_type, f"Error: {error_type}")
        self.video_label.config(text=msg, font=FONT_TITLE, fg=COLOR_ERROR)
        self.status_label.config(text=f"Estado: {msg}")
        self.status_indicator.itemconfig(self.status_dot, fill=COLOR_ERROR)

    def on_closing(self):
        print("Cerrando aplicación...")
        self.recognizer.stop()
        self.destroy()


if __name__ == "__main__":
    app = PlateRecognitionApp()
    app.mainloop()