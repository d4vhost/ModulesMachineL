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

# --- CONFIGURACIÓN DE ESTILO ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
COLOR_SUCCESS = "#34c759"
COLOR_ERROR = "#ff3b30"
COLOR_WARNING = "#ff9500"

FONT_TITLE = ("SF Pro Display", 18, "bold")
FONT_BODY = ("SF Pro Text", 11)
FONT_STATUS = ("SF Pro Text", 14, "bold")

# --- CONFIGURACIÓN OPTIMIZADA PARA FLUIDEZ + DETECCIÓN ---
CAMERA_WIDTH = 1280          # Resolución óptima (no Full HD para velocidad)
CAMERA_HEIGHT = 720
VIDEO_FPS = 30               # FPS objetivo

PROCESS_EVERY_N_FRAMES = 6   # Procesar cada 6 frames (balance perfecto)
PROCESS_SCALE_FACTOR = 0.5   # 50% para OCR (rápido pero efectivo)
DISPLAY_SCALE = 0.8          # Reducir video mostrado al 80% (más fluido)

MIN_CONFIDENCE = 0.25        # Confianza mínima
MIN_PLATE_CHARS = 4          # Mínimo caracteres para placa
MAX_TRACKING_AGE = 1.0       # Segundos sin ver la placa antes de eliminarla
IOU_THRESHOLD = 0.25         # Umbral para asociar detecciones

# --- CLASE OPTIMIZADA DE SEGUIMIENTO ---
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
        # Eliminar placas antiguas
        to_remove = [pid for pid, data in self.tracked_plates.items() 
                     if current_time - data['last_seen'] > MAX_TRACKING_AGE]
        for pid in to_remove:
            del self.tracked_plates[pid]
        
        # Asociar detecciones
        for bbox, text, conf in detections:
            best_match = None
            best_iou = IOU_THRESHOLD
            
            for pid, data in self.tracked_plates.items():
                iou = self._calculate_iou(bbox, data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = pid
            
            if best_match:
                # Actualizar placa existente
                old = self.tracked_plates[best_match]
                # Suavizar confianza
                new_conf = old['confidence'] * 0.6 + conf * 0.4
                # Mantener mejor texto
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
                # Nueva placa
                self.tracked_plates[self.next_id] = {
                    'bbox': bbox,
                    'text': text,
                    'confidence': conf,
                    'best_conf': conf,
                    'last_seen': current_time,
                    'detections': 1
                }
                self.next_id += 1
        
        # Retornar placas activas (solo las que han sido vistas varias veces)
        return [(d['bbox'], d['text'], d['confidence']) 
                for d in self.tracked_plates.values()
                if d['detections'] >= 2]  # Filtro: al menos 2 detecciones
    
    def get_all_active(self, current_time):
        """Obtiene todas las placas activas."""
        return [(d['bbox'], d['text'], d['confidence']) 
                for d in self.tracked_plates.values()
                if current_time - d['last_seen'] < MAX_TRACKING_AGE]


# --- CLASE DEL RECONOCEDOR ---
class LicensePlateRecognizer:
    def __init__(self, app_queue):
        self.app_queue = app_queue
        self.cap = None
        self.running = False
        self.processing = False
        
        self.tracker = PlateTracker()
        self.frame_counter = 0
        self.process_queue = queue.Queue(maxsize=1)
        
        # Inicializar EasyOCR en thread separado
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
                print(f"⚠️ GPU no disponible: {e}")
                self.reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)
                self.reader_ready = True
                print("EasyOCR listo (CPU)")
                self.app_queue.put(("status", "EasyOCR listo (CPU)", COLOR_WARNING))
            except Exception as e2:
                print(f"❌ Error: {e2}")
                self.app_queue.put(("error", "easyocr_error"))

    def start(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.app_queue.put(("error", "camera_error"))
                return
            
            # Configuración optimizada
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            self.running = True
            
            # Threads separados
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
            
            # Obtener placas activas del tracker
            current_time = time.time()
            active_plates = self.tracker.get_all_active(current_time)
            
            # Enviar frame INMEDIATAMENTE (sin esperar OCR)
            self.app_queue.put(("video_frame", frame, active_plates))
            
            # Enviar a procesamiento cada N frames
            if (self.frame_counter % PROCESS_EVERY_N_FRAMES == 0 and 
                self.reader_ready and not self.processing):
                
                # Limpiar cola si tiene frames viejos
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
        """Loop de procesamiento OCR - EN PARALELO."""
        while self.running:
            try:
                frame, timestamp = self.process_queue.get(timeout=1.0)
                
                if not self.reader_ready:
                    continue
                
                self.processing = True
                
                # Reducir frame para OCR (RÁPIDO)
                h, w = frame.shape[:2]
                small_w = int(w * PROCESS_SCALE_FACTOR)
                small_h = int(h * PROCESS_SCALE_FACTOR)
                small_frame = cv2.resize(frame, (small_w, small_h), 
                                        interpolation=cv2.INTER_LINEAR)
                
                # Preprocesamiento RÁPIDO pero efectivo
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # CLAHE para contraste
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                # Sharpen para mejorar bordes (rápido)
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                
                # OCR
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
                
                # Re-escalar y filtrar
                scale_inv = 1.0 / PROCESS_SCALE_FACTOR
                detections = []
                
                for bbox, text, prob in results:
                    if prob >= MIN_CONFIDENCE:
                        text_clean = "".join(text.split()).upper()
                        
                        # Validar
                        if (len(text_clean) >= MIN_PLATE_CHARS and 
                            any(c.isdigit() for c in text_clean)):
                            
                            scaled_bbox = [[int(p[0]*scale_inv), int(p[1]*scale_inv)] 
                                          for p in bbox]
                            detections.append((scaled_bbox, text_clean, prob))
                
                # Actualizar tracker
                if detections:
                    tracked = self.tracker.update(detections, timestamp)
                    if tracked:
                        texts = [t for _, t, _ in tracked]
                        self.app_queue.put(("status", 
                                          f" {len(tracked)}: {', '.join(texts[:2])}", 
                                          COLOR_SUCCESS))
                
                self.processing = False
                
            except queue.Empty:
                self.processing = False
            except Exception as e:
                print(f"❌ Error OCR: {e}")
                self.processing = False
                time.sleep(0.1)


# --- CLASE GUI ---
class PlateRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🚗 Reconocimiento de Placas")
        self.geometry("900x700")
        self.config(bg=COLOR_BG)
        self.resizable(False, False)

        self.app_queue = queue.Queue()
        self.recognizer = LicensePlateRecognizer(self.app_queue)
        
        # Estilos
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, 
                           fieldbackground=COLOR_CARD, borderwidth=0)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Status.TLabel", font=FONT_STATUS, background=COLOR_BG)
        
        # Layout
        self.main_frame = ttk.Frame(self, padding=10)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        ttk.Label(self.main_frame, 
                 text="🚗 Reconocimiento de Placas", 
                 style="Title.TLabel").pack(pady=(10, 5))
        
        # Info técnica
        info = (f"⚡ {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{VIDEO_FPS}fps | "
                f"Proc: 1/{PROCESS_EVERY_N_FRAMES} | Scale: {int(PROCESS_SCALE_FACTOR*100)}%")
        ttk.Label(self.main_frame, text=info, font=("SF Pro Text", 9), 
                 foreground=COLOR_ACCENT).pack(pady=(0, 10))

        # Panel de video
        self.camera_container = ttk.Frame(self.main_frame, style="Card.TFrame", 
                                         width=800, height=450)
        self.camera_container.pack(expand=True, padx=20, pady=10)
        self.camera_container.pack_propagate(False)
        
        self.video_label = tk.Label(self.camera_container, 
                                    text="Iniciando cámara...", 
                                    bg=COLOR_CARD, fg=COLOR_FG, font=FONT_BODY)
        self.video_label.pack(expand=True)
        
        # Status
        self.status_frame = ttk.Frame(self, padding=10)
        self.status_frame.pack(fill="x", pady=10, padx=20, side="bottom")

        self.status_label = ttk.Label(self.status_frame, 
                                     text="Estado: Iniciando...", 
                                     style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)
        
        # FPS counter
        self.fps_label = ttk.Label(self.status_frame, text="FPS: --", 
                                   font=("SF Pro Text", 10), 
                                   foreground=COLOR_ACCENT)
        self.fps_label.pack(side=tk.RIGHT)
        
        self.last_frame_time = time.time()
        self.fps_history = deque(maxlen=10)
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.recognizer.start()
        self.process_queue()

    def draw_results(self, frame, results):
        """Dibuja detecciones"""
        for bbox, text, conf in results:
            # Color por confianza
            conf_pct = int(conf * 100)
            if conf_pct >= 70:
                color = (0, 255, 0)
                thickness = 3
            elif conf_pct >= 50:
                color = (255, 255, 0)
                thickness = 2
            else:
                color = (255, 165, 0)
                thickness = 2
            
            pts = np.array(bbox, dtype=np.int32)
            
            # Polígono
            cv2.polylines(frame, [pts], True, (0,0,0), thickness+2)  # Sombra
            cv2.polylines(frame, [pts], True, color, thickness)
            
            # Texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            top_left = tuple(pts[0])
            
            (tw, th), bl = cv2.getTextSize(text, font, 0.9, 2)
            ty = max(top_left[1] - 10, th + 20)
            
            # Fondo texto
            cv2.rectangle(frame, 
                         (top_left[0]-5, ty-th-bl-5),
                         (top_left[0]+tw+5, ty+bl+5),
                         (0,0,0), -1)
            cv2.rectangle(frame,
                         (top_left[0]-5, ty-th-bl-5),
                         (top_left[0]+tw+5, ty+bl+5),
                         color, 2)
            
            # Texto placa
            cv2.putText(frame, text, (top_left[0], ty), 
                       font, 0.9, color, 2, cv2.LINE_AA)
            
            # Confianza
            cv2.putText(frame, f"{conf_pct}%", (top_left[0], ty+25),
                       font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        return frame

    def process_queue(self):
        """Procesa cola"""
        processed = 0
        
        try:
            while processed < 5:  # Max 5 por ciclo
                msg = self.app_queue.get_nowait()
                processed += 1
                
                if msg[0] == "video_frame":
                    frame_bgr, results = msg[1], msg[2]
                    
                    # Dibujar
                    frame_with_results = self.draw_results(frame_bgr, results)
                    
                    # Reducir para display (MUCHO MÁS FLUIDO)
                    display_h = int(frame_bgr.shape[0] * DISPLAY_SCALE)
                    display_w = int(frame_bgr.shape[1] * DISPLAY_SCALE)
                    display_frame = cv2.resize(frame_with_results, 
                                              (display_w, display_h),
                                              interpolation=cv2.INTER_LINEAR)
                    
                    # Convertir y mostrar
                    img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_fit = ImageOps.fit(img_pil, (800, 450), Image.Resampling.BILINEAR)
                    
                    # Esquinas redondeadas (rápido)
                    mask = Image.new('L', (800, 450), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.rounded_rectangle((0, 0, 800, 450), 12, fill=255)
                    img_fit.putalpha(mask)
                    
                    imgtk = ImageTk.PhotoImage(img_fit)
                    self.video_label.config(image=imgtk)
                    self.video_label.image = imgtk
                    
                    # Calcular FPS
                    current_time = time.time()
                    fps = 1.0 / (current_time - self.last_frame_time)
                    self.last_frame_time = current_time
                    self.fps_history.append(fps)
                    avg_fps = sum(self.fps_history) / len(self.fps_history)
                    self.fps_label.config(text=f"FPS: {int(avg_fps)}")
                    
                elif msg[0] == "status":
                    self.status_label.config(text=f"Estado: {msg[1]}", 
                                           foreground=msg[2])
                    
                elif msg[0] == "error":
                    self.handle_error(msg[1])

        except queue.Empty:
            pass
        
        self.after(10, self.process_queue)  # ~100 FPS GUI

    def handle_error(self, error_type):
        errors = {
            "camera_error": "Error de cámara",
            "easyocr_error": "Error EasyOCR",
        }
        msg = errors.get(error_type, f"Error: {error_type}")
        self.video_label.config(text=msg, font=FONT_TITLE, fg=COLOR_ERROR)
        self.status_label.config(text=f"Estado: {msg}", foreground=COLOR_ERROR)

    def on_closing(self):
        print("Cerrando...")
        self.recognizer.stop()
        self.destroy()


if __name__ == "__main__":
    app = PlateRecognitionApp()
    app.mainloop()