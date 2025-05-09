# -*- coding: utf-8 -*-
import os
# Configurações para otimizar TensorFlow em CPU e suprimir avisos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprimir todas as mensagens exceto erros fatais
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desativar otimizações OneDNN que podem causar conflitos
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Força o uso da CPU mesmo se GPU estiver disponível

import cv2
import numpy as np
import time
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import tensorflow as tf
import math # Adicionado para math.acos
from tensorflow.keras.models import load_model

# Configure TensorFlow to disable GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        # Disable all GPUs
        tf.config.set_visible_devices([], 'GPU')
        print("GPU usage disabled")
    except RuntimeError as e:
        print(f"Error disabling GPU: {e}")

# Disable eager execution warnings
tf.get_logger().setLevel('ERROR')

# --- Configurações Globais e Constantes ---
MP_HOLISTIC = mp.solutions.holistic  # Modelo Holístico
MP_DRAWING = mp.solutions.drawing_utils  # Utilitários de desenho

# Caminho para os dados exportados, arrays numpy
DATA_PATH = os.path.join('Sign_Data_App') # Alterado para evitar conflito com dados existentes

# Ações que tentamos detectar (nomes dos gestos)
# Estes devem ser os mesmos usados para treinar o modelo
ACTIONS = np.array(['Beleza'])
# Ou deixe o usuário definir/carregar dinamicamente se necessário
# ACTIONS = [] # Se for carregar de um arquivo ou entrada do usuário

# Número de sequências (vídeos) por ação
NO_SEQUENCES = 30
# Comprimento da sequência (frames por vídeo)
SEQUENCE_LENGTH = 30

# --- Funções do MediaPipe e Processamento ---

def mediapipe_detection(image, model):
    """
    Realiza a detecção de landmarks com MediaPipe.
    image: frame da câmera (BGR).
    model: instância do modelo Holistic do MediaPipe.
    Retorna: imagem com landmarks (BGR) e os resultados da detecção.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Conversão BGR para RGB
    image_rgb.flags.writeable = False                   # Imagem não é mais gravável (otimização)
    results = model.process(image_rgb)                  # Faz a predição
    image_rgb.flags.writeable = True                    # Imagem é gravável novamente
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Conversão RGB para BGR
    return image_bgr, results

def draw_styled_landmarks(image, results):
    """
    Desenha os landmarks na imagem com estilos personalizados.
    image: frame da câmera onde desenhar.
    results: resultados da detecção do MediaPipe.
    """
    # Conexões da face
    MP_DRAWING.draw_landmarks(image, results.face_landmarks, MP_HOLISTIC.FACEMESH_TESSELATION,
                             MP_DRAWING.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                             MP_DRAWING.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                             )
    # Conexões da pose
    MP_DRAWING.draw_landmarks(image, results.pose_landmarks, MP_HOLISTIC.POSE_CONNECTIONS,
                             MP_DRAWING.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                             MP_DRAWING.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                             )
    # Conexões da mão esquerda
    MP_DRAWING.draw_landmarks(image, results.left_hand_landmarks, MP_HOLISTIC.HAND_CONNECTIONS,
                             MP_DRAWING.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                             MP_DRAWING.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                             )
    # Conexões da mão direita
    MP_DRAWING.draw_landmarks(image, results.right_hand_landmarks, MP_HOLISTIC.HAND_CONNECTIONS,
                             MP_DRAWING.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                             MP_DRAWING.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    """
    Extrai os keypoints dos resultados do MediaPipe e calcula features adicionais.
    results: resultados da detecção do MediaPipe.
    Retorna: array numpy com todos os keypoints concatenados e features.
    """
    keypoints = []

    # Landmarks da pose
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    keypoints.extend(list(pose))

    # Landmarks da face
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    keypoints.extend(list(face))

    # Landmarks da mão esquerda
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    keypoints.extend(list(lh))

    # Landmarks da mão direita
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    keypoints.extend(list(rh))
    
    # Cálculo de features adicionais
    lh_face_dist, rh_face_dist, lh_rh_dist = 0.0, 0.0, 0.0
    finger_tip_diff, lh_finger_face_angle, rh_finger_face_angle = 0.0, 0.0, 0.0

    if results.face_landmarks and results.left_hand_landmarks and results.right_hand_landmarks:
        face_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        lh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        rh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

        face_center = np.mean(face_coords, axis=0)
        lh_center = np.mean(lh_coords, axis=0)
        rh_center = np.mean(rh_coords, axis=0)

        lh_face_dist = float(np.linalg.norm(lh_center - face_center))
        rh_face_dist = float(np.linalg.norm(rh_center - face_center))
        lh_rh_dist = float(np.linalg.norm(lh_center - rh_center))

        lh_finger_tip_idx = 8 
        rh_finger_tip_idx = 8
        
        if len(lh_coords) > lh_finger_tip_idx and len(rh_coords) > rh_finger_tip_idx:
            lh_finger_tip = lh_coords[lh_finger_tip_idx]
            rh_finger_tip = rh_coords[rh_finger_tip_idx]
            finger_tip_diff = float(np.linalg.norm(lh_finger_tip - rh_finger_tip))

            lh_finger_to_face_vec = face_center - lh_finger_tip
            rh_finger_to_face_vec = face_center - rh_finger_tip
            
            ref_vec = np.array([0, 1, 0]) 
            
            norm_lh_vec = np.linalg.norm(lh_finger_to_face_vec)
            norm_rh_vec = np.linalg.norm(rh_finger_to_face_vec)

            if norm_lh_vec > 1e-6:
                 lh_finger_face_angle = float(math.acos(np.clip(np.dot(lh_finger_to_face_vec, ref_vec) / norm_lh_vec, -1.0, 1.0))) # ADIÇÃO: np.clip
            if norm_rh_vec > 1e-6:
                 rh_finger_face_angle = float(math.acos(np.clip(np.dot(rh_finger_to_face_vec, ref_vec) / norm_rh_vec, -1.0, 1.0))) # ADIÇÃO: np.clip
    
    elif results.face_landmarks and results.left_hand_landmarks:
        face_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        lh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        face_center = np.mean(face_coords, axis=0)
        lh_center = np.mean(lh_coords, axis=0)
        lh_face_dist = float(np.linalg.norm(lh_center - face_center))
        # Cálculos para lh_finger_face_angle (se apenas uma mão está presente, o ângulo pode ser 0 ou calculado de forma diferente)
        # Mantendo 0.0 como no original para este caso
        lh_finger_tip_idx = 8
        if len(lh_coords) > lh_finger_tip_idx:
            lh_finger_tip = lh_coords[lh_finger_tip_idx]
            lh_finger_to_face_vec = face_center - lh_finger_tip
            ref_vec = np.array([0, 1, 0])
            norm_lh_vec = np.linalg.norm(lh_finger_to_face_vec)
            if norm_lh_vec > 1e-6:
                lh_finger_face_angle = float(math.acos(np.clip(np.dot(lh_finger_to_face_vec, ref_vec) / norm_lh_vec, -1.0, 1.0)))


    elif results.face_landmarks and results.right_hand_landmarks:
        face_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        rh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        face_center = np.mean(face_coords, axis=0)
        rh_center = np.mean(rh_coords, axis=0)
        rh_face_dist = float(np.linalg.norm(rh_center - face_center))
        # Cálculos para rh_finger_face_angle
        rh_finger_tip_idx = 8
        if len(rh_coords) > rh_finger_tip_idx:
            rh_finger_tip = rh_coords[rh_finger_tip_idx]
            rh_finger_to_face_vec = face_center - rh_finger_tip
            ref_vec = np.array([0, 1, 0])
            norm_rh_vec = np.linalg.norm(rh_finger_to_face_vec)
            if norm_rh_vec > 1e-6:
                rh_finger_face_angle = float(math.acos(np.clip(np.dot(rh_finger_to_face_vec, ref_vec) / norm_rh_vec, -1.0, 1.0)))
        
    additional_features = [float(lh_face_dist), float(rh_face_dist), float(lh_rh_dist), 
                          float(finger_tip_diff), float(lh_finger_face_angle), float(rh_finger_face_angle)]
    keypoints.extend(additional_features)
    
    expected_len = 1668
    current_len = len(keypoints)
    if current_len < expected_len:
        keypoints.extend([0.0] * (expected_len - current_len))
    elif current_len > expected_len:
        keypoints = keypoints[:expected_len]

    return np.array(keypoints)


# --- Classe da Aplicação Tkinter ---
class SignLanguageApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("App de Reconhecimento de sinais")
        self.root.geometry("1000x750")

        self.holistic_model = MP_HOLISTIC.Holistic(min_detection_confidence=0.55, min_tracking_confidence=0.55)
        self.cap = None
        self.current_frame = None
        self.is_running = False
        self.collection_thread = None
        self.inference_thread = None
        self.camera_thread = None # ADIÇÃO: Inicializar camera_thread
        self.stop_event = threading.Event()

        self.keras_model = None
        self.sequence_data = []
        self.sentence_predictions = []
        self.predictions_buffer = []
        self.last_detected_time = 0
        self.pause_between_signs_sec = 1.0
        
        self.detection_active = False
        self.reset_detection_interval = 3
        self.last_reset_time = 0
        
        self.capture_mode = False
        self.capture_countdown = 0
        self.capture_duration = 3
        
        self.continuous_mode = False
        self.gesture_history = []
        
        self.confidence_threshold = 0.7
        self.consistency_threshold = 0.6

        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # ADIÇÃO: Lidar com fechamento da janela

    def setup_gui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabelframe', background='#f0f0f0')
        style.configure('TLabelframe.Label', font=('Arial', 10, 'bold'))
        style.configure('TButton', font=('Arial', 9), background='#e0e0e0') # Cor de botão padrão mais neutra
        style.map('TButton', background=[('active', '#c0c0c0')])
        style.configure('Accent.TButton', font=('Arial', 9, 'bold'), foreground='white', background='#4a86e8')
        style.map('Accent.TButton', background=[('active', '#357ae8')])
        style.configure('TLabel', font=('Arial', 9), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Status.TLabel', font=('Arial', 10), background='#f0f0f0')
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab_camera = ttk.Frame(self.notebook)
        self.tab_collect = ttk.Frame(self.notebook)
        self.tab_inference = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_camera, text="Câmera")
        self.notebook.add(self.tab_collect, text="Coleta de Dados")
        self.notebook.add(self.tab_inference, text="Reconhecimento")
        
        self.setup_camera_tab()
        self.setup_collect_tab()
        self.setup_inference_tab()
        
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Status: Pronto", style='Status.TLabel')
        self.status_label.pack(side=tk.LEFT)
        
        exit_button = ttk.Button(status_frame, text="Sair", command=self.on_closing)
        exit_button.pack(side=tk.RIGHT, padx=5)

    def setup_camera_tab(self):
        camera_control_frame = ttk.LabelFrame(self.tab_camera, text="Controles da Câmera", padding="10")
        camera_control_frame.pack(fill=tk.X, pady=5)
        
        self.btn_start_cam = ttk.Button(camera_control_frame, text="Iniciar Câmera", 
                                        command=self.start_camera, style='Accent.TButton')
        self.btn_start_cam.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_stop_cam = ttk.Button(camera_control_frame, text="Parar Câmera", 
                                       command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop_cam.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(camera_control_frame, text="Dispositivo:").grid(row=0, column=2, padx=5, pady=5)
        self.camera_device_var = tk.StringVar(value="0")
        camera_device_entry = ttk.Entry(camera_control_frame, textvariable=self.camera_device_var, width=5)
        camera_device_entry.grid(row=0, column=3, padx=5, pady=5)
        
        resolution_frame = ttk.Frame(camera_control_frame)
        resolution_frame.grid(row=0, column=4, padx=10, pady=5)
        
        ttk.Label(resolution_frame, text="Resolução:").grid(row=0, column=0)
        self.resolution_var = tk.StringVar(value="640x480")
        resolution_combo = ttk.Combobox(resolution_frame, textvariable=self.resolution_var, 
                                        values=["320x240", "640x480", "800x600", "1280x720"], width=10, state="readonly")
        resolution_combo.grid(row=0, column=1, padx=5)
        resolution_combo.set("640x480") # Definir valor padrão
        
        camera_view_frame = ttk.LabelFrame(self.tab_camera, text="Visualização", padding="10")
        camera_view_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.video_label = ttk.Label(camera_view_frame, text="Câmera Desligada", anchor=tk.CENTER) # anchor=tk.CENTER
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        info_frame = ttk.Frame(self.tab_camera)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.camera_info_label = ttk.Label(info_frame, text="Câmera: Desconectada", style='Status.TLabel')
        self.camera_info_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = ttk.Label(info_frame, text="FPS: --", style='Status.TLabel')
        self.fps_label.pack(side=tk.RIGHT, padx=10)

    def setup_collect_tab(self):
        collect_frame = ttk.LabelFrame(self.tab_collect, text="Configurações de Coleta", padding="10")
        collect_frame.pack(fill=tk.X, pady=5)
        
        action_frame = ttk.Frame(collect_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(action_frame, text="Nome da Ação/Gesto:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.action_name_entry = ttk.Entry(action_frame, width=20)
        self.action_name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.btn_start_collect = ttk.Button(collect_frame, text="Iniciar Coleta para esta Ação", 
                                           command=self.start_data_collection, state=tk.DISABLED, style='Accent.TButton')
        self.btn_start_collect.pack(pady=10)
        
        self.collect_status_label = ttk.Label(self.tab_collect, text="Status da Coleta: Aguardando", font=("Arial", 10))
        self.collect_status_label.pack(pady=10)

    def setup_inference_tab(self):
        main_frame = ttk.Frame(self.tab_inference)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_column = ttk.Frame(main_frame)
        control_column.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10)) # Adicionado padx
        
        model_frame = ttk.LabelFrame(control_column, text="Modelo", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        self.btn_load_model = ttk.Button(model_frame, text="Carregar Modelo (.keras)", 
                                         command=self.load_keras_model_dialog, style='Accent.TButton')
        self.btn_load_model.pack(fill=tk.X, pady=5)
        
        self.model_status_label = ttk.Label(model_frame, text="Modelo: Não Carregado", style='Status.TLabel', wraplength=200) # wraplength
        self.model_status_label.pack(fill=tk.X, pady=5)
        
        classes_frame = ttk.LabelFrame(control_column, text="Classes Reconhecíveis", padding="10")
        classes_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        classes_scroll = ttk.Scrollbar(classes_frame)
        classes_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.classes_listbox = tk.Listbox(classes_frame, height=6, 
                                          yscrollcommand=classes_scroll.set,
                                          font=('Arial', 10),
                                          selectbackground='#4a86e8',
                                          exportselection=False) # exportselection=False
        self.classes_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        classes_scroll.config(command=self.classes_listbox.yview)
        
        control_frame = ttk.LabelFrame(control_column, text="Controle de Reconhecimento", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        self.btn_start_inference = ttk.Button(control_frame, text="Iniciar Reconhecimento", 
                                             command=self.start_inference_thread, style='Accent.TButton', state=tk.DISABLED) # state=tk.DISABLED
        self.btn_start_inference.pack(fill=tk.X, pady=5)
        
        self.btn_capture_gesture = ttk.Button(control_frame, text="Capturar Gesto", 
                                             command=self.capture_gesture, state=tk.DISABLED)
        self.btn_capture_gesture.pack(fill=tk.X, pady=5)
        
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.continuous_mode_var = tk.BooleanVar(value=False)
        self.chk_continuous_mode = ttk.Checkbutton(
            mode_frame, 
            text="Modo Contínuo", # Texto mais curto
            variable=self.continuous_mode_var,
            command=self.toggle_continuous_mode,
            state=tk.DISABLED
        )
        self.chk_continuous_mode.pack(side=tk.LEFT)
        
        settings_frame = ttk.LabelFrame(control_column, text="Ajustes de Sensibilidade", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)
        
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill=tk.X, pady=3)
        ttk.Label(conf_frame, text="Confiança:").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.7)
        self.confidence_label = ttk.Label(conf_frame, text="0.70", width=5, anchor=tk.E) # anchor
        self.confidence_label.pack(side=tk.RIGHT)
        self.confidence_scale = ttk.Scale(settings_frame, from_=0.5, to=0.95, variable=self.confidence_var, orient=tk.HORIZONTAL) # to=0.95
        self.confidence_scale.pack(fill=tk.X, pady=(0,5))
        self.confidence_var.trace_add("write", self.update_confidence_label)
        
        cons_frame = ttk.Frame(settings_frame)
        cons_frame.pack(fill=tk.X, pady=3)
        ttk.Label(cons_frame, text="Consistência:").pack(side=tk.LEFT)
        self.consistency_var = tk.DoubleVar(value=0.6)
        self.consistency_label = ttk.Label(cons_frame, text="0.60", width=5, anchor=tk.E) # anchor
        self.consistency_label.pack(side=tk.RIGHT)
        self.consistency_scale = ttk.Scale(settings_frame, from_=0.4, to=0.9, variable=self.consistency_var, orient=tk.HORIZONTAL) # to=0.9
        self.consistency_scale.pack(fill=tk.X, pady=(0,5))
        self.consistency_var.trace_add("write", self.update_consistency_label)
        
        pause_frame = ttk.Frame(settings_frame)
        pause_frame.pack(fill=tk.X, pady=3)
        ttk.Label(pause_frame, text="Pausa (s):").pack(side=tk.LEFT)
        self.pause_var = tk.DoubleVar(value=1.0)
        self.pause_label = ttk.Label(pause_frame, text="1.00", width=5, anchor=tk.E) # anchor
        self.pause_label.pack(side=tk.RIGHT)
        self.pause_scale = ttk.Scale(settings_frame, from_=0.5, to=3.0, variable=self.pause_var, orient=tk.HORIZONTAL)
        self.pause_scale.pack(fill=tk.X)
        self.pause_var.trace_add("write", self.update_pause_label)
        
        results_column = ttk.Frame(main_frame)
        results_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        view_frame = ttk.LabelFrame(results_column, text="Visualização em Tempo Real", padding="10")
        view_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.inference_video_label = ttk.Label(view_frame, text="Câmera Desligada", anchor=tk.CENTER) # anchor
        self.inference_video_label.pack(expand=True, fill=tk.BOTH)
        
        results_frame = ttk.LabelFrame(results_column, text="Resultados", padding="10")
        results_frame.pack(fill=tk.X, pady=5)
        
        self.detected_action_label = ttk.Label(results_frame, text="Ação Detectada: --", 
                                              font=("Arial", 16, "bold"), anchor=tk.W) # anchor
        self.detected_action_label.pack(fill=tk.X, pady=5)
        
        self.sentence_label = ttk.Label(results_frame, text="Frase: ", 
                                       font=("Arial", 12), anchor=tk.W, wraplength=400) # anchor, wraplength
        self.sentence_label.pack(fill=tk.X, pady=5)
        
        history_frame = ttk.LabelFrame(results_column, text="Histórico de Gestos", padding="10")
        history_frame.pack(fill=tk.X, pady=5)
        
        history_scroll = ttk.Scrollbar(history_frame)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(history_frame, height=5, 
                                         yscrollcommand=history_scroll.set,
                                         font=('Arial', 9),
                                         exportselection=False) # exportselection=False
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.config(command=self.history_listbox.yview)
        
        clear_btn = ttk.Button(history_frame, text="Limpar Histórico", 
                              command=lambda: (self.history_listbox.delete(0, tk.END), setattr(self, 'gesture_history', []))) # Limpa self.gesture_history também
        clear_btn.pack(side=tk.BOTTOM, pady=5, fill=tk.X) # fill=tk.X
        
    def update_status(self, message):
        if self.root.winfo_exists(): # Evita erro se a janela já foi destruída
            self.status_label.config(text=f"Status: {message}")

    def update_collect_status(self, message):
        if self.root.winfo_exists():
            self.collect_status_label.config(text=f"Status da Coleta: {message}")

    def update_detected_action(self, action_text):
        if self.root.winfo_exists():
            self.detected_action_label.config(text=f"Ação Detectada: {action_text}")
        
    def update_sentence_display(self):
        if self.root.winfo_exists():
            self.sentence_label.config(text=f"Frase: {' '.join(self.sentence_predictions)}")

    def start_camera(self):
        if self.is_running:
            return
            
        try:
            device_id_str = self.camera_device_var.get()
            if not device_id_str.isdigit():
                messagebox.showerror("Erro", "ID do dispositivo da câmera deve ser um número.")
                return
            device_id = int(device_id_str)
            
            resolution_str = self.resolution_var.get()
            if 'x' not in resolution_str:
                messagebox.showerror("Erro", "Formato de resolução inválido. Use LarguraxAltura (ex: 640x480).")
                return
            resolution = resolution_str.split('x')
            width, height = int(resolution[0]), int(resolution[1])
            
            # Inicializa a câmera sem o parâmetro específico do Windows
            self.cap = cv2.VideoCapture(device_id)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if not self.cap.isOpened():
                # Tenta com o backend v4l2 (Video for Linux 2) que é comum em sistemas Linux
                self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                if not self.cap.isOpened():
                    messagebox.showerror("Erro", f"Não foi possível abrir a câmera {device_id}. Verifique se está conectada e não em uso.")
                    self.cap = None # Garante que self.cap é None se falhar
                    return
            
            # Tenta garantir que a câmera está realmente funcionando lendo um frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                messagebox.showerror("Erro", f"A câmera {device_id} foi aberta, mas não está retornando imagens. Verifique as permissões ou tente outro dispositivo.")
                self.cap.release()
                self.cap = None
                return
                
            self.is_running = True
            self.btn_start_cam.config(state=tk.DISABLED)
            self.btn_stop_cam.config(state=tk.NORMAL)
            self.btn_start_collect.config(state=tk.NORMAL)
            if self.keras_model:
                self.btn_start_inference.config(state=tk.NORMAL)
                self.chk_continuous_mode.config(state=tk.NORMAL) # Habilita modo contínuo
                self.btn_capture_gesture.config(state=tk.NORMAL if not self.continuous_mode_var.get() else tk.DISABLED)

            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.camera_info_label.config(text=f"Câmera: {device_id} ({int(actual_width)}x{int(actual_height)})")
            
            self.stop_event.clear()
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.update_status("Câmera iniciada.")
            
        except ValueError:
             messagebox.showerror("Erro", "ID da câmera ou resolução inválida. Verifique os valores.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar a câmera: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_running = False # Garante que o estado é consistente

    def camera_loop(self):
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while self.is_running and not self.stop_event.is_set():
                if not self.cap or not self.cap.isOpened():
                    self.update_status("Câmera desconectada ou com erro.")
                    time.sleep(0.5)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.update_status("Erro ao ler frame da câmera.")
                    time.sleep(0.1)
                    continue
                    
                self.current_frame = frame.copy()
                
                active_collection = self.collection_thread and self.collection_thread.is_alive()
                active_inference = self.inference_thread and self.inference_thread.is_alive()
                
                if not active_collection and not active_inference:
                    image_processed, results = mediapipe_detection(frame, self.holistic_model)
                    draw_styled_landmarks(image_processed, results)
                    
                    cv2.putText(image_processed, f"FPS: {current_fps}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    self.show_frame_on_gui(image_processed) # Mostra na aba da câmera
                    if self.notebook.index(self.notebook.select()) == 2: # Se aba de inferência estiver ativa e sem inferência rodando
                        self.show_inference_frame(image_processed) # Mantém a visualização lá também
                
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = int(fps_counter / (time.time() - fps_start_time))
                    if self.root.winfo_exists():
                        self.fps_label.config(text=f"FPS: {current_fps}")
                    fps_counter = 0
                    fps_start_time = time.time()
                
                time.sleep(0.01) # Pequena pausa para não sobrecarregar a CPU
                
        except Exception as e:
            # Não usar messagebox aqui para não travar a thread, apenas logar
            print(f"Erro no loop da câmera: {e}")
            self.update_status(f"Erro no loop da câmera: {e}")
        finally:
            if self.is_running: # Se o loop terminar inesperadamente mas is_running ainda for True
                 if self.root.winfo_exists():
                    self.root.after(0, self.stop_camera)
                
    def stop_camera(self):
        self.update_status("Parando câmera...")
        self.stop_event.set()
        self.is_running = False
        
        threads_to_join = []
        if self.collection_thread and self.collection_thread.is_alive():
            threads_to_join.append(self.collection_thread)
        if self.inference_thread and self.inference_thread.is_alive():
            threads_to_join.append(self.inference_thread)
        if self.camera_thread and self.camera_thread.is_alive():
            threads_to_join.append(self.camera_thread)

        for thread in threads_to_join:
            try:
                thread.join(timeout=1.5) # Aumentado timeout ligeiramente
            except Exception as e:
                print(f"Erro ao aguardar thread {thread.name}: {e}")

        if self.cap:
            self.cap.release()
            self.cap = None
            
        if self.root.winfo_exists():
            self.current_frame = None
            # Limpa os labels de vídeo
            for label in [self.video_label, self.inference_video_label]:
                if label.winfo_exists():
                    label.config(image='', text="Câmera Desligada")
                    label.imgtk = None # Remove referência

            self.camera_info_label.config(text="Câmera: Desconectada")
            self.fps_label.config(text="FPS: --")
            
            self.btn_start_cam.config(state=tk.NORMAL)
            self.btn_stop_cam.config(state=tk.DISABLED)
            self.btn_start_collect.config(state=tk.DISABLED)
            self.btn_start_inference.config(state=tk.DISABLED)
            self.btn_capture_gesture.config(state=tk.DISABLED)
            self.chk_continuous_mode.config(state=tk.DISABLED)
        
        self.update_status("Câmera parada.") # Mensagem mais concisa

    def load_keras_model_dialog(self):
        filepath = filedialog.askopenfilename(
            title="Selecionar Modelo Keras",
            filetypes=(("Modelos Keras", "*.keras *.h5"), ("Todos os arquivos", "*.*"))
        )
        if filepath:
            try:
                self.update_status("Carregando modelo...")
                self.root.update_idletasks() # Força atualização da GUI
                
                # tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() or 1) # Usa todos os cores
                # tf.config.threading.set_inter_op_parallelism_threads(max(2, (os.cpu_count() or 1)//2))

                self.keras_model = load_model(filepath, compile=True) # Tentar com compile=True primeiro
                # Se der erro, pode ser necessário compile=False e depois compilar manualmente se o otimizador não for padrão.
                # self.keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Se necessário
                
                model_name = os.path.basename(filepath)
                self.model_status_label.config(text=f"Modelo: {model_name}")
                self.update_status(f"Modelo '{model_name}' carregado.")
                
                if self.is_running:
                    self.btn_start_inference.config(state=tk.NORMAL)
                    self.chk_continuous_mode.config(state=tk.NORMAL) # Habilita modo contínuo
                    self.btn_capture_gesture.config(state=tk.NORMAL if not self.continuous_mode_var.get() else tk.DISABLED)

                self.classes_listbox.delete(0, tk.END)
                for i, action in enumerate(ACTIONS): # Certifique-se que ACTIONS corresponde ao modelo
                    self.classes_listbox.insert(tk.END, f"{i+1}. {action}")
                if ACTIONS.size > 0:
                    self.classes_listbox.selection_set(0)
                    
                # Tentar obter input/output shape para info
                try:
                    input_shape = self.keras_model.input_shape
                    output_shape = self.keras_model.output_shape
                    info_msg = (f"Modelo carregado: {model_name}\n"
                                f"Input shape: {input_shape}\n"
                                f"Output shape: {output_shape}\n"
                                f"Classes (conforme config): {len(ACTIONS)} - {', '.join(ACTIONS)}")
                except Exception:
                    info_msg = (f"Modelo carregado: {model_name}\n"
                                f"Classes (conforme config): {len(ACTIONS)} - {', '.join(ACTIONS)}")

                messagebox.showinfo("Informações do Modelo", info_msg)
                
            except Exception as e:
                messagebox.showerror("Erro ao Carregar Modelo", f"Não foi possível carregar o modelo: {str(e)}")
                self.keras_model = None
                self.model_status_label.config(text="Modelo: Falha ao carregar")
                self.update_status(f"Erro ao carregar modelo.")

    def show_frame_on_gui(self, frame_to_show):
        if frame_to_show is None or not self.root.winfo_exists() or not self.video_label.winfo_exists():
            return
        try:
            cv2image = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            
            # Redimensionar se necessário para caber no label, mantendo a proporção
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            if label_width > 1 and label_height > 1: # Evitar divisão por zero se o label não estiver visível
                img_width, img_height = img.size
                scale = min(label_width / img_width, label_height / img_height)
                if scale < 1: # Só redimensiona se a imagem for maior que o label
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk, text="")
        except Exception as e:
            # print(f"Erro ao mostrar frame na GUI (aba câmera): {e}")
            pass # Silenciar erros frequentes se o widget estiver sendo destruído

    def start_inference_thread(self):
        if not self.is_running or not self.cap:
            messagebox.showerror("Erro", "A câmera não está iniciada.")
            return
        if not self.keras_model:
            messagebox.showerror("Erro", "Nenhum modelo Keras carregado.")
            return

        self.update_status("Iniciando reconhecimento...")
        
        self.btn_start_inference.config(state=tk.DISABLED)
        self.btn_start_collect.config(state=tk.DISABLED) # Não coletar durante inferência
        
        self.btn_capture_gesture.config(state=tk.NORMAL if not self.continuous_mode_var.get() else tk.DISABLED)
        self.chk_continuous_mode.config(state=tk.NORMAL)
        
        self.stop_event.clear()
        self.sequence_data = []
        self.sentence_predictions = []
        self.predictions_buffer = []
        # self.gesture_history = [] # Não limpar histórico aqui, limpar com botão
        self.last_detected_time = time.time() # Resetar timer
        
        self.update_sentence_display()
        self.update_detected_action("--")
        
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()
        
        self.update_status("Reconhecimento iniciado.")
    
    def inference_loop(self):
        min_sequence_for_pred = 10
        
        self.last_reset_time = time.time()
        self.detection_active = self.continuous_mode_var.get()
        self.capture_mode = False
        self.capture_countdown = 0
        self.last_countdown_update = 0
        
        frame_skip_counter = 0 
        max_frame_skip = 1 
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        last_processed_frame_for_skip = None

        try:
            while not self.stop_event.is_set():
                if not self.cap or not self.cap.isOpened():
                    self.update_status("Câmera desconectada durante inferência.")
                    time.sleep(0.5)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.update_status("Falha ao capturar frame para inferência.")
                    time.sleep(0.1)
                    continue

                image_processed = frame.copy() # Começa com o frame original

                frame_skip_counter = (frame_skip_counter + 1) % (max_frame_skip + 1)
                if frame_skip_counter != 0 and len(self.sequence_data) > 0: # Condição para pular
                    if last_processed_frame_for_skip is not None:
                        self.show_inference_frame(last_processed_frame_for_skip)
                    else: # Se não houver frame processado anterior, mostrar o atual bruto
                        self.show_inference_frame(image_processed)
                    time.sleep(0.01)
                    continue
                
                image_processed, results = mediapipe_detection(image_processed, self.holistic_model) # Processa o frame atual
                draw_styled_landmarks(image_processed, results)
                last_processed_frame_for_skip = image_processed.copy() # Guarda para o skip
                
                current_time = time.time()
                
                if self.capture_mode:
                    if current_time - self.last_countdown_update >= 1.0:
                        self.capture_countdown -= 1
                        self.last_countdown_update = current_time
                        
                        if self.capture_countdown > 0:
                            self.update_status(f"Capturando em {self.capture_countdown}s...")
                        else:
                            self.update_status("Capturando gesto...")
                            self.detection_active = True
                            self.sequence_data = []
                            self.predictions_buffer = []
                            
                    if self.capture_countdown > 0:
                        cv2.putText(image_processed, f"INICIA EM: {self.capture_countdown}", (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                    else: # Durante a captura ativa
                        cv2.putText(image_processed, "CAPTURANDO!", (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                        if len(self.sequence_data) >= SEQUENCE_LENGTH:
                            self.capture_mode = False
                            self.update_status("Gesto capturado. Analisando...")
                            # A detecção permanece ativa para processar a sequência coletada
                
                elif self.continuous_mode: # ALTERAÇÃO: elif para não conflitar com capture_mode
                    time_since_last_detection = current_time - self.last_detected_time
                    if not self.detection_active and time_since_last_detection >= self.pause_between_signs_sec:
                        self.detection_active = True
                        self.sequence_data = []
                        self.predictions_buffer = []
                        self.update_status("Detecção contínua reativada.")
                
                keypoints = extract_keypoints(results)
                
                if self.detection_active or (self.capture_mode and self.capture_countdown <=0) : # Coleta se detectando ou capturando (pós contagem)
                    self.sequence_data.append(keypoints)
                    self.sequence_data = self.sequence_data[-SEQUENCE_LENGTH:]
                
                predicted_action_text = "--"
                current_confidence = 0.0
                
                if len(self.sequence_data) == SEQUENCE_LENGTH and self.detection_active:
                    try:
                        res = self.keras_model.predict(np.expand_dims(self.sequence_data, axis=0), verbose=0)[0]
                        prediction_index = np.argmax(res)
                        current_confidence = res[prediction_index]
                        
                        self.predictions_buffer.append(prediction_index)
                        self.predictions_buffer = self.predictions_buffer[-min_sequence_for_pred:]
                        
                        buffer_consistency_threshold = int(min_sequence_for_pred * self.consistency_threshold)
                        
                        if current_confidence > self.confidence_threshold:
                            if len(self.predictions_buffer) >= buffer_consistency_threshold: # Precisa de um buffer mínimo para ter consistência
                                from collections import Counter
                                counter = Counter(self.predictions_buffer)
                                most_common_pred_tuple = counter.most_common(1)
                                if most_common_pred_tuple: # Se houver algo no buffer
                                    most_common_pred, count = most_common_pred_tuple[0]
                                    if count >= buffer_consistency_threshold:
                                        current_predicted_action = ACTIONS[most_common_pred]
                                        predicted_action_text = f"{current_predicted_action} ({current_confidence:.2f})" # Adiciona confiança
                                        
                                        is_new_detection = (not self.sentence_predictions or 
                                                           self.sentence_predictions[-1] != current_predicted_action)
                                        
                                        if is_new_detection:
                                            self.sentence_predictions.append(current_predicted_action)
                                            self.sentence_predictions = self.sentence_predictions[-5:]
                                            
                                            timestamp = time.strftime("%H:%M:%S")
                                            history_entry = f"{timestamp} - {current_predicted_action}"
                                            self.gesture_history.append(history_entry)
                                            
                                            if self.root.winfo_exists():
                                                self.root.after(0, self.update_history_listbox, history_entry)
                                                self.root.after(0, self.update_sentence_display)
                                            
                                            self.last_detected_time = current_time
                                            
                                            if not self.continuous_mode: # Modo manual
                                                self.detection_active = False # Para de detectar após uma captura
                                                if self.root.winfo_exists():
                                                    self.root.after(0, lambda: self.btn_capture_gesture.config(state=tk.NORMAL))
                                                self.update_status("Gesto reconhecido. Clique 'Capturar Gesto' novamente.")
                                            else: # Modo contínuo
                                                self.detection_active = False # Inicia pausa
                                                self.update_status(f"'{current_predicted_action}' detectado. Pausando...")
                                            
                                            self.predictions_buffer = []
                                            self.sequence_data = [] # Limpa sequência para não redetectar o mesmo gesto imediatamente
                    except Exception as e:
                        print(f"Erro durante predição: {e}")
                        predicted_action_text = "Erro na predição"

                if self.root.winfo_exists():
                    self.root.after(0, self.update_detected_action, predicted_action_text.split(' (')[0]) # Mostra só o nome

                # Overlay de informações no frame
                y_offset = 30
                cv2.putText(image_processed, ' '.join(self.sentence_predictions), (10, image_processed.shape[0] - y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                y_offset += 30

                if predicted_action_text != "--" and "Erro" not in predicted_action_text:
                    cv2.putText(image_processed, f"Detectado: {predicted_action_text}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    status_msg = ""
                    if self.continuous_mode:
                        status_msg = "Modo Contínuo"
                        if not self.detection_active: status_msg += f" (Pausado {self.pause_between_signs_sec:.1f}s)"
                    elif self.capture_mode and self.capture_countdown <=0:
                        status_msg = "Capturando..."
                    elif not self.detection_active :
                         status_msg = "Aguardando Captura/Modo Contínuo"
                    
                    if status_msg:
                         cv2.putText(image_processed, status_msg, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2, cv2.LINE_AA)
                
                # Info no canto superior direito
                info_y = 30
                cv2.putText(image_processed, f"FPS: {current_fps}", (image_processed.shape[1] - 150, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                info_y += 20
                cv2.putText(image_processed, "MODO: " + ("CONT" if self.continuous_mode else "MANUAL"), (image_processed.shape[1] - 150, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                info_y += 20
                cv2.putText(image_processed, f"Conf: {self.confidence_threshold:.2f}", (image_processed.shape[1] - 150, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

                self.show_inference_frame(image_processed)
                
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = int(fps_counter / (time.time() - fps_start_time))
                    fps_counter = 0
                    fps_start_time = time.time()
                
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Erro fatal na inferência: {e}") # Log para console
            self.update_status(f"Erro na inferência: {e}")
            # Evitar messagebox de thread para não travar
        finally:
            if self.root.winfo_exists():
                self.root.after(0, self._on_inference_end)
            
    def show_inference_frame(self, frame_to_show):
        if frame_to_show is None or not self.root.winfo_exists() or not self.inference_video_label.winfo_exists():
            return
        try:
            cv2image = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)

            label_width = self.inference_video_label.winfo_width()
            label_height = self.inference_video_label.winfo_height()
            if label_width > 1 and label_height > 1:
                img_width, img_height = img.size
                scale = min(label_width / img_width, label_height / img_height)
                if scale < 1:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS) # Usar LANCZOS para melhor qualidade

            imgtk = ImageTk.PhotoImage(image=img)
            
            self.inference_video_label.imgtk = imgtk
            self.inference_video_label.config(image=imgtk, text="")
            
            # Atualiza o label da aba da câmera também se a aba de inferência estiver ativa
            if self.notebook.index(self.notebook.select()) == 2: # 2 é o índice da aba de inferência
                 if self.video_label.winfo_exists():
                    self.video_label.imgtk = imgtk
                    self.video_label.config(image=imgtk, text="")
            
        except Exception as e:
            # print(f"Erro ao mostrar frame na GUI (inferência): {e}")
            pass

    def capture_gesture(self):
        if not self.is_running or not self.keras_model or self.continuous_mode: # Não capturar se em modo contínuo
            return
            
        self.sequence_data = []
        self.predictions_buffer = []
        
        self.capture_mode = True # Ativa o modo de captura
        self.detection_active = False # Desativa detecção até contagem terminar
        self.capture_countdown = self.capture_duration 
        self.last_countdown_update = time.time()
        
        self.update_status(f"Preparando captura em {self.capture_countdown}s...")
        self.btn_capture_gesture.config(state=tk.DISABLED)
        
    def _on_inference_end(self):
        if not self.root.winfo_exists(): return

        is_cam_running_and_model_loaded = self.is_running and self.keras_model
        self.btn_start_inference.config(state=tk.NORMAL if is_cam_running_and_model_loaded else tk.DISABLED)
        
        # Só reabilita capture_gesture se não estiver em modo contínuo
        can_capture = is_cam_running_and_model_loaded and not self.continuous_mode_var.get()
        self.btn_capture_gesture.config(state=tk.NORMAL if can_capture else tk.DISABLED)
        self.chk_continuous_mode.config(state=tk.NORMAL if is_cam_running_and_model_loaded else tk.DISABLED)
        
        if self.is_running:
            self.btn_start_collect.config(state=tk.NORMAL)

        final_status = "Reconhecimento parado."
        if self.stop_event.is_set() and not self.is_running : # Se foi parado globalmente (on_closing)
             final_status = "Aplicação encerrando."
        elif self.stop_event.is_set(): # Se foi parado especificamente para inferência
            final_status = "Reconhecimento interrompido."
        
        self.update_status(final_status)
        self.update_detected_action("--")
        
    def update_confidence_label(self, *args):
        value = self.confidence_var.get()
        if self.root.winfo_exists(): self.confidence_label.config(text=f"{value:.2f}")
        self.confidence_threshold = value
        
    def update_consistency_label(self, *args):
        value = self.consistency_var.get()
        if self.root.winfo_exists(): self.consistency_label.config(text=f"{value:.2f}")
        self.consistency_threshold = value
        
    def update_pause_label(self, *args):
        value = self.pause_var.get()
        if self.root.winfo_exists(): self.pause_label.config(text=f"{value:.2f}")
        self.pause_between_signs_sec = value
        
    def toggle_continuous_mode(self):
        self.continuous_mode = self.continuous_mode_var.get()
        
        if self.continuous_mode:
            self.detection_active = True # Começa a detectar imediatamente
            self.capture_mode = False # Desliga modo de captura manual
            self.update_status("Modo contínuo ativado.")
            self.btn_capture_gesture.config(state=tk.DISABLED)
            # Limpar sequência e buffer para nova detecção contínua
            self.sequence_data = []
            self.predictions_buffer = []
            self.last_detected_time = time.time() # Reset timer para evitar pausa imediata
        else:
            self.detection_active = False
            self.update_status("Modo manual ativado.")
            if self.is_running and self.keras_model: # Só habilita se relevante
                self.btn_capture_gesture.config(state=tk.NORMAL)
            
    def update_history_listbox(self, history_entry):
        if self.root.winfo_exists() and self.history_listbox.winfo_exists():
            self.history_listbox.insert(tk.END, history_entry)
            self.history_listbox.see(tk.END)
        
    def on_closing(self):
        if messagebox.askokcancel("Sair", "Você tem certeza que quer sair?"):
            self.update_status("Encerrando aplicação...")
            self.stop_event.set()
            self.is_running = False # Sinaliza para todos os loops pararem
            
            # Garante que stop_camera seja chamado para limpar threads da câmera
            # e liberar o dispositivo self.cap ANTES de holistic_model.close()
            if self.cap and self.cap.isOpened():
                 # A chamada a stop_camera já faz join nas threads
                 # e libera self.cap
                 # Não precisa chamar self.stop_camera() diretamente se o fechamento da janela
                 # já está sendo tratado e ela faz o join.
                 # Apenas certificar que as threads terminem.
                 pass

            threads_to_close = []
            if self.camera_thread and self.camera_thread.is_alive(): threads_to_close.append(self.camera_thread)
            if self.collection_thread and self.collection_thread.is_alive(): threads_to_close.append(self.collection_thread)
            if self.inference_thread and self.inference_thread.is_alive(): threads_to_close.append(self.inference_thread)

            for t in threads_to_close:
                try:
                    t.join(timeout=2.0)
                except Exception as e:
                    print(f"Erro ao fechar thread {t.name}: {e}")

            if self.cap: # Libera a câmera se ainda estiver aberta
                self.cap.release()
                self.cap = None
            
            if self.holistic_model:
                self.holistic_model.close() # ADIÇÃO: Liberar modelo MediaPipe
                self.holistic_model = None

            if self.root.winfo_exists():
                self.root.destroy()

    def start_data_collection(self):
        action_name = self.action_name_entry.get().strip()
        if not action_name:
            messagebox.showerror("Erro", "Por favor, insira um nome para a ação/gesto.")
            return
        if not self.is_running or not self.cap:
            messagebox.showerror("Erro", "A câmera não está iniciada.")
            return

        self.update_status(f"Iniciando coleta para: {action_name}")
        self.update_collect_status(f"Preparando coleta para '{action_name}'...")
        
        self.btn_start_collect.config(state=tk.DISABLED)
        self.btn_start_inference.config(state=tk.DISABLED) # Não inferir durante coleta
        self.chk_continuous_mode.config(state=tk.DISABLED)
        self.btn_capture_gesture.config(state=tk.DISABLED)

        self.stop_event.clear()

        self.collection_thread = threading.Thread(target=self.data_collection_loop, args=(action_name,), daemon=True)
        self.collection_thread.start()
        
    def data_collection_loop(self, action_name):
        try:
            action_path = os.path.join(DATA_PATH, action_name)
            os.makedirs(action_path, exist_ok=True) # Cria diretório da ação
            if self.root.winfo_exists(): self.update_collect_status(f"Diretório criado para '{action_name}'")

            for sequence in range(NO_SEQUENCES):
                if self.stop_event.is_set(): break
                
                sequence_path = os.path.join(action_path, str(sequence))
                os.makedirs(sequence_path, exist_ok=True) # ALTERAÇÃO: Mover para cá

                if self.root.winfo_exists(): self.update_collect_status(f"'{action_name}', Seq {sequence + 1}/{NO_SEQUENCES}. Prepare-se...")
                
                # Contagem regressiva
                countdown_duration = 3 # Segundos
                for i in range(countdown_duration, 0, -1):
                    if self.stop_event.is_set(): return
                    if not self.cap or not self.cap.isOpened(): return # Checa câmera
                    ret, frame = self.cap.read()
                    if not ret or frame is None: continue
                    
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"INICIANDO EM {i}...", (100, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(display_frame, f"Coleta: {action_name}, Seq: {sequence+1}", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    self.show_frame_on_gui(display_frame)
                    time.sleep(1)

                # Coleta de frames
                for frame_num in range(SEQUENCE_LENGTH):
                    if self.stop_event.is_set(): break
                    if not self.cap or not self.cap.isOpened(): return
                        
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        self.update_status("Falha ao capturar frame. Coleta interrompida.")
                        return

                    image_processed, results = mediapipe_detection(frame, self.holistic_model)
                    draw_styled_landmarks(image_processed, results)

                    if self.root.winfo_exists(): self.update_collect_status(f"'{action_name}', Seq {sequence + 1}, Frame {frame_num + 1}")
                    cv2.putText(image_processed, f"GRAVANDO: {action_name} - Seq {sequence+1} Frame {frame_num+1}",
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    self.show_frame_on_gui(image_processed)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(sequence_path, str(frame_num)) # Salva dentro da pasta da sequência
                    np.save(npy_path, keypoints)
                    
                    time.sleep(0.05) 

                if self.stop_event.is_set(): break
                if self.root.winfo_exists(): self.update_collect_status(f"Seq {sequence+1} de '{action_name}' coletada. Pausa...")
                time.sleep(1)

            if not self.stop_event.is_set():
                if self.root.winfo_exists(): self.update_collect_status(f"Coleta para '{action_name}' concluída!")
                messagebox.showinfo("Coleta Concluída", f"Todos os dados para '{action_name}' foram coletados.")
            else:
                if self.root.winfo_exists(): self.update_collect_status(f"Coleta para '{action_name}' interrompida.")

        except Exception as e:
            if self.root.winfo_exists(): self.update_status(f"Erro na coleta: {e}")
            messagebox.showerror("Erro de Coleta", f"Ocorreu um erro: {e}")
        finally:
            if self.root.winfo_exists():
                self.root.after(0, self._on_collection_end)
            
    def _on_collection_end(self):
        if not self.root.winfo_exists(): return

        self.btn_start_collect.config(state=tk.NORMAL if self.is_running else tk.DISABLED)
        is_cam_running_and_model_loaded = self.is_running and self.keras_model
        if is_cam_running_and_model_loaded:
             self.btn_start_inference.config(state=tk.NORMAL)
             self.chk_continuous_mode.config(state=tk.NORMAL)
             self.btn_capture_gesture.config(state=tk.NORMAL if not self.continuous_mode_var.get() else tk.DISABLED)

        final_status = "Coleta finalizada."
        if self.stop_event.is_set():
            final_status = "Coleta interrompida."
        self.update_status(final_status)

# --- Função Principal ---
if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)

    main_window = tk.Tk()
    app = SignLanguageApp(main_window)
    main_window.mainloop()