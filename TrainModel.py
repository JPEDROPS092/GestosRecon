# -*- coding: utf-8 -*-
import os
# Configurações para otimizar TensorFlow em CPU e suprimir avisos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprimir mensagens de log do TensorFlow (0=all, 1=INFO, 2=WARNING, 3=ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Otimizações OneDNN para CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Força o uso da CPU mesmo se GPU estiver disponível

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import datetime
import pickle

# Disable eager execution warnings
tf.get_logger().setLevel('ERROR')

# Configurações para otimização em CPU
tf.config.threading.set_intra_op_parallelism_threads(4)  # Ajuste conforme número de cores da CPU
tf.config.threading.set_inter_op_parallelism_threads(2)  # Geralmente metade do número de cores

# --- Configurações Globais e Constantes ---
# Caminho para os dados exportados, arrays numpy
DATA_PATH = os.path.join('Sign_Data_App')

# Ações que tentamos detectar (nomes dos gestos)
# Estes devem ser os mesmos usados para treinar o modelo
ACTIONS = np.array(['Beleza'])

# Número de sequências (vídeos) por ação
NO_SEQUENCES = 30
# Comprimento da sequência (frames por vídeo)
SEQUENCE_LENGTH = 30

# --- Funções de Preparação de Dados ---

def load_data():
    """
    Carrega os dados de treinamento do diretório DATA_PATH.
    Retorna: X (features), y (labels)
    """
    sequences, labels = [], []
    
    print(f"Procurando dados em: {os.path.abspath(DATA_PATH)}")
    
    # Verifica se o diretório de dados existe
    if not os.path.exists(DATA_PATH):
        raise ValueError(f"Diretório de dados {DATA_PATH} não encontrado.")
    
    # Lista todos os itens no diretório de dados
    all_items = os.listdir(DATA_PATH)
    print(f"Itens encontrados em {DATA_PATH}: {all_items}")
    
    # Verifica quais ações existem no diretório de dados
    available_actions = []
    for action in all_items:
        action_path = os.path.join(DATA_PATH, action)
        if os.path.isdir(action_path):
            if action in ACTIONS:
                available_actions.append(action)
            else:
                print(f"Diretório {action} encontrado, mas não está na lista ACTIONS: {ACTIONS}")
    
    if not available_actions:
        raise ValueError(f"Nenhuma das ações {ACTIONS} foi encontrada em {DATA_PATH}. Execute a coleta de dados primeiro.")
    
    print(f"Ações encontradas para treinamento: {available_actions}")
    
    # Carrega os dados para cada ação
    for action in available_actions:
        action_index = np.where(ACTIONS == action)[0][0]
        print(f"Processando ação: {action} (índice {action_index})")
        
        # Lista todas as sequências para esta ação
        action_path = os.path.join(DATA_PATH, action)
        sequence_dirs = os.listdir(action_path)
        print(f"Sequências encontradas para {action}: {sequence_dirs}")
        
        # Para cada vídeo/sequência
        for sequence in range(NO_SEQUENCES):
            sequence_path = os.path.join(DATA_PATH, action, str(sequence))
            
            # Verifica se o diretório da sequência existe
            if not os.path.exists(sequence_path):
                print(f"Diretório não encontrado: {sequence_path}, pulando...")
                continue
                
            # Lista todos os arquivos nesta sequência
            sequence_files = os.listdir(sequence_path)
            print(f"Arquivos na sequência {sequence}: {len(sequence_files)}")
            
            window = []
            # Para cada frame na sequência
            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                
                # Verifica se o arquivo do frame existe
                if not os.path.exists(frame_path):
                    print(f"Arquivo não encontrado: {frame_path}, pulando...")
                    continue
                    
                try:
                    res = np.load(frame_path)
                    window.append(res)
                except Exception as e:
                    print(f"Erro ao carregar {frame_path}: {e}")
                
            # Adiciona a sequência apenas se tiver o número correto de frames
            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(action_index)
                print(f"Sequência {sequence} adicionada com sucesso ({len(window)} frames)")
            else:
                print(f"Sequência incompleta em {sequence_path}: {len(window)}/{SEQUENCE_LENGTH} frames, pulando...")
    
    # Converte para arrays numpy
    if len(sequences) == 0:
        raise ValueError(f"Nenhuma sequência válida encontrada para as ações {ACTIONS}. Verifique os dados de treinamento.")
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Dados carregados: {len(sequences)} sequências, com forma {X.shape}")
    
    return X, y

def preprocess_data(X, y):
    """
    Pré-processa os dados para treinamento.
    X: features (sequências de keypoints)
    y: labels (índices das ações)
    Retorna: X_train, X_test, y_train, y_test
    """
    # Normalização (opcional, dependendo dos dados)
    # X = X / np.max(X)
    
    # Conversão para one-hot encoding
    y_categorical = tf.keras.utils.to_categorical(y).astype(int)
    
    # Divisão em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    print(f"Divisão treino/teste: {X_train.shape}, {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# --- Funções de Modelo ---

def create_lstm_model(input_shape, num_classes):
    """
    Cria um modelo LSTM para classificação de sequências.
    input_shape: forma dos dados de entrada (SEQUENCE_LENGTH, num_features)
    num_classes: número de classes (ações)
    Retorna: modelo Keras compilado
    """
    # Cria o modelo usando a API Sequencial com Input como primeira camada
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilação do modelo com otimizações para CPU
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    print(model.summary())
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=16):
    """
    Treina o modelo com os dados fornecidos.
    Retorna: histórico de treinamento
    """
    # Cria diretório para logs e checkpoints
    log_dir = os.path.join('Logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs('Models', exist_ok=True)
    
    # Callbacks para monitoramento e salvamento
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint_path = os.path.join('Models', 'model_checkpoint.keras')
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Treinamento do modelo
    print(f"Iniciando treinamento com {epochs} épocas, batch size {batch_size}...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard_callback, checkpoint_callback, early_stopping]
    )
    
    training_time = time.time() - start_time
    print(f"Treinamento concluído em {training_time:.2f} segundos.")
    
    return history, checkpoint_path

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo com os dados de teste.
    """
    # Avaliação do modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
    
    # Previsões para análise mais detalhada
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calcula a matriz de confusão (opcional)
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Imprime relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=ACTIONS))
    
    return cm, y_pred, y_true_classes

def plot_training_history(history):
    """
    Plota o histórico de treinamento (acurácia e perda).
    """
    # Plota acurácia
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['categorical_accuracy'], label='Treino')
    plt.plot(history.history['val_categorical_accuracy'], label='Validação')
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend()
    
    # Plota perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    Plota a matriz de confusão.
    """
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def save_model(model, filepath='Models/sign_language_model.keras'):
    """
    Salva o modelo treinado.
    """
    model.save(filepath)
    print(f"Modelo salvo em: {filepath}")
    
    # Salva também as ações para referência
    with open('Models/actions.pkl', 'wb') as f:
        pickle.dump(ACTIONS, f)
    
    return filepath

# --- Função Principal ---
def main():
    print("=== Iniciando Treinamento do Modelo de Reconhecimento de Libras ===")
    
    # 1. Carrega os dados
    try:
        X, y = load_data()
    except ValueError as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    # 2. Pré-processa os dados
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # 3. Cria o modelo
    input_shape = (SEQUENCE_LENGTH, X.shape[2])  # (30, 1668) - 30 frames, 1668 features por frame
    num_classes = len(ACTIONS)
    model = create_lstm_model(input_shape, num_classes)
    
    # 4. Treina o modelo
    history, checkpoint_path = train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=16)
    
    # 5. Avalia o modelo
    cm, _, _ = evaluate_model(model, X_test, y_test)
    
    # 6. Plota resultados
    plot_training_history(history)
    plot_confusion_matrix(cm, ACTIONS)
    
    # 7. Salva o modelo final
    model_path = save_model(model)
    
    print(f"=== Treinamento Concluído! Modelo salvo em {model_path} ===")
    print(f"Você pode carregar este modelo no aplicativo principal para reconhecimento em tempo real.")

if __name__ == "__main__":
    main()
