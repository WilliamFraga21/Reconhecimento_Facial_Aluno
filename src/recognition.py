import face_recognition
import cv2
import numpy as np
import pickle
import sqlite3
from datetime import datetime
import time
import csv

# Carregar os dados do modelo treinado
with open('src/trained_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Conectar ao banco de dados
def insert_presence(ra, nome, turma, periodo, data, hora):
    conn = sqlite3.connect('database/presenca.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO presenca (ra, nome, turma, periodo, data, hora) 
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (ra, nome, turma, periodo, data, hora))
    conn.commit()
    conn.close()

# Criar o arquivo de registro
def create_log_file():
    with open('presenca_log.csv', 'w', newline='') as csvfile:
        fieldnames = ['RA', 'Nome', 'Turma', 'Periodo', 'Data', 'Hora']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

# Adicionar entrada no arquivo de registro
def add_to_log(ra, nome, turma, periodo, data, hora):
    with open('presenca_log.csv', 'a', newline='') as csvfile:
        fieldnames = ['RA', 'Nome', 'Turma', 'Periodo', 'Data', 'Hora']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'RA': ra, 'Nome': nome, 'Turma': turma, 'Periodo': periodo, 'Data': data, 'Hora': hora})

# Inicializar a webcam
video_capture = cv2.VideoCapture(0)

# Criar o arquivo de registro se não existir
create_log_file()

while True:
    # Capturar frame por frame
    ret, frame = video_capture.read()
    
    # Converter o frame de BGR (OpenCV) para RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Encontrar todas as faces e encodings no frame atual
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        # Calcular a menor distância para encontrar o melhor ajuste
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        # Exibir o nome da pessoa reconhecida
        print(f"Aluno presente: {name}")
        
        # Obter data e hora atual
        now = datetime.now()
        data = now.strftime("%Y-%m-%d")
        hora = now.strftime("%H:%M:%S")
        
        # Inserir presença no banco de dados
        ra, turma, periodo = name.split('_')
        insert_presence(ra, name, turma, periodo, data, hora)
        
        # Adicionar entrada no arquivo de registro
        add_to_log(ra, name, turma, periodo, data, hora)
    
    # Aguardar 20 segundos antes da próxima verificação
    time.sleep(10)

# Liberar a captura de vídeo
video_capture.release()
cv2.destroyAllWindows()
