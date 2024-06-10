import tkinter as tk
from tkinter import messagebox, END
import threading
import time
import cv2
import face_recognition
import numpy as np
import pickle
import sqlite3
from datetime import datetime
import csv

class PresenceSystem:
    def __init__(self):
        # Inicialização da interface gráfica
        self.root = tk.Tk()
        self.root.title("Sistema de Presença")
        
        # Etiquetas para exibir o número de alunos presentes e ausentes
        self.students_present_label = tk.Label(self.root, text="Alunos Presentes: 0")
        self.students_present_label.pack()
        
        self.students_absent_label = tk.Label(self.root, text="Alunos Ausentes: 0")
        self.students_absent_label.pack()
        
        # Caixa de texto para exibir a lista de alunos presentes
        self.students_present_text = tk.Text(self.root, height=10, width=50)
        self.students_present_text.pack()
        
        # Botão para iniciar/parar o reconhecimento facial
        self.toggle_button = tk.Button(self.root, text="Iniciar Reconhecimento", command=self.toggle_recognition)
        self.toggle_button.pack()
        
        # Variáveis de controle
        self.recognition_running = False
        self.students_present = set()
        self.all_known_students = set()
        
        # Carregar os dados do modelo treinado
        with open('src/trained_faces.pkl', 'rb') as f:
            self.known_face_encodings, self.known_face_names = pickle.load(f)
        
        # Obter todos os alunos conhecidos
        self.all_known_students.update(self.known_face_names)
        
        # Configuração para fechar a janela corretamente
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Criar o arquivo de registro de presença se não existir
        self.create_log_file()
        
    def toggle_recognition(self):
        # Método para alternar entre iniciar e parar o reconhecimento facial
        if self.recognition_running:
            self.recognition_running = False
            self.toggle_button.config(text="Iniciar Reconhecimento")
            self.reset_data()
        else:
            self.recognition_running = True
            self.toggle_button.config(text="Parar Reconhecimento")
            threading.Thread(target=self.start_recognition).start()
            
    def start_recognition(self):
        # Método para iniciar o reconhecimento facial em uma thread separada
        video_capture = cv2.VideoCapture(0)
        while self.recognition_running:
            self.students_present.clear()
            ret, frame = video_capture.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Desconhecido"
                
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    if name.count('_') == 3:
                        nome, ra, curso, periodo = name.split('_')
                    else:
                        nome, ra, curso, periodo = ("Desconhecido", "Desconhecido", "Desconhecido", "Desconhecido")
                    
                    now = datetime.now()
                    data = now.strftime("%Y-%m-%d")
                    hora = now.strftime("%H:%M:%S")
                    
                    self.insert_presence(ra, nome, curso, periodo, data, hora)
                    self.add_to_log(ra, nome, curso, periodo, data, hora)
                    self.students_present.add(name)
                else:
                    print("Aluno não conhecido")
            
            self.update_presence_labels()
            self.update_present_list()
            time.sleep(3)
        
        video_capture.release()

    def insert_presence(self, ra, nome, curso, periodo, data, hora):
        # Método para inserir dados de presença no banco de dados SQLite
        conn = sqlite3.connect('database/presenca.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO presenca (ra, nome, curso, periodo, data, hora) 
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (ra, nome, curso, periodo, data, hora))
        conn.commit()
        conn.close()
        print(f"Dados do aluno {nome} inseridos com sucesso no banco de dados.")
    
    def create_log_file(self):
        # Método para criar o arquivo de log de presença se ele não existir
        with open('presenca_log.csv', 'w', newline='') as csvfile:
            fieldnames = ['RA', 'Nome', 'Curso', 'Periodo', 'Data', 'Hora']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def add_to_log(self, ra, nome, curso, periodo, data, hora):
        # Método para adicionar dados de presença ao arquivo de log de presença
        with open('presenca_log.csv', 'a', newline='') as csvfile:
            fieldnames = ['RA', 'Nome', 'Curso', 'Periodo', 'Data', 'Hora']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'RA': ra, 'Nome': nome, 'Curso': curso, 'Periodo': periodo, 'Data': data, 'Hora': hora})
        print(f"Dados do aluno {nome} adicionados ao log de presença.")
    
    def update_presence_labels(self):
        # Método para atualizar as etiquetas na interface gráfica com o número de alunos presentes e ausentes
        students_present_count = len(self.students_present)
        students_absent_count = len(self.all_known_students) - students_present_count
        
        self.students_present_label.config(text=f"Alunos Presentes: {students_present_count}")
        self.students_absent_label.config(text=f"Alunos Ausentes: {students_absent_count}")
    
    def update_present_list(self):
        # Método para atualizar o texto na caixa de texto com a lista de alunos presentes
        self.students_present_text.delete(1.0, END)
        for student in self.students_present:
            self.students_present_text.insert(tk.END, f"{student}\n")
    
    def reset_data(self):
        # Método para limpar os dados de alunos presentes
        self.students_present.clear()
    
    def on_close(self):
        # Método chamado ao fechar a janela da aplicação
        if messagebox.askokcancel("Fechar", "Deseja fechar o sistema?"):
            self.recognition_running = False
            self.reset_data()
            self.root.destroy()

if __name__ == "__main__":
    # Ponto de entrada do programa
    app = PresenceSystem()
    app.root.mainloop()
