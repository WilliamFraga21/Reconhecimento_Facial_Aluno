import face_recognition
import os
import pickle

# Diretório onde as imagens dos alunos estão armazenadas
ALUNOS_DIR = 'alunos/'

# Função para carregar as imagens e criar os embeddings
def train_model():
    known_face_encodings = []
    known_face_names = []

    for root, dirs, files in os.walk(ALUNOS_DIR):
        for filename in files:
            if filename.endswith(('jpg', 'jpeg', 'png')):
                # Carregar a imagem
                img_path = os.path.join(root, filename)
                image = face_recognition.load_image_file(img_path)
                
                # Obter o encoding da face
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(os.path.basename(root))

    # Salvar os dados em um arquivo
    with open('src/trained_faces.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

# Treinar o modelo
train_model()
