import face_recognition
import os
import pickle

# Diretório onde as imagens dos alunos estão armazenadas
ALUNOS_DIR = 'alunos/'

def load_images(directory):
    """
    Carrega imagens de um diretório e retorna as codificações faciais e os nomes das pessoas nelas contidas.
    """
    face_encodings = []
    face_names = []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(root, filename)
                try:
                    image = face_recognition.load_image_file(img_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    face_encodings.append(encoding)
                    face_names.append(os.path.basename(root))
                except Exception as e:
                    print(f"Erro ao processar a imagem {img_path}: {e}")

    return face_encodings, face_names

def save_model(face_encodings, face_names, filename='src/trained_faces.pkl'):
    """
    Salva as codificações faciais e os nomes em um arquivo pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump((face_encodings, face_names), f)

def train_model():
    """
    Função principal para treinar o modelo.
    """
    face_encodings, face_names = load_images(ALUNOS_DIR)
    save_model(face_encodings, face_names)

# Treinar o modelo
train_model()
