import os

# Caminho para o arquivo do banco de dados
db_path = 'database/presenca.db'

# Verificar se o arquivo existe
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Banco de dados '{db_path}' deletado com sucesso.")
else:
    print(f"Banco de dados '{db_path}' não encontrado.")
