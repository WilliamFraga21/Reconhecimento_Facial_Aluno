import sqlite3

# Conexão com o banco de dados
conn = sqlite3.connect('database/presenca.db')
cursor = conn.cursor()

# Criação da tabela de presença
cursor.execute('''
CREATE TABLE IF NOT EXISTS presenca (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ra TEXT NOT NULL,
    nome TEXT NOT NULL,
    curso TEXT NOT NULL,
    periodo TEXT NOT NULL,
    data TEXT NOT NULL,
    hora TEXT NOT NULL
);

''')
conn.commit()
conn.close()
