import mysql.connector
from datetime import datetime, timedelta
import random
import numpy as np

# Configuração da conexão
def get_db_connection():
    return mysql.connector.connect(
        host="34.151.221.45",
        user="agrovim_user",
        password="Senha2025",
        database="dados_producao"
    )

# Função para gerar dados realistas
def generate_sample_data(num_records=10000):  # Aumentei para 10.000 registros
    data = []
    now = datetime.now()
    turbine_ids = [f"TURB-{i:02d}" for i in range(1, 6)]  # 5 turbinas
    
    for _ in range(num_records):
        # Janela de tempo de 6 meses (180 dias) em vez de 30 dias
        timestamp = now - timedelta(
            days=random.uniform(0, 180),
            hours=random.uniform(0, 24),
            minutes=random.uniform(0, 60)
        )
        
        turbina = random.choice(turbine_ids)
        
        # Gerar dados baseados no status
        if random.random() < 0.9:  # Normal
            acelerometro = random.uniform(1.5, 3.0)
            strain_gauge = random.uniform(300, 600)
            torque = random.uniform(10000, 15000)
            anemometro = random.uniform(5.0, 10.0)
            status = "Normal"
        elif random.random() < 0.7:  # Alerta
            acelerometro = random.uniform(3.0, 5.0)
            strain_gauge = random.uniform(600, 1500)
            torque = random.uniform(15000, 20000)
            anemometro = random.uniform(10.0, 20.0)
            status = "Alerta"
        else:  # Falha
            acelerometro = random.uniform(5.0, 8.0)
            strain_gauge = random.uniform(1500, 2500)
            torque = random.uniform(20000, 30000)
            anemometro = random.uniform(20.0, 30.0)
            status = "Falha"
        
        # Adicionar ruído e garantir limites
        acelerometro = max(0, min(10, acelerometro + random.gauss(0, 0.1)))
        strain_gauge = max(0, min(3000, strain_gauge + random.gauss(0, 50)))
        torque = max(0, min(30000, torque + random.gauss(0, 500)))
        anemometro = max(0, min(30, anemometro + random.gauss(0, 0.5)))
        
        # Outros sensores com valores correlacionados
        termopar = 25 + (acelerometro * 2) + random.gauss(0, 2)
        sensor_ir = 20 + (strain_gauge / 100) + random.gauss(0, 3)
        corrente = 800 + (torque / 30) + random.gauss(0, 50)
        tensao = 650 + (anemometro * 2) + random.gauss(0, 10)
        
        data.append((
            timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            round(acelerometro, 2),
            round(strain_gauge, 2),
            round(torque, 2),
            round(termopar, 2),
            round(sensor_ir, 2),
            round(corrente, 2),
            round(tensao, 2),
            round(anemometro, 2),
            round(random.uniform(100, 200), 2),  # CataVento
            round(random.uniform(40, 80), 2),    # SensorUmidade
            round(random.uniform(10, 20), 2),    # Encoder
            round(random.uniform(15, 25), 2),    # SensorAngulo
            round(random.uniform(15, 50), 2),    # Fluxometro
            turbina,
            status
        ))
    
    return data

# Inserir dados no banco
def insert_sample_data():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Criar tabela se não existir
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dados (
            id INT AUTO_INCREMENT PRIMARY KEY,
            TimeStamp DATETIME,
            Acelerometro FLOAT,
            StrainGauge FLOAT,
            SensorTorque FLOAT,
            Termopar FLOAT,
            SensorIR FLOAT,
            SensorCorrente FLOAT,
            SensorTensao FLOAT,
            Anemometro FLOAT,
            CataVento FLOAT,
            SensorUmidade FLOAT,
            Encoder FLOAT,
            SensorAngulo FLOAT,
            Fluxometro FLOAT,
            Turbina VARCHAR(20),
            Status VARCHAR(20)
        )
        """)
        
        # Gerar dados - aumentei para 10.000 registros
        sample_data = generate_sample_data(10000)
        
        # Query de inserção
        insert_query = """
        INSERT INTO dados (
            TimeStamp, Acelerometro, StrainGauge, SensorTorque, Termopar,
            SensorIR, SensorCorrente, SensorTensao, Anemometro, CataVento,
            SensorUmidade, Encoder, SensorAngulo, Fluxometro, Turbina, Status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Inserir em lotes de 100
        batch_size = 100
        for i in range(0, len(sample_data), batch_size):
            batch = sample_data[i:i+batch_size]
            cursor.executemany(insert_query, batch)
            conn.commit()
            print(f"Inseridos {i+len(batch)} registros")
        
        print("✅ Dados inseridos com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao inserir dados: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

if __name__ == "__main__":
    print("Iniciando população do banco de dados...")
    insert_sample_data()
    print("Processo concluído.")
