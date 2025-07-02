import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import time

# Configuração da página
st.set_page_config(
    page_title="Monitoramento de Turbinas",
    page_icon="🌪️",
    layout="wide"
)

# --- Conexão com Banco de Dados (Versão Corrigida) ---
@st.cache_resource(ttl=60)  # Reconecta a cada 1 minuto
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="34.151.221.45",
            user="agrovim_user",
            password="Senha2025",
            database="dados_producao",
            connect_timeout=3  # Timeout reduzido para falha rápida
        )
        
        # Teste simples de conexão
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return conn
            
    except Exception as e:
        st.error(f"⚠️ Falha na conexão com o banco de dados. Erro: {str(e)}")
        st.markdown("""
        **Verifique:**
        1. Servidor MySQL em 34.151.221.45
        2. Usuário: agrovim_user
        3. Senha correta
        4. Conexão de rede
        """)
        return None

# --- Carregar Dados ---
@st.cache_data(ttl=300)
def load_data(days=30):
    conn = None
    try:
        conn = get_db_connection()
        if conn is None or not conn.is_connected():
            return pd.DataFrame()
            
        query = f"""
        SELECT * FROM dados 
        WHERE TimeStamp >= NOW() - INTERVAL {days} DAY
        ORDER BY TimeStamp DESC
        """
        df = pd.read_sql(query, conn)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()

# --- Modelo de Predição ---
def train_model(df):
    try:
        # Pré-processamento
        df['hora'] = df['TimeStamp'].dt.hour
        df['dia_semana'] = df['TimeStamp'].dt.dayofweek
        
        # Seleção de features
        features = ['Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro', 'hora', 'dia_semana']
        target = 'Status'
        
        # Transformar status em numérico (1 para Falha, 0 para outros)
        df['status_numerico'] = df[target].apply(lambda x: 1 if x == 'Falha' else 0)
        
        X = df[features]
        y = df['status_numerico']
        
        # Treinar modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
        
    except Exception as e:
        st.error(f"Erro no treinamento do modelo: {str(e)}")
        return None

# --- Seção de Predição ---
def show_prediction_section(df, model):
    st.header("🔮 Predição de Falhas")
    
    with st.expander("Configuração de Predição"):
        col1, col2 = st.columns(2)
        with col1:
            accel = st.slider("Acelerômetro", 0.0, 10.0, 2.5)
            strain = st.slider("Strain Gauge", 0, 3000, 500)
        with col2:
            torque = st.slider("Torque", 0, 30000, 12000)
            wind = st.slider("Velocidade do Vento", 0.0, 30.0, 7.5)
        
        hora = st.slider("Hora do Dia", 0, 23, 12)
        dia_semana = st.selectbox("Dia da Semana", ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"], index=0)
        dia_num = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"].index(dia_semana)
        
        if st.button("Calcular Probabilidade de Falha") and model:
            input_data = [[accel, strain, torque, wind, hora, dia_num]]
            prob = model.predict(input_data)[0] * 100
            
            # Mostrar resultado
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidade de Falha"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prob
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendações
            if prob > 70:
                st.error("⚠️ Alto risco de falha! Recomenda-se:")
                st.markdown("- Parada para manutenção preventiva\n- Verificação dos componentes críticos")
            elif prob > 30:
                st.warning("⚠️ Risco moderado de falha. Monitorar:")
                st.markdown("- Vibrações anormais\n- Temperatura dos componentes")
            else:
                st.success("✅ Condição operacional normal")

# --- Página Principal ---
def main():
    st.title("🌪️ Monitoramento de Turbinas Eólicas")
    
    # Verificação de conexão
    if get_db_connection() is None:
        return
    
    # Carregar dados
    df = load_data(60)
    if df.empty:
        st.warning("Nenhum dado encontrado no banco de dados!")
        return
    
    # Treinar modelo
    model = train_model(df)
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Visualização dos dados
        st.header("📊 Dados das Turbinas")
        selected_turbine = st.selectbox("Selecione a Turbina", df['Turbina'].unique())
        
        filtered_df = df[df['Turbina'] == selected_turbine]
        
        fig = px.line(
            filtered_df, 
            x='TimeStamp', 
            y='Acelerometro',
            color='Status',
            title=f"Variação do Acelerômetro - {selected_turbine}",
            color_discrete_map={
                'Normal': 'green',
                'Alerta': 'orange',
                'Falha': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Estatísticas rápidas
        st.header("📈 Estatísticas")
        st.metric("Total de Registros", len(df))
        st.metric("Turbinas Ativas", df['Turbina'].nunique())
        
        status_counts = df['Status'].value_counts()
        st.plotly_chart(
            px.pie(
                status_counts, 
                values=status_counts.values, 
                names=status_counts.index,
                title="Distribuição de Status"
            ),
            use_container_width=True
        )
    
    # Seção de predição
    if model:
        show_prediction_section(df, model)
    else:
        st.warning("Modelo de predição não disponível")

if __name__ == "__main__":
    main()
