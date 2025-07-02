import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from mysql.connector import Error
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import joblib

# Configuração da página
st.set_page_config(
    page_title="Monitoramento Avançado de Turbinas",
    page_icon="🌪️",
    layout="wide"
)

# --- Conexão com Banco de Dados (Versão Corrigida) ---
@st.cache_resource(ttl=60)
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="34.151.221.45",
            user="agrovim_user",
            password="Senha2025",
            database="dados_producao",
            connect_timeout=3,
            buffered=True  # Solução para "Unread result found"
        )
        
        # Teste de conexão com tratamento seguro do cursor
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchall()  # Garante que todos os resultados são lidos
            return conn
        finally:
            if cursor:
                cursor.close()
                
    except Exception as e:
        st.error(f"⚠️ Falha na conexão: {str(e)}")
        st.markdown("""
        **Soluções para tentar:**
        1. Verifique se o servidor MySQL (34.151.221.45) está online
        2. Confira usuário (agrovim_user) e senha
        3. Valide as regras de firewall/ACL
        4. Recarregue a página após corrigir
        """)
        return None

# --- Carregar Dados com Tratamento Completo ---
@st.cache_data(ttl=300)
def load_monitoring_data(days=30):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame()
            
        cursor = conn.cursor()
        query = f"""
        SELECT * FROM dados 
        WHERE TimeStamp >= NOW() - INTERVAL {days} DAY
        ORDER BY TimeStamp DESC
        """
        
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]  # Obtém nomes das colunas
        data = cursor.fetchall()  # Lê todos os resultados imediatamente
        
        df = pd.DataFrame(data, columns=columns)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

# --- Modelo Preditivo de Falhas ---
def train_failure_model(df):
    try:
        # Pré-processamento
        df['hora'] = df['TimeStamp'].dt.hour
        df['dia_semana'] = df['TimeStamp'].dt.dayofweek
        
        # Features e target
        features = ['Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro', 'hora', 'dia_semana']
        df['target'] = df['Status'].apply(lambda x: 1 if x == 'Falha' else 0)
        
        # Treinamento
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(df[features], df['target'])
        return model
    except Exception as e:
        st.error(f"Erro no treinamento: {str(e)}")
        return None

# --- Interface do Simulador de Falhas ---
def show_failure_simulator(model):
    st.header("🔮 Simulador de Falhas")
    
    with st.expander("Configurar Parâmetros", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            accel = st.slider("Acelerômetro (m/s²)", 0.0, 10.0, 2.5)
            strain = st.slider("Strain Gauge (μm/m)", 0, 3000, 500)
        with col2:
            torque = st.slider("Torque (Nm)", 0, 30000, 12000)
            wind = st.slider("Veloc. Vento (m/s)", 0.0, 30.0, 7.5)
        
        hora = st.slider("Hora do Dia", 0, 23, 12)
        dia_semana = st.selectbox("Dia da Semana", 
                                ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
                                index=0)
        dia_num = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"].index(dia_semana)
        
        if st.button("Calcular Probabilidade de Falha") and model:
            input_data = [[accel, strain, torque, wind, hora, dia_num]]
            prob = model.predict(input_data)[0] * 100
            
            # Visualização
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidade de Falha"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "orange"},
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
                st.error("⚠️ ALERTA: Alto risco de falha iminente!")
                st.markdown("""
                - Parar turbina imediatamente
                - Realizar inspeção completa
                - Verificar sistema de frenagem
                """)
            elif prob > 30:
                st.warning("⚠️ Atenção: Risco moderado de falha")
                st.markdown("""
                - Aumentar frequência de monitoramento
                - Verificar parafusos e fixações
                - Monitorar temperatura
                """)
            else:
                st.success("✅ Operação dentro dos parâmetros normais")

# --- Página Principal ---
def main():
    st.title("🌪️ Painel de Monitoramento de Turbinas")
    
    # Verificação inicial da conexão
    with st.spinner("Conectando ao banco de dados..."):
        if get_db_connection() is None:
            return
    
    # Carregar dados
    df = load_monitoring_data(60)
    if df.empty:
        st.warning("Nenhum dado encontrado na tabela 'dados'")
        return
    
    # Treinar modelo
    model = train_failure_model(df)
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Filtros
        selected_turbine = st.selectbox(
            "Selecione a Turbina",
            options=df['Turbina'].unique()
        )
        
        # Dados filtrados
        filtered_df = df[df['Turbina'] == selected_turbine]
        
        # Gráfico temporal
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
        
        # Mostrar simulador se o modelo estiver disponível
        if model:
            show_failure_simulator(model)
    
    with col2:
        # Estatísticas rápidas
        st.metric("Total de Registros", len(df))
        st.metric("Turbinas Monitoradas", df['Turbina'].nunique())
        
        # Distribuição de status
        status_counts = df['Status'].value_counts()
        st.plotly_chart(
            px.pie(
                status_counts,
                names=status_counts.index,
                values=status_counts.values,
                title="Distribuição de Status",
                color=status_counts.index,
                color_discrete_map={
                    'Normal': 'green',
                    'Alerta': 'orange',
                    'Falha': 'red'
                }
            ),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
