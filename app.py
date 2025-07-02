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

# --- Conexão com Banco de Dados ---
@st.cache_resource(ttl=3600)
def get_db_connection():
    """
    Padronização: Sempre usar esta função para obter conexões
    """
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            conn = mysql.connector.connect(
                host="34.151.221.45",
                user="agrovim_user",
                password="Senha2025",
                database="dados_producao",
                connection_timeout=5
            )
            
            # Teste de conexão real
            if conn.is_connected():
                cursor = conn.cursor()
                cursor.execute("SELECT 1")  # Teste simples
                cursor.close()
                return conn
                
        except Error as e:
            st.warning(f"Tentativa {attempt + 1}/{max_retries} falhou: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    st.error("Falha permanente na conexão com o banco de dados")
    return None

# --- Carregar Dados ---
@st.cache_data(ttl=300)
def load_monitoring_data(days=30):
    conn = None
    try:
        conn = get_db_connection()
        if not conn or not conn.is_connected():
            return pd.DataFrame()
            
        query = f"""
        SELECT * FROM dados 
        WHERE TimeStamp >= NOW() - INTERVAL {days} DAY
        ORDER BY TimeStamp DESC
        """
        
        df = pd.read_sql(query, conn)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        return df
        
    except Error as e:
        st.error(f"Erro na consulta: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()

# --- Verificação de Conexão ---
def check_db_connection():
    conn = None
    try:
        conn = get_db_connection()
        if conn and conn.is_connected():
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM dados LIMIT 1")  # Teste com a tabela real
            cursor.close()
            return True
        return False
    except:
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()

# --- Análise de Status ---
def analyze_status(df):
    status_counts = df['Status'].value_counts()
    fig = px.pie(
        status_counts,
        names=status_counts.index,
        values=status_counts.values,
        title="Distribuição de Status",
        color=status_counts.index,
        color_discrete_map={'Normal':'green','Alerta':'orange','Falha':'red'}
    )
    return fig

# --- Modelo Preditivo ---
def train_failure_model(df):
    try:
        # Usando colunas existentes na tabela 'dados'
        X = df[['Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro']]
        y = df['Status'].apply(lambda x: 1 if x == 'Falha' else 0)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Erro no modelo: {str(e)}")
        return None

# --- Página Principal ---
def main():
    st.title("🌪️ Painel de Monitoramento de Turbinas")
    
    # Verificação inicial da conexão
    with st.spinner("Verificando conexão com o banco de dados..."):
        if not check_db_connection():
            st.error("""
            ⚠️ Falha na conexão com o banco. Verifique:
            1. Servidor MySQL em 34.151.221.45
            2. Usuário: agrovim_user
            3. Tabela: 'dados'
            """)
            return
    
    # Carregar dados
    df = load_monitoring_data(60)
    
    if df.empty:
        st.warning("Nenhum dado encontrado na tabela 'dados'")
        return

    # Sidebar - Filtros
    with st.sidebar:
        st.header("🔧 Filtros")
        min_date = df['TimeStamp'].min().to_pydatetime()
        max_date = df['TimeStamp'].max().to_pydatetime()
        date_range = st.date_input(
            "Período",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        selected_turbines = st.multiselect(
            "Turbinas",
            options=df['Turbina'].unique(),
            default=df['Turbina'].unique()
        )
        
        selected_status = st.multiselect(
            "Status",
            options=df['Status'].unique(),
            default=df['Status'].unique()
        )

    # Aplicar filtros
    if len(date_range) == 2:
        mask = (
            (df['TimeStamp'].dt.date >= date_range[0]) & 
            (df['TimeStamp'].dt.date <= date_range[1]) & 
            (df['Turbina'].isin(selected_turbines)) & 
            (df['Status'].isin(selected_status)))
        filtered_df = df.loc[mask]
    else:
        filtered_df = df

    # Métricas em Tempo Real
    st.header("📊 Métricas Operacionais")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Turbinas Monitoradas", filtered_df['Turbina'].nunique())
    with col2:
        st.metric("Registros", len(filtered_df))
    with col3:
        failure_rate = len(filtered_df[filtered_df['Status'] == 'Falha']) / len(filtered_df) if len(filtered_df) > 0 else 0
        st.metric("Taxa de Falhas", f"{failure_rate:.2%}")
    with col4:
        last_update = filtered_df['TimeStamp'].max().strftime("%d/%m/%Y %H:%M")
        st.metric("Última Atualização", last_update)

    # Visualizações
    tab1, tab2, tab3 = st.tabs(["📈 Tendências", "🔍 Análise de Status", "🛠️ Manutenção Preditiva"])

    with tab1:
        st.subheader("Comportamento Temporal")
        sensor_options = ['Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro']
        selected_sensor = st.selectbox("Selecione o Sensor", sensor_options)
        
        fig = px.line(
            filtered_df,
            x='TimeStamp',
            y=selected_sensor,
            color='Turbina',
            title=f"Variação do {selected_sensor}",
            hover_data=['Status']
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Análise de Status")
        st.plotly_chart(analyze_status(filtered_df), use_container_width=True)
        
        st.subheader("Correlação entre Sensores")
        numeric_cols = filtered_df.select_dtypes(include=['float64']).columns
        corr_matrix = filtered_df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Previsão de Falhas")
        model = train_failure_model(df)
        
        if model:
            st.success("Modelo carregado com sucesso!")
            
            with st.expander("🔮 Simulador de Condições"):
                col1, col2 = st.columns(2)
                with col1:
                    accel = st.slider("Acelerômetro", 0.0, 10.0, 2.5)
                    strain = st.slider("Strain Gauge", 0, 3000, 500)
                with col2:
                    torque = st.slider("Torque", 0, 30000, 12000)
                    wind = st.slider("Velocidade do Vento (Anemômetro)", 0.0, 30.0, 7.5)
                
                if st.button("Prever Probabilidade de Falha"):
                    prediction = model.predict([[accel, strain, torque, wind]])[0]
                    prob = prediction * 100
                    
                    gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probabilidade de Falha"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 80], 'color': "orange"},
                                {'range': [80, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': prob
                            }
                        }
                    ))
                    st.plotly_chart(gauge, use_container_width=True)

    # Dados Brutos
    with st.expander("📁 Visualizar Dados Completos"):
        st.dataframe(
            filtered_df.sort_values('TimeStamp', ascending=False),
            height=300,
            use_container_width=True
        )

if __name__ == "__main__":
    main()
