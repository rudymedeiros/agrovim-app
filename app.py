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
@st.cache_resource(ttl=3600)  # Reconecta a cada 1 hora
def get_db_connection(max_retries=3, retry_delay=2):
    retry_count = 0
    last_exception = None
    
    while retry_count < max_retries:
        try:
            conn = mysql.connector.connect(
                host="34.151.221.45",
                user="agrovim_user",
                password="Senha2025",
                database="dados_producao",
                connection_timeout=5,
                pool_name="agrovim_pool",
                pool_size=3
            )
            
            if conn.is_connected():
                return conn
                
        except Error as e:
            last_exception = e
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)
    
    st.error(f"Falha na conexão após {max_retries} tentativas. Erro: {str(last_exception)}")
    return None

# --- Carregar Dados do MySQL ---
@st.cache_data(ttl=300)
def load_monitoring_data(days=30):
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame()
            
        query = f"""
        SELECT * FROM dados_turbinas 
        WHERE TimeStamp >= NOW() - INTERVAL {days} DAY
        ORDER BY TimeStamp DESC
        """
        
        df = pd.read_sql(query, conn)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        return df
        
    except Error as e:
        st.error(f"Erro na consulta SQL: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()

# --- Verificação de Conexão ---
def check_db_connection():
    test_conn = None
    try:
        test_conn = get_db_connection(max_retries=1)
        return test_conn and test_conn.is_connected()
    finally:
        if test_conn and test_conn.is_connected():
            test_conn.close()

# --- Análise de Status ---
def analyze_status(df):
    status_counts = df['Status'].value_counts()
    fig = px.pie(
        status_counts,
        names=status_counts.index,
        values=status_counts.values,
        title="Distribuição de Status",
        color=status_counts.index,
        color_discrete_map={'Normal':'#2ecc71','Alerta':'#f39c12','Falha':'#e74c3c'}
    )
    return fig

# --- Modelo Preditivo ---
def train_failure_model(df):
    try:
        X = df[['Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro']]
        y = df['Status'].apply(lambda x: 1 if x == 'Falha' else 0)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Erro no treinamento: {str(e)}")
        return None

# --- Página Principal ---
def main():
    st.title("🌪️ Painel de Monitoramento Avançado")
    
    # Verificação inicial da conexão
    if not check_db_connection():
        st.error("⚠️ Não foi possível estabelecer conexão com o banco de dados. Verifique:")
        st.markdown("""
        - Credenciais de acesso
        - Endereço do servidor MySQL (34.151.221.45)
        - Regras de firewall na nuvem
        - Status do serviço MySQL
        """)
        return
    
    # Carregar dados
    df = load_monitoring_data(60)
    
    if df.empty:
        st.warning("Nenhum dado encontrado no banco de dados!")
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
            (df['Status'].isin(selected_status))
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
        st.metric("Taxa de Falhas", 
                 f"{len(filtered_df[filtered_df['Status'] == 'Falha']) / len(filtered_df):.2%}")
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
