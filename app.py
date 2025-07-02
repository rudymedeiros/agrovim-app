import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import joblib

# Configuração da página
st.set_page_config(
    page_title="Monitoramento de Turbinas",
    page_icon="🌪️",
    layout="wide"
)

# --- Conexão com o banco (nome uniformizado) ---
@st.cache_resource
def get_db_connection():
    try:
        return mysql.connector.connect(
            host="34.151.221.45",
            user="agrovim_user",
            password="Senha2025",
            database="dados_producao"
        )
    except Exception as e:
        st.error(f"Erro de conexão: {str(e)}")
        return None

# --- Carregar dados de produção ---
@st.cache_data(ttl=300)
def load_production_data(days=30):
    conn = get_db_connection()
    if conn:
        try:
            query = f"""
            SELECT data_hora, producao, temperatura, pressao 
            FROM producao 
            WHERE data_hora >= NOW() - INTERVAL {days} DAY
            ORDER BY data_hora
            """
            return pd.read_sql(query, conn)
        finally:
            conn.close()
    return pd.DataFrame()

# --- Carregar dados de monitoramento ---
@st.cache_data(ttl=300)
def load_monitoring_data():
    conn = get_db_connection()
    if conn:
        try:
            query = "SELECT * FROM dados ORDER BY TimeStamp DESC"
            df = pd.read_sql(query, conn)
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
            return df
        finally:
            conn.close()
    return pd.DataFrame()

# --- Modelo de previsão ---
def train_prediction_model(df):
    try:
        df = df.copy()
        df['hora'] = df['data_hora'].dt.hour
        df['dia_semana'] = df['data_hora'].dt.dayofweek
        
        X = df[['temperatura', 'pressao', 'hora', 'dia_semana']]
        y = df['producao']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Erro no modelo: {str(e)}")
        return None

# --- Seção de Previsão ---
def show_prediction_section():
    st.header("🔮 Previsão de Produção")
    df = load_production_data(60)
    
    if df.empty:
        st.warning("Dados insuficientes para previsão.")
        return
    
    model = train_prediction_model(df)
    if not model:
        return

    cols = st.columns(3)
    with cols[0]:
        temp = st.number_input("Temperatura (°C)", value=25.0)
    with cols[1]:
        pressao = st.number_input("Pressão (hPa)", value=1013.0)
    with cols[2]:
        horas = st.slider("Horas à frente", 1, 24, 6)

    if st.button("Prever Produção"):
        hora_atual = datetime.now().hour
        dia_semana = datetime.now().weekday()
        
        previsoes = []
        for h in range(hora_atual, hora_atual + horas):
            h = h % 24
            pred = model.predict([[temp, pressao, h, dia_semana]])[0]
            previsoes.append(pred)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(horas)),
            y=previsoes,
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title=f"Previsão para as próximas {horas} horas",
            xaxis_title="Horas",
            yaxis_title="Produção (unidades)"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Main ---
def main():
    st.title("🌪️ Dashboard de Monitoramento de Turbinas")
    
    # Seção de previsão
    show_prediction_section()
    
    # Seção de monitoramento
    df = load_monitoring_data()
    if df.empty:
        st.warning("Nenhum dado encontrado!")
        return

    # Filtros (sidebar)
    with st.sidebar:
        st.header("Filtros")
        min_date = df['TimeStamp'].min().date()
        max_date = df['TimeStamp'].max().date()
        date_range = st.date_input(
            "Período",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        turbines = st.multiselect(
            "Turbinas",
            options=df['Turbina'].unique(),
            default=df['Turbina'].unique()
        )
        
        statuses = st.multiselect(
            "Status",
            options=df['Status'].unique(),
            default=df['Status'].unique()
        )

    # Aplicar filtros
    if len(date_range) == 2:
        mask = (
            (df['TimeStamp'].dt.date >= date_range[0]) & 
            (df['TimeStamp'].dt.date <= date_range[1]) & 
            (df['Turbina'].isin(turbines)) & 
            (df['Status'].isin(statuses))
        filtered_df = df.loc[mask]
    else:
        filtered_df = df

    # Métricas
    cols = st.columns(3)
    cols[0].metric("Registros", len(filtered_df))
    cols[1].metric("Turbinas Ativas", filtered_df['Turbina'].nunique())
    cols[2].metric("Status Predominante", filtered_df['Status'].mode()[0])

    # Gráficos
    if not filtered_df.empty:
        fig1 = px.line(
            filtered_df,
            x='TimeStamp',
            y='Acelerometro',
            color='Turbina',
            title="Acelerômetro por Turbina"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.bar(
            filtered_df['Status'].value_counts(),
            title="Distribuição de Status",
            color=filtered_df['Status'].value_counts().index,
            color_discrete_map={'Normal':'green', 'Alerta':'orange', 'Falha':'red'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Dados brutos
    with st.expander("Visualizar Dados Brutos"):
        st.dataframe(filtered_df)

if __name__ == "__main__":
    main()
