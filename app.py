import streamlit as st
import pandas as pd
import plotly.express as px
import mysql.connector
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Monitoramento de Turbinas",
    page_icon="🌪️",
    layout="wide"
)

# Conexão com o banco de dados
def connect_db():
    try:
        conn = mysql.connector.connect(
            host="34.151.221.45",
            user="agrovim_user",
            password="Senha2025",
            database="dados_producao"
        )
        return conn
    except Exception as e:
        st.error(f"Erro de conexão: {str(e)}")
        return None

# Carregar dados
def load_data():
    conn = connect_db()
    if conn:
        try:
            query = "SELECT * FROM dados ORDER BY TimeStamp DESC"
            df = pd.read_sql(query, conn)
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
            return df
        except Exception as e:
            st.error(f"Erro na consulta: {str(e)}")
            return pd.DataFrame()
        finally:
            if conn.is_connected():
                conn.close()
    return pd.DataFrame()

def main():
    st.title("🌪️ Dashboard de Monitoramento de Turbinas")
    
    # Carregar dados
    df = load_data()
    
    if df.empty:
        st.warning("Nenhum dado encontrado no banco de dados!")
        return
    
    # Filtros na sidebar
    with st.sidebar:
        st.header("Filtros")
        
        # Filtro de data
        min_date = df['TimeStamp'].min().date()
        max_date = df['TimeStamp'].max().date()
        date_range = st.date_input(
            "Selecione o período",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filtro de turbina
        selected_turbines = st.multiselect(
            "Selecione as turbinas",
            options=df['Turbina'].unique(),
            default=df['Turbina'].unique()
        )
        
        # Filtro de status
        selected_status = st.multiselect(
            "Selecione os status",
            options=df['Status'].unique(),
            default=df['Status'].unique()
        )
    
    # Aplicar filtros - VERSÃO CORRIGIDA
    if len(date_range) == 2:
        mask = (
            (df['TimeStamp'].dt.date >= date_range[0]) & 
            (df['TimeStamp'].dt.date <= date_range[1]) & 
            (df['Turbina'].isin(selected_turbines)) & 
            (df['Status'].isin(selected_status)))
        filtered_df = df.loc[mask]
    else:
        filtered_df = df.copy()
    
    # Métricas
    st.subheader("Métricas Principais")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", len(filtered_df))
    with col2:
        st.metric("Turbinas Ativas", len(filtered_df['Turbina'].unique()))
    with col3:
        status_counts = filtered_df['Status'].value_counts()
        st.metric("Status Predominante", status_counts.idxmax())
    
    # Gráficos
    st.subheader("Análise de Dados")
    
    if not filtered_df.empty:
        # Gráfico de linha para acelerômetro
        fig1 = px.line(
            filtered_df,
            x='TimeStamp',
            y='Acelerometro',
            color='Turbina',
            title="Variação do Acelerômetro"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Gráfico de barras para status
        fig2 = px.bar(
            filtered_df['Status'].value_counts(),
            title="Distribuição de Status",
            labels={'value': 'Contagem', 'index': 'Status'},
            color=filtered_df['Status'].value_counts().index,
            color_discrete_map={
                'Normal': 'green',
                'Alerta': 'orange',
                'Falha': 'red'
            }
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Dados brutos
    st.subheader("Dados Brutos")
    st.dataframe(filtered_df)

if __name__ == "__main__":
    main()
