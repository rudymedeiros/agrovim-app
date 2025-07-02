import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import mysql.connector
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Configuração da página
st.set_page_config(
    page_title="Painel Avançado de Turbinas",
    page_icon="🌪️",
    layout="wide"
)

# --- Estilo CSS personalizado ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-style {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Conexão com Banco de Dados ---
@st.cache_resource(ttl=60)
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="34.151.221.45",
            user="agrovim_user",
            password="Senha2025",
            database="dados_producao",
            connect_timeout=3,
            buffered=True
        )
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchall()
            return conn
        finally:
            if cursor:
                cursor.close()                
    except Exception as e:
        st.error(f"⚠️ Falha na conexão: {str(e)}")
        st.markdown("""
        **Soluções para tentar:**
        1. Verifique se o servidor MySQL está online
        2. Confira usuário e senha
        3. Valide as regras de firewall/ACL
        4. Recarregue a página após corrigir
        """)
        return None

# --- Carregar Dados ---
@st.cache_data(ttl=300)
def load_monitoring_data(days=90):
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
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()
        
        df = pd.DataFrame(data, columns=columns)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        
        # Engenharia de features
        df['hora'] = df['TimeStamp'].dt.hour
        df['dia_semana'] = df['TimeStamp'].dt.dayofweek
        df['target'] = df['Status'].apply(lambda x: 1 if x == 'Falha' else 0)
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

# --- Visualizações Profissionais ---
def show_complete_dashboard(df, selected_turbine):
    filtered_df = df[df['Turbina'] == selected_turbine]
    
    # KPIs no topo
    st.subheader(f"📊 Métricas da Turbina {selected_turbine}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Registros", len(filtered_df))
    
    with col2:
        st.metric("Última Leitura", filtered_df['TimeStamp'].max().strftime('%d/%m/%Y %H:%M'))
    
    with col3:
        st.metric("Status Atual", filtered_df['Status'].iloc[0])
    
    with col4:
        st.metric("Média Acelerômetro", f"{filtered_df['Acelerometro'].mean():.2f} m/s²")
    
    # Gráficos principais
    st.markdown('<div class="header-style">📈 Série Temporal Completa</div>', unsafe_allow_html=True)
    
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                         subplot_titles=("Acelerômetro (m/s²)", "Strain Gauge (μm/m)", "Torque (Nm)"))
    
    # Acelerômetro
    fig.add_trace(
        go.Scatter(
            x=filtered_df['TimeStamp'],
            y=filtered_df['Acelerometro'],
            name="Acelerômetro",
            line=dict(color='#3498db'),
            hovertemplate="%{x|%d/%m %H:%M}<br>%{y:.2f} m/s²<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Strain Gauge
    fig.add_trace(
        go.Scatter(
            x=filtered_df['TimeStamp'],
            y=filtered_df['StrainGauge'],
            name="Strain Gauge",
            line=dict(color='#e74c3c'),
            hovertemplate="%{x|%d/%m %H:%M}<br>%{y} μm/m<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Torque
    fig.add_trace(
        go.Scatter(
            x=filtered_df['TimeStamp'],
            y=filtered_df['SensorTorque'],
            name="Torque",
            line=dict(color='#2ecc71'),
            hovertemplate="%{x|%d/%m %H:%M}<br>%{y} Nm<extra></extra>"
        ),
        row=3, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise de Correlação
    st.markdown('<div class="header-style">🔄 Correlação entre Parâmetros</div>', unsafe_allow_html=True)
    
    corr_df = filtered_df[['Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro']].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribuição dos Sensores
    st.markdown('<div class="header-style">📊 Distribuição dos Parâmetros</div>', unsafe_allow_html=True)
    
    fig = px.histogram(
        filtered_df,
        x=['Acelerometro', 'StrainGauge', 'SensorTorque'],
        nbins=30,
        facet_col='variable',
        facet_col_wrap=2,
        facet_row_spacing=0.1,
        facet_col_spacing=0.05,
        color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71']
    )
    fig.update_xaxes(matches=None)
    st.plotly_chart(fig, use_container_width=True)
    
    # Status por Hora do Dia
    st.markdown('<div class="header-style">🕒 Status por Hora do Dia</div>', unsafe_allow_html=True)
    
    fig = px.box(
        filtered_df,
        x='hora',
        y='Acelerometro',
        color='Status',
        color_discrete_map={'Normal': '#2ecc71', 'Alerta': '#f39c12', 'Falha': '#e74c3c'},
        category_orders={"hora": list(range(24))}
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Página Principal ---
def main():
    st.title("🌪️ Painel Avançado de Monitoramento de Turbinas")
    
    # Verificação inicial da conexão
    with st.spinner("Conectando ao banco de dados..."):
        if get_db_connection() is None:
            return
    
    # Carregar dados
    df = load_monitoring_data(90)
    if df.empty:
        st.warning("Nenhum dado encontrado na tabela 'dados'")
        return
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Filtros
        selected_turbine = st.selectbox(
            "Selecione a Turbina",
            options=df['Turbina'].unique(),
            key='turbine_select'
        )
        
        # Dashboard completo
        show_complete_dashboard(df, selected_turbine)
    
    with col2:
        # Visão geral
        st.markdown('<div class="header-style">🔍 Visão Geral</div>', unsafe_allow_html=True)
        
        # Status atual
        status_counts = df['Status'].value_counts()
        fig = px.pie(
            status_counts,
            names=status_counts.index,
            values=status_counts.values,
            hole=0.3,
            color=status_counts.index,
            color_discrete_map={'Normal': '#2ecc71', 'Alerta': '#f39c12', 'Falha': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Últimas leituras
        st.markdown("### Últimas Leituras")
        st.dataframe(
            df.head(10)[['TimeStamp', 'Turbina', 'Acelerometro', 'Status']]
            .sort_values('TimeStamp', ascending=False)
            .style.format({'TimeStamp': lambda x: x.strftime('%d/%m %H:%M')}),
            height=300
        )

if __name__ == "__main__":
    main()
