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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Configuração da página
st.set_page_config(
    page_title="Painel Inteligente de Turbinas",
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
    .alert-high {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff0000;
    }
    .alert-medium {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
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
        return None

# --- Carregar Dados com Engenharia de Features ---
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
        df['mes'] = df['TimeStamp'].dt.month
        
        # Criando médias móveis
        for window in [3, 6, 12]:  # horas
            df[f'acel_media_{window}h'] = df.groupby('Turbina')['Acelerometro'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Target para classificação
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

# --- Modelo Preditivo Avançado ---
def train_advanced_model(df):
    try:
        # Features selecionadas
        features = ['Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro', 
                   'hora', 'dia_semana', 'mes', 'acel_media_3h', 'acel_media_6h']
        
        X = df[features]
        y = df['target']
        
        # Divisão temporal (não aleatória)
        tscv = TimeSeriesSplit(n_splits=3)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        # Calibração de probabilidades
        calibrated_model = CalibratedClassifierCV(pipeline, cv=3, method='isotonic')
        calibrated_model.fit(X_train, y_train)
        
        # Avaliação
        y_proba = calibrated_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        
        return calibrated_model, features, roc_auc
        
    except Exception as e:
        st.error(f"Erro no treinamento: {str(e)}")
        return None, None, None

# --- Explicação do Modelo ---
def explain_model(model, features):
    try:
        # Acessar o modelo base dentro do pipeline
        rf_model = model.base_estimator.named_steps['model']
        
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            st.subheader("📊 Fatores Mais Influentes")
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Contribuição de Cada Variável para o Modelo',
                color='Importance',
                color_continuous_scale='Bluered'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Explicação textual
            st.markdown("""
            **Interpretação:**
            - Valores maiores indicam maior influência na previsão de falhas
            - Variáveis com importância próxima a zero têm pouco impacto
            - Médias móveis (acel_media) capturam tendências temporais
            """)
    except Exception as e:
        st.warning(f"Não foi possível explicar o modelo: {str(e)}")

# --- Simulador de Falhas Inteligente ---
def show_ai_simulator(model, features, roc_auc):
    st.header("🤖 Simulador Preditivo de Falhas")
    
    with st.expander("Configurar Parâmetros de Simulação", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            accel = st.slider("Acelerômetro (m/s²)", 0.0, 10.0, 2.5, key='accel_sim')
            strain = st.slider("Strain Gauge (μm/m)", 0, 3000, 500, key='strain_sim')
            torque = st.slider("Torque (Nm)", 0, 30000, 12000, key='torque_sim')
        with col2:
            wind = st.slider("Veloc. Vento (m/s)", 0.0, 30.0, 7.5, key='wind_sim')
            hora = st.slider("Hora do Dia", 0, 23, 12, key='hora_sim')
            dia_semana = st.selectbox("Dia da Semana", 
                                    ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
                                    index=0, key='dia_semana_sim')
        
        dia_num = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"].index(dia_semana)
        mes = st.slider("Mês", 1, 12, 6, key='mes_sim')
        
        # Calcular médias móveis simuladas
        acel_media_3h = accel * np.random.uniform(0.9, 1.1)
        acel_media_6h = accel * np.random.uniform(0.85, 1.15)
        
        if st.button("Prever Risco de Falha", key='predict_btn') and model:
            input_data = pd.DataFrame([[
                accel, strain, torque, wind,
                hora, dia_num, mes, acel_media_3h, acel_media_6h
            ]], columns=features)
            
            prob = model.predict_proba(input_data)[0][1] * 100
            
            # Mostrar performance do modelo
            st.metric("Desempenho do Modelo (AUC-ROC)", f"{roc_auc:.2%}")
            
            # Visualização
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidade de Falha"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prob
                    }
                },
                delta={'reference': 50, 'increasing': {'color': "red"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Explicação do modelo
            explain_model(model, features)
            
            # Recomendações baseadas em IA
            st.subheader("🛠️ Recomendações Inteligentes")
            if prob > 80:
                st.markdown('<div class="alert-high">🔴 <strong>ALERTA CRÍTICO</strong><br>'
                           'Probabilidade de falha muito alta. Ações recomendadas:<br>'
                           '- Parada imediata da turbina<br>'
                           '- Inspeção completa de todos os componentes<br>'
                           '- Contatar equipe de manutenção urgente</div>', 
                           unsafe_allow_html=True)
            elif prob > 50:
                st.markdown('<div class="alert-medium">🟠 <strong>ALERTA MODERADO</strong><br>'
                           'Risco elevado de falha. Ações recomendadas:<br>'
                           '- Aumentar frequência de monitoramento<br>'
                           '- Verificar sistema de frenagem<br>'
                           '- Agendar manutenção preventiva nas próximas 48h</div>', 
                           unsafe_allow_html=True)
            else:
                st.success("🟢 **STATUS NORMAL**\n\n"
                         "Operação dentro dos parâmetros esperados. "
                         "Manter monitoramento regular.")

# --- Visualização de Dados Completa ---
def show_complete_dashboard(df, selected_turbine):
    filtered_df = df[df['Turbina'] == selected_turbine]
    
    # KPIs no topo
    st.subheader(f"📊 Métricas da Turbina {selected_turbine}")
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Total de Registros", len(filtered_df))
    
    with cols[1]:
        st.metric("Última Leitura", filtered_df['TimeStamp'].max().strftime('%d/%m %H:%M'))
    
    with cols[2]:
        st.metric("Status Atual", filtered_df['Status'].iloc[0])
    
    with cols[3]:
        st.metric("Tendência Acelerômetro", 
                 f"{filtered_df['Acelerometro'].diff().mean():.2f} m/s²/h",
                 help="Variação média por hora")
    
    # Gráficos temporais
    st.markdown('<div class="header-style">📈 Séries Temporais dos Sensores</div>', unsafe_allow_html=True)
    
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                         subplot_titles=("Acelerômetro (m/s²)", "Strain Gauge (μm/m)", "Torque (Nm)"))
    
    # Acelerômetro com médias móveis
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
    fig.add_trace(
        go.Scatter(
            x=filtered_df['TimeStamp'],
            y=filtered_df['acel_media_3h'],
            name="Média 3h",
            line=dict(color='#2980b9', dash='dot'),
            hovertemplate="Média 3h: %{y:.2f} m/s²<extra></extra>"
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
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# --- Página Principal ---
def main():
    st.title("🌪️ Painel Inteligente de Monitoramento de Turbinas")
    
    # Verificação inicial da conexão
    with st.spinner("Conectando ao banco de dados..."):
        if get_db_connection() is None:
            return
    
    # Carregar dados
    df = load_monitoring_data(90)
    if df.empty:
        st.warning("Nenhum dado encontrado na tabela 'dados'")
        return
    
    # Treinar modelo
    with st.spinner("Treinando modelo preditivo..."):
        model, features, roc_auc = train_advanced_model(df)
    
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
        
        # Simulador de IA
        if model:
            show_ai_simulator(model, features, roc_auc)
    
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
        
        # Análise de Correlação
        st.markdown("### 🔗 Correlação entre Sensores")
        corr_matrix = df[['Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro']].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
