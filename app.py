import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Monitoramento Inteligente de Turbinas",
    page_icon="🌪️",
    layout="wide"
)

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
        ORDER BY TimeStamp ASC
        """
        
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()
        
        df = pd.DataFrame(data, columns=columns)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        
        # Engenharia de features
        df['hora'] = df['TimeStamp'].dt.hour
        df['dia_semana'] = df['TimeStamp'].dt.dayofweek
        
        # Target binário
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

# --- Modelo Preditivo Aprimorado ---
def train_advanced_model(df):
    try:
        # Features selecionadas
        features = ['Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro', 'hora', 'dia_semana']
        
        X = df[features]
        y = df['target']
        
        # Divisão temporal
        tscv = TimeSeriesSplit(n_splits=3)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Pipeline simplificado
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

# --- Explicação do Modelo sem SHAP ---
def explain_model(model, features):
    try:
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            st.subheader("📊 Fatores Mais Importantes")
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Contribuição de Cada Variável'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("O modelo não fornece importância de características.")
    except Exception as e:
        st.warning(f"Não foi possível explicar o modelo: {str(e)}")

# --- Simulador de Falhas ---
def show_failure_simulator(model, features, roc_auc):
    st.header("🔮 Simulador de Risco de Falhas")
    
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
        
        if st.button("Calcular Risco de Falha") and model:
            input_data = pd.DataFrame([[
                accel, strain, torque, wind, hora, dia_num
            ]], columns=features)
            
            prob = model.predict_proba(input_data)[0][1] * 100
            
            # Mostrar performance do modelo
            st.metric("Desempenho do Modelo (AUC-ROC)", f"{roc_auc:.2%}")
            
            # Visualização
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
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
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Explicação do modelo
            explain_model(model, features)
            
            # Recomendações
            st.subheader("🛠️ Recomendações de Ação")
            if prob > 80:
                st.error("**Nível Crítico** - Ação Imediata Necessária")
                st.markdown("""
                - 🔴 Parada de emergência imediata
                - 🔧 Inspeção completa dos componentes
                - 📉 Análise detalhada de vibração
                """)
            elif prob > 50:
                st.warning("**Nível Alto** - Ação Preventiva Recomendada")
                st.markdown("""
                - ⚠️ Aumentar frequência de monitoramento
                - 🔍 Verificar parafusos e fixações
                - 📅 Agendar manutenção preventiva
                """)
            else:
                st.success("**Nível Normal** - Operação Regular")
                st.markdown("""
                - ✅ Continuar monitoramento rotineiro
                - 📋 Manter checklist de manutenção
                """)

# --- Página Principal ---
def main():
    st.title("🌪️ Painel de Monitoramento de Turbinas")
    
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
        
        # Destacar falhas
        falhas = filtered_df[filtered_df['Status'] == 'Falha']
        if not falhas.empty:
            fig.add_trace(
                go.Scatter(
                    x=falhas['TimeStamp'],
                    y=falhas['Acelerometro'],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Falha Detectada'
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar simulador
        if model:
            show_failure_simulator(model, features, roc_auc)
    
    with col2:
        # Estatísticas rápidas
        st.metric("Total de Registros", len(df))
        st.metric("Turbinas Monitoradas", df['Turbina'].nunique())
        st.metric("Falhas Detectadas", df['Status'].value_counts().get('Falha', 0))
        
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
