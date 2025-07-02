import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
import joblib
import shap

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
        ORDER BY TimeStamp ASC  # Ordem cronológica para análise temporal
        """
        
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()
        
        df = pd.DataFrame(data, columns=columns)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        
        # Engenharia de features temporal
        df['hora'] = df['TimeStamp'].dt.hour
        df['dia_semana'] = df['TimeStamp'].dt.dayofweek
        df['dia_mes'] = df['TimeStamp'].dt.day
        df['mes'] = df['TimeStamp'].dt.month
        
        # Features de média móvel
        for window in [3, 6, 12]:  # horas
            df[f'acel_media_{window}h'] = df.groupby('Turbina')['Acelerometro'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'strain_media_{window}h'] = df.groupby('Turbina')['StrainGauge'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Target binário com histórico
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
        # Selecionar features
        base_features = ['Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro', 
                        'hora', 'dia_semana', 'dia_mes', 'mes']
        moving_avg_features = [col for col in df.columns if 'media_' in col]
        features = base_features + moving_avg_features
        
        X = df[features]
        y = df['target']
        
        # Divisão temporal (não aleatória)
        tscv = TimeSeriesSplit(n_splits=3)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Pipeline avançado
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(
                RandomForestClassifier(n_estimators=50, random_state=42),
                threshold='median'
            )),
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

# --- Explicação do Modelo com SHAP ---
def explain_model(model, features, sample_data):
    try:
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
            feat_imp = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            st.subheader("📊 Fatores Mais Influentes")
            fig = px.bar(
                feat_imp,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Importância das Variáveis'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("O modelo não fornece importância de características.")
    except Exception as e:
        st.warning(f"Não foi possível gerar explicação: {str(e)}")

# --- Simulador de Falhas Avançado ---
def show_advanced_simulator(model, features, roc_auc):
    st.header("🔮 Simulador de Falhas Inteligente")
    
    with st.expander("Configurar Parâmetros", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            accel = st.slider("Acelerômetro (m/s²)", 0.0, 10.0, 2.5)
            strain = st.slider("Strain Gauge (μm/m)", 0, 3000, 500)
            torque = st.slider("Torque (Nm)", 0, 30000, 12000)
        with col2:
            wind = st.slider("Veloc. Vento (m/s)", 0.0, 30.0, 7.5)
            hora = st.slider("Hora do Dia", 0, 23, 12)
            dia_semana = st.selectbox("Dia da Semana", 
                                    ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
                                    index=0)
        
        dia_num = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"].index(dia_semana)
        dia_mes = st.slider("Dia do Mês", 1, 31, 15)
        mes = st.slider("Mês", 1, 12, 6)
        
        # Calcular médias móveis simuladas
        acel_media_3h = accel * np.random.uniform(0.9, 1.1)
        acel_media_6h = accel * np.random.uniform(0.85, 1.15)
        strain_media_3h = strain * np.random.uniform(0.9, 1.1)
        
        if st.button("Calcular Risco de Falha") and model:
            # Criar dataframe de input
            input_data = pd.DataFrame([[
                accel, strain, torque, wind,
                hora, dia_num, dia_mes, mes,
                acel_media_3h, acel_media_6h,
                strain_media_3h
            ]], columns=features)
            
            # Previsão
            prob = model.predict_proba(input_data)[0][1] * 100
            
            # Mostrar métrica de performance
            st.metric("AUC-ROC do Modelo", f"{roc_auc:.2%}")
            
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
            st.subheader("📊 Explicação da Previsão")
            explain_model(model, features, input_data)
            
            # Sistema de recomendação contextual
            st.subheader("🛠️ Recomendações Inteligentes")
            
            if prob > 80:
                st.error("**Nível Crítico** - Ação Imediata Necessária")
                st.markdown("""
                - 🔴 **Parada de Emergência**: Desligamento imediato da turbina
                - 🔧 **Inspeção Completa**: Verificar todos os componentes mecânicos
                - 📉 **Análise de Vibração**: Teste detalhado de vibração
                - 👨‍🔧 **Equipe Técnica**: Acionar equipe de manutenção urgente
                """)
            elif prob > 50:
                st.warning("**Nível Alto** - Ação Preventiva Recomendada")
                st.markdown("""
                - ⚠️ **Monitoramento Intensivo**: Aumentar frequência de coleta de dados
                - 🔍 **Inspeção Visual**: Verificar parafusos e fixações
                - 📅 **Manutenção Programada**: Agendar para as próximas 72h
                - 📊 **Análise de Tendência**: Verificar histórico de parâmetros
                """)
            elif prob > 20:
                st.info("**Nível Moderado** - Monitoramento Aumentado")
                st.markdown("""
                - 📌 **Verificação Periódica**: Aumentar checklist de manutenção
                - 📈 **Análise de Dados**: Monitorar tendências nos parâmetros
                - 🔔 **Alerta Preventivo**: Notificar equipe de monitoramento
                """)
            else:
                st.success("**Nível Normal** - Operação Regular")
                st.markdown("""
                - ✅ **Monitoramento Rotineiro**: Continuar operação normal
                - 📋 **Checklist Diário**: Manter procedimentos padrão
                - 🔄 **Coleta de Dados**: Manter registro contínuo
                """)

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
    
    # Treinar modelo avançado
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
        
        # Gráfico temporal com anomalias
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
        
        # Adicionar marcadores para falhas
        falhas = filtered_df[filtered_df['Status'] == 'Falha']
        if not falhas.empty:
            fig.add_trace(
                go.Scatter(
                    x=falhas['TimeStamp'],
                    y=falhas['Acelerometro'],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Falha Detectada',
                    hovertext=falhas.apply(
                        lambda row: f"Falha em {row['TimeStamp']}<br>Torque: {row['SensorTorque']}<br>Vento: {row['Anemometro']}",
                        axis=1
                    )
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar simulador avançado
        if model:
            show_advanced_simulator(model, features, roc_auc)
    
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
        
        # Histograma de features importantes
        st.plotly_chart(
            px.histogram(
                df,
                x='Acelerometro',
                color='Status',
                title="Distribuição do Acelerômetro",
                color_discrete_map={
                    'Normal': 'green',
                    'Alerta': 'orange',
                    'Falha': 'red'
                },
                nbins=30
            ),
            use_container_width=True
        )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    main()
