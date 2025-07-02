import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
from sklearn.pipeline import Pipeline

# Configuração da página
st.set_page_config(
    page_title="Monitoramento Avançado de Turbinas",
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

# --- Carregar Dados ---
@st.cache_data(ttl=300)
def load_monitoring_data(days=90):  # Aumentado para 90 dias para mais dados
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
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

# --- Pré-processamento Avançado ---
def preprocess_data(df):
    try:
        # Criar features temporais
        df['hora'] = df['TimeStamp'].dt.hour
        df['dia_semana'] = df['TimeStamp'].dt.dayofweek
        df['dia_mes'] = df['TimeStamp'].dt.day
        df['mes'] = df['TimeStamp'].dt.month
        
        # Criar features de média móvel
        df['acel_media_3h'] = df.groupby('Turbina')['Acelerometro'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Criar target binário (1 para Falha, 0 para outros)
        df['target'] = df['Status'].apply(lambda x: 1 if x == 'Falha' else 0)
        
        # Features selecionadas
        features = [
            'Acelerometro', 'StrainGauge', 'SensorTorque', 'Anemometro',
            'hora', 'dia_semana', 'dia_mes', 'mes', 'acel_media_3h'
        ]
        
        return df, features
    except Exception as e:
        st.error(f"Erro no pré-processamento: {str(e)}")
        return None, None

# --- Treinar e Avaliar Modelo ---
def train_and_evaluate_model(df, features):
    try:
        # Filtrar dados e separar features/target
        X = df[features]
        y = df['target']
        
        # Balanceamento de classes com SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Divisão treino-teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
        )
        
        # Pipeline com normalização e modelo
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
        
        # Treinamento
        pipeline.fit(X_train, y_train)
        
        # Avaliação
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Métricas
        accuracy = pipeline.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(pipeline, X_res, y_res, cv=5, scoring='roc_auc')
        
        # Relatório de classificação
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return pipeline, {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    except Exception as e:
        st.error(f"Erro no treinamento: {str(e)}")
        return None, None

# --- Visualização de Métricas ---
def show_metrics(metrics):
    st.subheader("📊 Métricas do Modelo")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia", f"{metrics['accuracy']:.2%}")
    col2.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")
    col3.metric("Validação Cruzada (Média ± DP)", 
               f"{metrics['cv_mean']:.2%} ± {metrics['cv_std']:.2%}")
    
    st.subheader("Relatório de Classificação")
    report_df = pd.DataFrame(metrics['report']).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))
    
    st.subheader("Matriz de Confusão")
    fig = px.imshow(
        metrics['confusion_matrix'],
        labels=dict(x="Previsto", y="Real", color="Casos"),
        x=['Normal', 'Falha'],
        y=['Normal', 'Falha'],
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Simulador de Falhas Aprimorado ---
def show_failure_simulator(model, features):
    st.header("🔮 Simulador de Falhas Avançado")
    
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
        
        # Calcular média móvel simulada
        acel_media_3h = accel * np.random.uniform(0.9, 1.1)
        
        if st.button("Calcular Probabilidade de Falha") and model:
            input_data = [[
                accel, strain, torque, wind,
                hora, dia_num, dia_mes, mes, acel_media_3h
            ]]
            
            # Obter probabilidade e classe prevista
            prob = model.predict_proba(input_data)[0][1] * 100
            pred_class = model.predict(input_data)[0]
            
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
            st.subheader("📌 Explicação da Previsão")
            
            if pred_class == 1:
                st.error("**Predição:** Risco de Falha Detectado")
            else:
                st.success("**Predição:** Operação Normal")
            
            # Mostrar importância das features
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                st.subheader("📈 Fatores Mais Influentes")
                importances = model.named_steps['model'].feature_importances_
                feat_imp = pd.DataFrame({
                    'Feature': features,
                    'Importância': importances
                }).sort_values('Importância', ascending=False)
                
                fig = px.bar(
                    feat_imp,
                    x='Importância',
                    y='Feature',
                    orientation='h',
                    title='Importância das Variáveis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recomendações contextuais
            st.subheader("🛠️ Recomendações de Ação")
            
            if prob > 80:
                st.error("**Ação Imediata Necessária**")
                st.markdown("""
                - **Parada de Emergência**: Desligar turbina imediatamente
                - **Inspeção Completa**: Verificar todos os componentes mecânicos
                - **Análise de Vibração**: Realizar teste de vibração detalhado
                - **Contatar Engenharia**: Notificar equipe técnica
                """)
            elif prob > 50:
                st.warning("**Ação Preventiva Recomendada**")
                st.markdown("""
                - **Monitoramento Intensivo**: Aumentar frequência de coleta de dados
                - **Inspeção Visual**: Verificar parafusos e fixações
                - **Plano de Contingência**: Preparar procedimento de parada
                - **Manutenção Programada**: Agendar para breve
                """)
            else:
                st.success("**Operação Normal**")
                st.markdown("""
                - **Monitoramento Rotineiro**: Continuar operação normal
                - **Verificação Periódica**: Manter checklist de manutenção
                - **Coleta de Dados**: Manter registro para análise futura
                """)

# --- Página Principal ---
def main():
    st.title("🌪️ Painel de Monitoramento Avançado de Turbinas")
    
    # Verificação inicial da conexão
    with st.spinner("Conectando ao banco de dados..."):
        if get_db_connection() is None:
            return
    
    # Carregar dados
    df = load_monitoring_data(90)  # Carrega 90 dias de dados
    if df.empty:
        st.warning("Nenhum dado encontrado na tabela 'dados'")
        return
    
    # Pré-processamento
    df_processed, features = preprocess_data(df)
    if df_processed is None:
        return
    
    # Treinar e avaliar modelo
    model, metrics = train_and_evaluate_model(df_processed, features)
    if model is None:
        return
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Mostrar métricas do modelo
        show_metrics(metrics)
        
        # Filtros para visualização
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
        
        # Mostrar simulador aprimorado
        show_failure_simulator(model, features)
    
    with col2:
        # Estatísticas rápidas
        st.metric("Total de Registros", len(df))
        st.metric("Turbinas Monitoradas", df['Turbina'].nunique())
        st.metric("Falhas Registradas", df['Status'].value_counts().get('Falha', 0))
        
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
                title="Distribuição do Acelerômetro por Status",
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
    main()
