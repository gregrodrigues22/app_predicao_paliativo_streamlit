import streamlit as st
import pandas as pd
import os
import joblib
import h2o
import gdown
import plotly.graph_objects as go
from h2o.estimators import H2OGenericEstimator

# Título do formulário
st.title("Explicação do Modelo")

st.markdown("**Problema**: A transição do cuidado curativo para o paliativo ainda é um desafio na prática clínica, especialmente para pacientes com câncer atendidos em serviços de emergência. Muitos médicos tendem a ser excessivamente otimistas em relação à sobrevida de seus pacientes, o que pode atrasar a recomendação para cuidados paliativos")

st.markdown("**Solução**: Este aplicativo utiliza um modelo de aprendizado de máquina desenvolvido a partir de dados reais de pacientes atendidos no pronto-socorro do Instituto do Câncer do Estado de São Paulo (ICESP). O objetivo é prever a sobrevida de curto prazo (menos de seis meses) e longo prazo (mais de seis meses) desses pacientes a partir de variáveis clínicas e demográficas disponíveis no momento da admissão na emergência.")

st.markdown("**Metodologia**: A metodologia aplicada envolve modelos de aprendizado de máquina, como Gradient Boosting Machine (GBM), para analisar fatores como idade, diagnóstico oncológico, estado clínico na emergência, sinais vitais e o escore funcional (ECOG) da última consulta eletiva. O modelo foi validado e apresentou alto desempenho, demonstrando que é possível identificar, de forma automatizada e com alta precisão, pacientes que podem se beneficiar de uma avaliação mais detalhada para cuidados paliativos.")

#Baixando modelo
try:
    # URL do modelo no Google Drive
    file_id = "1IEGIuHt1l8xwR_Jl5J_fuKf0h5Fkdwx2"  # Substitua pelo ID do seu arquivo
    url = f"https://drive.google.com/uc?id={file_id}"

    # Nome do arquivo para salvar localmente
    model_filename = "modelo_em_mojo.zip"

    # Baixar o modelo
    @st.cache_resource
    def download_model():
        gdown.download(url, model_filename, quiet=False)
        return model_filename
        
    # Baixar o modelo
    download_model()

    # Iniciar o H2O e carregar o modelo
    h2o.init()

    # Carregar o modelo H2O
    model = h2o.import_mojo(model_filename)
    st.session_state['model'] = model
        
    st.write("✅ Modelo carregado com sucesso!")

    model_performance = model.model_performance()
    auc = model_performance.auc()  # Área sob a curva ROC
    logloss = model_performance.logloss()  # Log Loss
    rmse = model_performance.rmse()  # Root Mean Squared Error
    mse = model_performance.mse()  # Mean Squared Error
    r2 = model_performance.r2()  # Coeficiente de determinação R²

    st.subheader("📊 Métricas do Modelo")

    # Obter os valores de FPR e TPR para a curva ROC
    fpr, tpr = model_performance.roc()

    # Criar o gráfico da Curva ROC com Plotly
    fig_roc = go.Figure()

    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Curva ROC (AUC = {auc:.4f})',
        line=dict(color='blue')
    ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Baseline',
        line=dict(dash='dash', color='gray')
    ))
    
    fig_roc.update_layout(
        title="Curva ROC",
        xaxis_title="Taxa de Falsos Positivos",
        yaxis_title="Taxa de Verdadeiros Positivos",
        template="plotly_white"
    )
    
    # Exibir no Streamlit
    st.plotly_chart(fig_roc)
    st.markdown(
    """
    Interpretação da Curva ROC:  
    A Curva ROC (Receiver Operating Characteristic) mostra o desempenho do modelo ao variar o limiar de decisão.  
    - O eixo **X** representa a **Taxa de Falsos Positivos (FPR)**, ou seja, a proporção de negativos que foram incorretamente classificados como positivos.  
    - O eixo **Y** representa a **Taxa de Verdadeiros Positivos (TPR)**, que indica a proporção de positivos corretamente identificados.  
    - A linha pontilhada representa o **modelo aleatório**, enquanto a curva azul representa o modelo preditivo.  
    - O valor de **AUC (Área Sob a Curva)** indica a capacidade do modelo em distinguir as classes.  
      - Um **AUC próximo de 1** indica um modelo altamente discriminativo.  
      - Um **AUC de 0.5** indica um modelo sem capacidade preditiva, equivalente ao acaso.  
      
    Neste caso, o AUC de **0.9123** sugere que o modelo tem um excelente desempenho na diferenciação entre as classes.
    """
)
    st.markdown("**Shap Values**")
    st.image("assets/shap_importance.png", caption="Gráfico de Importância dos Atributos via SHAP")


    st.write(f"🔹 **Log Loss:** {logloss:.4f}")
    st.write(f"🔹 **RMSE:** {rmse:.4f}")
    st.write(f"🔹 **MSE:** {mse:.4f}")
    st.write(f"🔹 **R²:** {r2:.4f}")
        
except Exception as e:
        st.write("❌ Erro ao carregar modelo...", str(e))

