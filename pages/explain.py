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
    st.write(f"🔹 **AUC:** {auc:.4f}")
    st.write(f"🔹 **Log Loss:** {logloss:.4f}")
    st.write(f"🔹 **RMSE:** {rmse:.4f}")
    st.write(f"🔹 **MSE:** {mse:.4f}")
    st.write(f"🔹 **R²:** {r2:.4f}")
        
except Exception as e:
        st.write("❌ Erro ao carregar modelo...", str(e))

