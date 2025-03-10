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

    st.markdown(
    """
    O gráfico de importância dos atributos via SHAP mostra como cada variável contribui para as previsões do modelo. Cada ponto no gráfico representa um valor SHAP para uma observação do conjunto de dados.

    Elementos do Gráfico
    
    1. Eixo Y - Nome das variáveis:
    Listadas do topo para a base, em ordem de importância.
    Quanto mais alto na lista, maior a influência da variável na predição do modelo.
    
    2. Eixo X - Valor SHAP:
    Indica a magnitude e a direção do impacto de cada variável na predição.
    Valores positivos deslocam a predição para maior probabilidade da classe-alvo.
    Valores negativos deslocam a predição para menor probabilidade da classe-alvo.
    
    3. Cores - Valor Normalizado da Variável:
    Azul: Valores baixos da variável.
    Rosa: Valores altos da variável.
    Isso ajuda a entender a relação entre o valor da variável e sua influência no modelo.
    
    4. Como interpretar?
    Variáveis com maior dispersão horizontal (ou seja, uma grande variação nos valores SHAP) indicam maior impacto nas predições do modelo.
    Se os pontos de uma variável estiverem predominantemente na direita (valores SHAP positivos), essa variável aumenta a chance da classe predita.
    Se os pontos estiverem na esquerda (valores SHAP negativos), essa variável reduz a chance da classe predita.
    Sobreposição de cores indica que a relação da variável com a predição pode ser complexa, não apenas linear.

    5. Aplicação do modelo
    >• ECOG : Interpretação esperada: O ECOG é uma medida importante para prever a sobrevida dos pacientes oncológicos. Valores mais altos indicam pior estado funcional do paciente, o que está fortemente associado a menor sobrevida. Concordância: Faz sentido que o ECOG seja a variável mais importante no modelo. Isso está de acordo com a literatura médica, que sugere que o estado funcional é um dos melhores preditores de prognóstico em pacientes oncológicos.
    
    >• Internação recente: Interpretação esperada: Internações recentes podem indicar uma piora significativa no quadro clínico do paciente, o que está diretamente relacionado com uma menor sobrevida. Concordância: Faz sentido que essa variável seja importante no modelo, pois pacientes que foram internados recentemente tendem a ter um estado de saúde mais grave.
    
    >• ICD: Refere-se ao código da condição clínica primária do paciente. A ampla distribuição de valores SHAP sugere que diferentes códigos ICD têm pesos distintos na decisão do modelo.
    
    >• Frequência Cardíaca: Interpretação esperada: A frequência cardíaca é um indicador importante do estado geral de saúde. Valores anormais podem indicar complicações que afetam o prognóstico. Concordância: A presença de HR_knn como uma variável importante está alinhada com o fato de que sinais vitais podem ajudar a identificar pacientes em risco.

    >• Peso e Altura: Interpretação esperada: Como discutido anteriormente, o peso e a altura individuais são esperados como menos importantes, já que o IMC já captura essa informação. Surpresa: Essas variáveis ainda aparecem como relevantes no gráfico, sugerindo que há um impacto adicional de peso e altura que o IMC sozinho não explica.

    >• Valores Ausentes de ECOG e IMC: Interpretação esperada: As variáveis missing_ecog e missing_bmi indicam ausência de dados. A expectativa é que essas variáveis não tivessem um impacto tão grande. Surpresa: O impacto de missing_ecog é maior do que esperado. Isso sugere que a ausência do ECOG pode ser um indicador de pacientes em pior estado (por exemplo, pacientes que não completaram avaliações devido à gravidade da doença).

    >• Idade: Interpretação esperada: Idade é uma variável fundamental em modelos de prognóstico oncológico. Em geral, pacientes mais idosos tendem a ter um prognóstico pior, especialmente em pacientes com câncer avançado. Concordância: Isso está alinhado com a literatura médica: idade avançada é um fator de risco para pior prognóstico em pacientes oncológicos. Faz sentido que o modelo esteja capturando essa informação corretamente.

    >• Tempo entre Última Consulta e PS: Pacientes que vão ao PS logo após uma consulta ambulatorial podem estar fazendo isso porque tiveram um agravamento rápido do quadro clínico. Isso pode indicar que o paciente estava estável na consulta, mas piorou rapidamente, o que pode ser um indicativo de baixa sobrevida. Pacientes que só vão ao PS após um longo intervalo sem consultas podem ser aqueles que estavam lidando bem com suas condições crônicas por mais tempo. Esses pacientes podem ter ido ao PS por motivos não emergenciais ou em uma situação menos crítica. Baixo TI_median está associado a pior prognóstico, provavelmente porque esses pacientes tiveram um agravamento rápido após a última consulta. Alto TI_median está associado a melhor prognóstico, provavelmente porque esses pacientes não apresentaram complicações graves por um longo período.

    """
)
    
except Exception as e:
        st.write("❌ Erro ao carregar modelo...", str(e))

