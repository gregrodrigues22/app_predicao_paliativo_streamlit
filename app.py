import streamlit as st
import pandas as pd
import os
import joblib
import h2o
import gdown
import plotly.graph_objects as go
from h2o.estimators import H2OGenericEstimator

st.set_page_config(
        page_title="📈 Predição de Sobrevida",
)

# Título do formulário
st.title("Predição de Sobrevida na Emergência para Pacientes Oncológicos 📈🎯")

st.write(
    "Preencha os campos abaixo com os valores correspondentes às variáveis utilizadas no modelo preditivo."
)

#Criando listas
cid = pd.read_csv("cids.csv")
cid_list = cid['Codigo'].tolist()

status_options = ["Nenhuma das anteriores(Verde)", "Outras situações que requerem atend. com urgência intermediária - (Amarelo)", 
                  "Suspeita/Confirmação de NF - (Amarelo)", "Dor Intensa (> 7 em 10) - (Amarelo)", "Sala de Emergência - (Vermelho)", 
                  "Suspeita de SCM - (Amarelo)",
                  "Sepse - (Amarelo)","Dessaturação - (Amarelo)", "Hemorragia com potencial risco de vida - (Amarelo)", 
                  "Sinais de choque - (Vermelho)", 
                  "Fase Final de Vida - (Amarelo)", "Outras situações que requerem atend. Prioritário - (Vermelho)", "IRA - (Amarelo)", 
                  "Desconforto Respiratório - (Vermelho)", "Distúrbio Hidroeletrolítico com risco de instabilidade - (Amarelo)",
                  "Rebaixamento do Nível de Consciência - (Vermelho)", "Suspeita de SCA - (Vermelho)", "Sangramento Ativo Ameaçador à Vida - (Vermelho)",
                  "Suspeita de Síndrome de Lise Tumoral - (Vermelho)"]

priority_options = ["Verde", "Amarelo", "Vermelho"]

tendency_options = ["Estável", "Instável", "Melhorando"]
 
# Criando o formulário
with st.form(key="input_form"):

    # Seção 1: Informações Pessoais
    st.subheader("Informações Pessoais")
    age = st.number_input("Idade", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Sexo", options=["Masculino", "Feminino"])

    st.markdown("---")  # Linha de separação

    # Seção 2: Sinais Vitais
    st.subheader("Sinais Vitais")
    mbp = st.number_input("Pressão Arterial", min_value=0.0, step=0.1)
    hr = st.number_input("Frequência Cardíaca", min_value=0.0, step=0.1)
    saot = st.number_input("Saturação de Oxigênio", min_value=0.0, step=0.1)

    st.markdown("---")  # Linha de separação

    # Seção 3: Antropometria
    st.subheader("Antropometria")
    missing_bmi = st.checkbox("Ausência de Antropometria")
    height = st.number_input("Altura em centímetros", min_value=0.0, step=0.1)
    weight = st.number_input("Peso em kilogramas", min_value=0.0, step=0.1)
    bmi = st.number_input("Índice de Massa Corporal", min_value=0.0, step=0.1)

    st.markdown("---")  # Linha de separação

    # Seção 4: Diagnóstico e Status
    st.subheader("Diagnóstico e Status")
    icd = st.selectbox("CID", options=cid_list)
    status_original = st.selectbox("Status Original", options=status_options)
    status_priority = st.selectbox("Prioridade", options=priority_options)

    st.markdown("---")  # Linha de separação

    # Seção 5: Histórico Clínico
    st.subheader("Histórico Clínico")
    ti = st.number_input("Tempo entre Última Consulta e PS em dias", min_value=0.0, step=0.1)
    tdr = st.selectbox("Internação Recente", options=["Não", "Sim"])
    tendency = st.selectbox("Tendência", options=tendency_options)

    st.markdown("---")  # Linha de separação

    # Seção 6: Escore Funcional
    st.subheader("Escore Funcional")
    missing_ecog = st.checkbox("Ausência de ECOG")
    ecog = st.number_input("ECOG", min_value=0.0, max_value=4.0, step=0.1)

    st.markdown("---")  # Linha de separação

    # Botão de envio
    submit_button = st.form_submit_button(label="Enviar")

# Exibir os dados submetidos
if submit_button:
    st.success("Dados enviados com sucesso!")
    st.write("Valores inseridos:")
    st.write({
        "Idade": age,
        "Sexo": gender,
        "Pressão Arterial": mbp,
        "Frequência Cardíaca": hr,
        "Saturação de Oxigênio": saot,
        "Altura": height,
        "Peso Estimado": weight,
        "Índice de Massa Corporal": bmi,
        "Ausência de Antropometria": missing_bmi,
        "CID": icd,
        "Status Original": status_original,
        "Prioridade do Status": status_priority,
        "Tempo entre Última Consulta e PS": ti,
        "Internação Recente": tdr,
        "Tendência": tendency,
        "ECOG": ecog,
        "Ausência de ECOG": missing_ecog,
    })

if submit_button:
    # Criar DataFrame a partir dos inputs do usuário
    df_input = pd.DataFrame({
        "age": [age],
        "gender": [gender],  
        "mbp": [mbp],
        "hr": [hr],
        "saot": [saot],
        "height": [height],
        "weight": [weight],
        "bmi": [bmi],
        "icd": icd,  
        "status_original": [status_original],  
        "status_priority": [status_priority],  
        "ti": [ti],  
        "tdr": [tdr],  
        "tendency": [tendency],  
        "ecog": [ecog],  
        "missing_ecog": [missing_ecog],
        "missing_bmi": [missing_bmi]
    })

if submit_button:

    #Print:
    #st.subheader("📊 Dados inseridos pelo usuário")
    #st.dataframe(df_input)

    #Encoding CID
    try:
        encoding_maps = joblib.load("encoding_maps.joblib")
        encoding_maps_cid = encoding_maps["ICD"]
        #st.write("✅ Preparando input do Dado 'CID' para predição...")
        df_input["icd_processed"] = df_input["icd"].str.split(" - ").str[0].str.lower()
        df_input["icd_encoded"] = df_input["icd_processed"].map(encoding_maps_cid).fillna(0.34162670016104163)
        #st.write(df_input["icd_encoded"])
    except:
        st.write("❌ Erro ao preparar Dado 'CID' para predição...")

    #Encoding Status_Original
    try:
        encoding_maps_status = encoding_maps["Status_Original"]
        #st.write("✅ Preparando input do Dado 'Status' para predição...")
        df_input["status_original_encoded"] = df_input["status_original"].map(encoding_maps_status).fillna(0.31209494163715695)
        #st.write(df_input["status_original_encoded"])
    except:
        st.write("❌ Erro ao preparar Dado 'Status' para predição...")

    #Encoding Status_Ordinal
    try:
        priority_encoding = {"Verde": 1,"Amarelo": 2,"Vermelho": 3}
        #st.write("✅ Preparando input do Dado 'Prioridade' para predição...")
        df_input["status_priority_encoded"] = df_input["status_priority"].map(priority_encoding).fillna(1)
        #st.write(df_input["status_priority_encoded"])
    except:
        st.write("❌ Erro ao preparar Dado 'Prioridade' para predição...")      

    #Encoding Sexo
    try:
        gender_encoding = {"Feminino": 1,"Masculino": 0}
        #st.write("✅ Preparando input do Dado 'Sexo' para predição...")
        df_input["gender_encoded"] = df_input["gender"].map(gender_encoding).fillna(1)
        #st.write(df_input["gender_encoded"])
    except:
        st.write("❌ Erro ao preparar Dado 'Sexo' para predição...")    

    #Encoding TDR
    try:
        tdr_encoding = {"Não": 0,"Sim": 1}
        #st.write("✅ Preparando input do Dado 'Reinternação' para predição...")
        df_input["tdr_encoded"] = df_input["tdr"].map(tdr_encoding).fillna(0)
        #st.write(df_input["tdr_encoded"])
    except:
        st.write("❌ Erro ao preparar Dado 'Reinternação' para predição...")  

    #Encoding Tendência
    try:
        tendency_encoding = {"Estável": 1,"Melhorando": 2, "Instável": 3}
        #st.write("✅ Preparando input do Dado 'Tendência' para predição...")
        df_input["tendency_encoded"] = df_input["tendency"].map(tendency_encoding).fillna(1)
        #st.write(df_input["tendency_encoded"])
    except:
        st.write("❌ Erro ao preparar Dado 'Tendência' para predição...")  

    #Encoding missing_ecog
    try:
        missing_ecog_encoding = {"False": 0,"True": 1}
        #st.write("✅ Preparando input do Dado 'Ecog' para predição...")
        df_input["missing_ecog_encoded"] = df_input["missing_ecog"].astype(str).map(missing_ecog_encoding).fillna(1)
        #st.write(df_input["missing_ecog_encoded"])
    except:
        st.write("❌ Erro ao preparar Dado 'Ecog' para predição...")

    #Encoding missing_bmi
    try:
        missing_bmi_encoding = {"False": 0,"True": 1}
        #st.write("✅ Preparando input do Dado 'IMC' para predição...")
        df_input["missing_bmi_encoded"] = df_input["missing_bmi"].astype(str).map(missing_bmi_encoding).fillna(1)
        #    st.write(df_input["missing_bmi_encoded"])
    except:
        st.write("❌ Erro ao preparar Dado 'IMC' para predição...")

    #Encoding ti e os 
    try:
        df_input["ti_segundos"] = df_input["ti"] * 86400
        df_input["saot_fracao"] = df_input["saot"] /100
        #st.write("✅ Preparando input do Dado 'Tempo entre Última Consulta e PS' para predição...")
    except:
        st.write("❌ Erro ao preparar Dado 'Tempo entre Última Consulta e PS' para predição...")

    #Scaler
    try:
        scaler = joblib.load("scaler.joblib")
        print(type(scaler))
        #st.write("✅ Preparando normalização dos dados...")
    except Exception as e:
        print("❌ Erro ao preparar normalização dados...", str(e))

    #Mapeamento para Scaler
    try:
        selected_columns = [
        'bmi', 'hr', 'mbp', 'saot_fracao', 'weight',
        'height', 'ti_segundos', 'ecog', 'missing_bmi_encoded',
        'missing_ecog_encoded', 'gender_encoded', 'tdr_encoded', 'tendency_encoded',
        'age', 'status_priority_encoded', 'icd_encoded', 'status_original_encoded']
        df_input = df_input.filter(items=selected_columns)
        column_scale_mapping = {
        "bmi": "BMI_knn",
        "hr": "HR_knn",
        "mbp": "MBP_knn",
        "saot_fracao": "OS_knn",
        "weight": "Weight_knn",
        "height": "Height_knn",
        "ti_segundos": "TI_median",
        "ecog": "ECOG_median",
        "missing_bmi_encoded": "missing_bmi",
        "missing_ecog_encoded": "missing_ecog",
        "gender_encoded": "Gender_binary",
        "tdr_encoded": "TDR_binary",
        "tendency_encoded": "Tendency_ordinal",
        "age": "Age",
        "status_priority_encoded": "Status_priority",
        "icd_encoded": "ICD",
        "status_original_encoded": "Status_Original"
        }
        df_input = df_input.rename(columns=column_scale_mapping)
        #st.write("✅ Preparando nomenclatura dos dados...")
    except Exception as e:
        print("❌ Erro ao preparar nomenclatura das variáveis dados...", str(e))

    #Aplicando scaler
    try:
        df_input_scaled = df_input.copy()
        expected_columns = ['BMI_knn', 'HR_knn', 'MBP_knn', 'OS_knn', 'Weight_knn','Height_knn', 
                            'TI_median', 'ECOG_median', 'missing_bmi','missing_ecog', 'Gender_binary', 
                            'TDR_binary', 'Tendency_ordinal','Age', 'Status_priority', 'ICD', 'Status_Original']
        df_input_scaled.columns = df_input_scaled.columns.astype(str)
        df_input_scaled = df_input_scaled.astype(float)
        df_input_scaled.columns = df_input_scaled.columns.str.strip()
        #st.write("Colunas no df_input_scaled:", df_input_scaled.columns.tolist())
        #st.write("Colunas esperadas pelo scaler:", expected_columns)
        #st.write(df_input_scaled)
        #print("No DataFrame mas não no Scaler:", set(df_input_scaled.columns) - set(expected_columns))
        #print("No Scaler mas não no DataFrame:", set(expected_columns) - set(df_input_scaled.columns))
        #st.write(df_input_scaled.dtypes)
        df_input_scaled[expected_columns] = scaler.transform(df_input_scaled[expected_columns])
        #st.write(df_input_scaled)
        #st.write("✅ Realizando normalização dos dados...")
    except Exception as e:
        st.write("❌ Erro ao normalizar dados...", str(e))

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
    except Exception as e:
        st.write("❌ Erro ao carregar modelo...", str(e))

    #Mostrando predição
    
    try:
        h2o_df = h2o.H2OFrame(df_input_scaled)
        predictions = model.predict(h2o_df)
        predictions_df = predictions.as_data_frame()
        #st.write(predictions_df)
        #st.write("✅ Predição realizada com sucesso!")
        
        st.markdown("### Resultado da Predição:")
        # Definir os limiares para cada classe
        limiar_classe_0 = 0.4283  # Ajuste conforme necessário
        limiar_classe_1 = 0.5717  # Ajuste conforme necessário
        
        fig = go.Figure()
        
        # Adicionar as barras para as probabilidades das classes
        fig.add_trace(go.Bar(
            x=['Classe 0 - Longa Sobrevida', 'Classe 1 - Baixa Sobrevida'],
            y=[predictions_df['p0'][0], predictions_df['p1'][0]],  # Probabilidades
            marker=dict(color=['green', 'red']),  # Cores das classes
            text=[f"{predictions_df['p0'][0]:.2%}", f"{predictions_df['p1'][0]:.2%}"],  # Adicionar percentuais
            textposition='auto',
        ))
        
        # Adicionar linha do limiar **para a Classe 0** (linha horizontal completa)
        fig.add_trace(go.Scatter(
            x=['Classe 0 - Longa Sobrevida', 'Classe 1 - Baixa Sobrevida'],  # Agora atravessa todo o gráfico
            y=[limiar_classe_0, limiar_classe_0],  # Linha no valor do limiar da classe 0
            mode="lines",
            line=dict(color="blue", dash="dash"),
            name=f"Limiar Classe 0 ({limiar_classe_0:.2%})"
        ))
        
        # Adicionar linha do limiar **para a Classe 1** (linha horizontal completa)
        fig.add_trace(go.Scatter(
            x=['Classe 0 - Longa Sobrevida', 'Classe 1 - Baixa Sobrevida'],  # Agora atravessa todo o gráfico
            y=[limiar_classe_1, limiar_classe_1],  # Linha no valor do limiar da classe 1
            mode="lines",
            line=dict(color="purple", dash="dash"),
            name=f"Limiar Classe 1 ({limiar_classe_1:.2%})"
        ))
        
        # Ajustar layout
        fig.update_layout(
            title="Distribuição das Probabilidades das Classes",
            xaxis_title="Classes",
            yaxis_title="Probabilidade",
            yaxis=dict(range=[0, 1]),  # Garante que o eixo Y vai de 0 a 1
            showlegend=True  # Exibe as legendas
        )

        # Exibir no Streamlit
        st.plotly_chart(fig)

        if predictions_df['predict'][0] == 1:
            st.success("A Classe 1 - Baixa Sobrevida - foi predita com probabilidade maior que o limiar de 57,17%. Portanto, classe predita para o paciente em questão é de Baixa Sobrevida") 
        else:
            st.warning("A Classe 0 - Longa Sobrevida - foi predita com probabilidade maior que o limiar de 42,83%. Portanto, classe predita para o paciente em questão é de Longa Sobrevida")

        st.markdown("### Explicação da Predição:")
        
        # Gerar a explicação SHAP apenas para essa instância
        shap_values = model.predict_contributions(h2o_df)
        shap_df = shap_values.as_data_frame()
        shap_df = shap_df.drop(columns=["BiasTerm"], errors="ignore")
        shap_df_melted = shap_df.T.reset_index()  # Transpõe e reseta o índice
        shap_df_melted.columns = ["Feature", "Importance"]  # Renomeia colunas
        
        rename_dict = {
            "BMI_knn": "IMC",
            "HR_knn": "Frequência Cardíaca",
            "MBP_knn": "Pressão Média",
            "OS_knn": "Saturação de Oxigênio",
            "Weight_knn": "Peso",
            "Height_knn": "Altura",
            "TI_median": "Tempo entre Consulta e PS",
            "ECOG_median": "ECOG",
            "missing_bmi": "IMC ausente",
            "missing_ecog": "ECOG ausente",
            "Gender_binary": "Sexo",
            "TDR_binary": "Internação Recente",
            "Tendency_ordinal": "Tendência",
            "Age": "Idade",
            "Status_priority": "Prioridade Status",
            "ICD": "CID",
            "Status_Original": "Status Original"
        }
        
        # Aplicar os novos nomes das features
        shap_df_melted["Feature"] = shap_df_melted["Feature"].replace(rename_dict)

        # Ordenar pela importância absoluta (valores mais influentes primeiro)
        shap_df_melted = shap_df_melted.reindex(shap_df_melted["Importance"].abs().sort_values(ascending=True).index)
        
        # Adicionar barras ao gráfico
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=shap_df_melted["Feature"],  # Eixo Y com os nomes das features
            x=shap_df_melted["Importance"],  # Eixo X com a importância SHAP
            orientation='h',  # Gráfico de barras horizontais
            marker=dict(color=shap_df_melted["Importance"], colorscale="RdBu"),  # Cores para facilitar leitura
        ))
        
        # Ajustar layout do gráfico
        fig.update_layout(
            title="Explicação da Predição (SHAP) - Paciente em avaliação",
            xaxis_title="Importância SHAP",
            yaxis_title="Feature",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=False),
        )
        
        # Exibir no Streamlit
        st.plotly_chart(fig)
    
    except Exception as e:
        st.write("❌ Erro ao realizar predição...", str(e))


