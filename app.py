# ==============================================================
# Set up
# ==============================================================
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import os
import joblib
import h2o
import gdown
import plotly.graph_objects as go
from h2o.estimators import H2OGenericEstimator

# --------------------------------------------------------------
# Configuração da página
# --------------------------------------------------------------
st.set_page_config(
    page_title="📈 Predição de Sobrevida",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------------------
# Utilitário: localizar primeiro arquivo existente (logo/foto)
# --------------------------------------------------------------
ASSETS = Path("assets")

def first_existing(*names, base=ASSETS):
    for n in names:
        p = base / n if base else Path(n)
        if p.exists():
            return p
    return None

LOGO = first_existing("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")

# --------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------
st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428); 
                padding: 36px; border-radius: 14px; margin-bottom:28px'>
        <h1 style='color: white; margin: 0;'>📊 Predição de Sobrevida na Urgência-Emergência</h1>
        <p style='color: #e8eef7; margin: 8px 0 0 0; font-size: 1.05rem;'>
            Explore a predição para tomada de decisão no point of care.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Esconde a lista padrão de páginas no topo da sidebar
st.markdown(
    """
    <style>
      [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# Sidebar 
# --------------------------------------------------------------

with st.sidebar:
    if LOGO:
        st.image(str(LOGO), use_container_width=True)
    else:
        st.warning("Logo não encontrada em assets/.")
    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.header("Menu")

    # Se estiver em app multipágina, esses page_links funcionam nativamente.
    with st.expander("Predição no PoC", expanded=True):
        # Link para a própria página (opcional em multipage)
        st.page_link("app.py", label="Predição de Sobrevida", icon="📈")

    # Se estiver em app multipágina, esses page_links funcionam nativamente.
    with st.expander("Explicação do Modelo", expanded=True):
        # Link para a própria página (opcional em multipage)
        st.page_link("pages/explain.py", label="Explicação do Modelo", icon="📙")

    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)

    st.subheader("Conecte-se")
    st.markdown(
        """
        - 💼 [LinkedIn](https://www.linkedin.com/in/gregorio-healthdata/)
        - ▶️ [YouTube](https://www.youtube.com/@Patients2Python)
        - 📸 [Instagram](https://www.instagram.com/patients2python/)
        - 🌐 [Site](https://patients2python.com.br/)
        - 🐙 [GitHub](https://github.com/gregrodrigues22)
        - 👥💬 [Comunidade](https://chat.whatsapp.com/CBn0GBRQie5B8aKppPigdd)
        - 🤝💬 [WhatsApp](https://patients2python.sprinthub.site/r/whatsapp-olz)
        - 🎓 [Escola](https://app.patients2python.com.br/browse)
        """,
        unsafe_allow_html=True
        )

# --------------------------------------------------------------
# CONTEÚDO PRINCIPAL DO APP (AJUSTADO + MODELO EM CACHE)
# --------------------------------------------------------------

st.title("Predição no PoC 📈🎯")
st.write("Preencha os campos abaixo com os valores correspondentes às variáveis utilizadas no modelo preditivo.")

# ---------- carregadores em cache ----------
@st.cache_data(show_spinner=False)
def load_cids():
    df = pd.read_csv("cids.csv")
    cols_lower = {c.lower(): c for c in df.columns}
    codigo_col = cols_lower.get("codigo")
    desc_col   = cols_lower.get("descricao")
    if codigo_col and desc_col:
        options = (df[codigo_col].astype(str) + " - " + df[desc_col].astype(str)).tolist()
    elif codigo_col:
        options = df[codigo_col].astype(str).tolist()
    else:
        options = df.get("Codigo", pd.Series(dtype=str)).astype(str).tolist()
    return df, options

@st.cache_data(show_spinner=False)
def load_encoding_maps():
    return joblib.load("encoding_maps.joblib")

@st.cache_data(show_spinner=False)
def load_scaler():
    return joblib.load("scaler.joblib")

# ---------- MODELO H2O EM CACHE ----------
@st.cache_resource(show_spinner=True)
def get_model():
    """
    Baixa o MOJO (se necessário), inicializa o H2O e carrega o modelo.
    Executa apenas uma vez por sessão graças ao cache_resource.
    """
    file_id = "1IEGIuHt1l8xwR_Jl5J_fuKf0h5Fkdwx2"  # ajuste se mudar
    url = f"https://drive.google.com/uc?id={file_id}"
    model_filename = "modelo_em_mojo.zip"

    # Baixa apenas se não existir localmente
    if not os.path.exists(model_filename):
        gdown.download(url, model_filename, quiet=True)

    # Inicia H2O (chamado uma vez por sessão)
    h2o.init()

    # Importa o MOJO
    model = h2o.import_mojo(model_filename)
    return model

cid_df, cid_options = load_cids()

status_options = [
    "Nenhuma das anteriores(Verde)",
    "Outras situações que requerem atend. com urgência intermediária - (Amarelo)",
    "Suspeita/Confirmação de NF - (Amarelo)",
    "Dor Intensa (> 7 em 10) - (Amarelo)",
    "Sala de Emergência - (Vermelho)",
    "Suspeita de SCM - (Amarelo)",
    "Sepse - (Amarelo)",
    "Dessaturação - (Amarelo)",
    "Hemorragia com potencial risco de vida - (Amarelo)",
    "Sinais de choque - (Vermelho)",
    "Fase Final de Vida - (Amarelo)",
    "Outras situações que requerem atend. Prioritário - (Vermelho)",
    "IRA - (Amarelo)",
    "Desconforto Respiratório - (Vermelho)",
    "Distúrbio Hidroeletrolítico com risco de instabilidade - (Amarelo)",
    "Rebaixamento do Nível de Consciência - (Vermelho)",
    "Suspeita de SCA - (Vermelho)",
    "Sangramento Ativo Ameaçador à Vida - (Vermelho)",
    "Suspeita de Síndrome de Lise Tumoral - (Vermelho)"
]
tendency_options = ["Estável", "Instável", "Melhorando"]

# ---------- FORMULÁRIO (reordenado + componentes pedidos) ----------
with st.form(key="input_form"):

    # 1) Idade (slider)
    age = st.slider("Idade (anos)", min_value=0, max_value=120, value=60, step=1)

    # 2) Sexo (radio)
    gender = st.radio("Sexo", options=["Masculino", "Feminino"], horizontal=True)

    # 3) Pressão Arterial Média (slider)
    mbp = st.slider("Pressão Arterial Média (mmHg)", min_value=40, max_value=140, value=90, step=1)

    # 4) Frequência Cardíaca (slider)
    hr = st.slider("Frequência Cardíaca (bpm)", min_value=30, max_value=200, value=90, step=1)

    # 5) Saturação de Oxigênio (slider)
    saot = st.slider("Saturação de Oxigênio (%)", min_value=50, max_value=100, value=97, step=1)

    st.markdown("---")

    # 6) Antropometria (condicional)
    st.subheader("Antropometria")
    missing_bmi = st.checkbox("Ausência de Antropometria")
    if not missing_bmi:
        height = st.slider("Altura (cm)", min_value=120, max_value=220, value=170, step=1)
        weight = st.slider("Peso (kg)", min_value=30, max_value=200, value=70, step=1)
        bmi = round(weight / ((height / 100) ** 2), 1)
        st.caption(f"IMC calculado automaticamente: **{bmi} kg/m²**")
    else:
        height, weight, bmi = 0.0, 0.0, 0.0

    st.markdown("---")

    # 7) Status Original (select)
    status_original = st.selectbox("Status Original (classificação clínica)", options=status_options)

    # 8) Prioridade (cor → valor)
    prioridade_color = st.radio(
        "Prioridade (cor)", options=["🟢 Verde", "🟡 Amarelo", "🔴 Vermelho"],
        index=1, horizontal=True
    )
    priority_map_display_to_value = {
        "🟢 Verde": "Verde",
        "🟡 Amarelo": "Amarelo",
        "🔴 Vermelho": "Vermelho",
    }
    status_priority = priority_map_display_to_value[prioridade_color]

    # 9) Tendência (radio)
    tendency = st.radio("Tendência clínica", options=tendency_options, horizontal=True)

    # 10) CID (melhor nomenclatura)
    icd = st.selectbox("CID-10 (Código – Descrição)", options=cid_options)

    # 11) ECOG (condicional; radio 0–4)
    st.subheader("Escore Funcional (ECOG)")
    missing_ecog = st.checkbox("Ausência de ECOG")
    if not missing_ecog:
        ecog = st.radio("ECOG", options=[0, 1, 2, 3, 4], index=0, horizontal=True)
    else:
        ecog = 0.0

    # 12) Tempo entre última consulta e PS (slider dias)
    ti = st.slider("Tempo entre Última Consulta e PS (dias)", min_value=0, max_value=365, value=7, step=1)

    # 13) Internação Recente (radio)
    tdr = st.radio("Internação Recente", options=["Não", "Sim"], horizontal=True)

    st.markdown("---")
    submit_button = st.form_submit_button(label="Enviar")

# ---------- Exibição e DF de entrada ----------
if submit_button:
    st.success("Dados enviados com sucesso!")
    st.write({
        "Idade": age,
        "Sexo": gender,
        "Pressão Arterial (MBP)": mbp,
        "Frequência Cardíaca (HR)": hr,
        "Saturação de Oxigênio (%)": saot,
        "Ausência de Antropometria": missing_bmi,
        "Altura (cm)": height,
        "Peso (kg)": weight,
        "IMC (auto)": bmi,
        "Status Original": status_original,
        "Prioridade (cor)": status_priority,
        "Tendência": tendency,
        "CID-10": icd,
        "Ausência de ECOG": missing_ecog,
        "ECOG": ecog if not missing_ecog else None,
        "Tempo entre Última Consulta e PS (dias)": ti,
        "Internação Recente": tdr,
    })

if submit_button:
    # ⚠️ icd precisa ser lista [icd], não string isolada
    df_input = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "mbp": [float(mbp)],
        "hr": [float(hr)],
        "saot": [float(saot)],
        "height": [float(height)],
        "weight": [float(weight)],
        "bmi": [float(bmi)],
        "icd": [icd],
        "status_original": [status_original],
        "status_priority": [status_priority],
        "ti": [float(ti)],
        "tdr": [tdr],
        "tendency": [tendency],
        "ecog": [float(ecog)],
        "missing_ecog": [bool(missing_ecog)],
        "missing_bmi": [bool(missing_bmi)]
    })

if submit_button:
    # ---------- ENCODING ----------
    try:
        encoding_maps = load_encoding_maps()
        encoding_maps_cid = encoding_maps["ICD"]
        df_input["icd_processed"] = df_input["icd"].astype(str).str.split(" - ").str[0].str.lower()
        df_input["icd_encoded"] = df_input["icd_processed"].map(encoding_maps_cid).fillna(0.34162670016104163)
    except Exception as e:
        st.write("❌ Erro ao preparar Dado 'CID' para predição...", str(e))

    try:
        encoding_maps_status = encoding_maps["Status_Original"]
        df_input["status_original_encoded"] = df_input["status_original"].map(encoding_maps_status).fillna(0.31209494163715695)
    except Exception as e:
        st.write("❌ Erro ao preparar Dado 'Status' para predição...", str(e))

    try:
        priority_encoding = {"Verde": 1, "Amarelo": 2, "Vermelho": 3}
        df_input["status_priority_encoded"] = df_input["status_priority"].map(priority_encoding).fillna(1)
    except Exception as e:
        st.write("❌ Erro ao preparar Dado 'Prioridade' para predição...", str(e))

    try:
        gender_encoding = {"Feminino": 1, "Masculino": 0}
        df_input["gender_encoded"] = df_input["gender"].map(gender_encoding).fillna(1)
    except Exception as e:
        st.write("❌ Erro ao preparar Dado 'Sexo' para predição...", str(e))

    try:
        tdr_encoding = {"Não": 0, "Sim": 1}
        df_input["tdr_encoded"] = df_input["tdr"].map(tdr_encoding).fillna(0)
    except Exception as e:
        st.write("❌ Erro ao preparar Dado 'Reinternação' para predição...", str(e))

    try:
        tendency_encoding = {"Estável": 1, "Melhorando": 2, "Instável": 3}
        df_input["tendency_encoded"] = df_input["tendency"].map(tendency_encoding).fillna(1)
    except Exception as e:
        st.write("❌ Erro ao preparar Dado 'Tendência' para predição...", str(e))

    try:
        missing_ecog_encoding = {"False": 0, "True": 1}
        df_input["missing_ecog_encoded"] = df_input["missing_ecog"].astype(str).map(missing_ecog_encoding).fillna(1)
    except Exception as e:
        st.write("❌ Erro ao preparar Dado 'Ecog' para predição...", str(e))

    try:
        missing_bmi_encoding = {"False": 0, "True": 1}
        df_input["missing_bmi_encoded"] = df_input["missing_bmi"].astype(str).map(missing_bmi_encoding).fillna(1)
    except Exception as e:
        st.write("❌ Erro ao preparar Dado 'IMC' para predição...", str(e))

    try:
        df_input["ti_segundos"] = df_input["ti"] * 86400.0
        df_input["saot_fracao"] = df_input["saot"] / 100.0
    except Exception as e:
        st.write("❌ Erro ao preparar Dado 'Tempo entre Última Consulta e PS' para predição...", str(e))

    # ---------- SCALER ----------
    try:
        scaler = load_scaler()
    except Exception as e:
        st.write("❌ Erro ao preparar normalização dados...", str(e))

    try:
        selected_columns = [
            'bmi','hr','mbp','saot_fracao','weight','height','ti_segundos','ecog',
            'missing_bmi_encoded','missing_ecog_encoded','gender_encoded','tdr_encoded','tendency_encoded',
            'age','status_priority_encoded','icd_encoded','status_original_encoded'
        ]
        df_input = df_input.filter(items=selected_columns)
        column_scale_mapping = {
            "bmi": "BMI_knn", "hr": "HR_knn", "mbp": "MBP_knn", "saot_fracao": "OS_knn",
            "weight": "Weight_knn", "height": "Height_knn", "ti_segundos": "TI_median",
            "ecog": "ECOG_median", "missing_bmi_encoded": "missing_bmi", "missing_ecog_encoded": "missing_ecog",
            "gender_encoded": "Gender_binary", "tdr_encoded": "TDR_binary", "tendency_encoded": "Tendency_ordinal",
            "age": "Age", "status_priority_encoded": "Status_priority", "icd_encoded": "ICD",
            "status_original_encoded": "Status_Original"
        }
        df_input = df_input.rename(columns=column_scale_mapping)

        expected_columns = [
            'BMI_knn','HR_knn','MBP_knn','OS_knn','Weight_knn','Height_knn',
            'TI_median','ECOG_median','missing_bmi','missing_ecog','Gender_binary',
            'TDR_binary','Tendency_ordinal','Age','Status_priority','ICD','Status_Original'
        ]
        df_input_scaled = df_input.copy().astype(float)
        df_input_scaled = df_input_scaled[expected_columns]
        df_input_scaled[expected_columns] = scaler.transform(df_input_scaled[expected_columns])
    except Exception as e:
        st.write("❌ Erro ao normalizar dados...", str(e))

    # ---------- PREDIÇÃO COM MODELO EM CACHE ----------
    try:
        model = get_model()  # <- carrega do cache
        st.session_state['model'] = model  # opcional
        st.success("✅ Modelo carregado (cache).")

        h2o_df = h2o.H2OFrame(df_input_scaled)
        predictions = model.predict(h2o_df)
        predictions_df = predictions.as_data_frame()

        st.markdown("### Resultado da Predição:")

        limiar_classe_0 = 0.4283
        limiar_classe_1 = 0.5717

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Classe 0 - Longa Sobrevida', 'Classe 1 - Baixa Sobrevida'],
            y=[predictions_df['p0'][0], predictions_df['p1'][0]],
            marker=dict(color=['green', 'red']),
            text=[f"{predictions_df['p0'][0]:.2%}", f"{predictions_df['p1'][0]:.2%}"],
            textposition='auto',
        ))
        fig.add_trace(go.Scatter(
            x=['Classe 0 - Longa Sobrevida', 'Classe 1 - Baixa Sobrevida'],
            y=[limiar_classe_0, limiar_classe_0],
            mode="lines", line=dict(color="blue", dash="dash"),
            name=f"Limiar Classe 0 ({limiar_classe_0:.2%})"
        ))
        fig.add_trace(go.Scatter(
            x=['Classe 0 - Longa Sobrevida', 'Classe 1 - Baixa Sobrevida'],
            y=[limiar_classe_1, limiar_classe_1],
            mode="lines", line=dict(color="purple", dash="dash"),
            name=f"Limiar Classe 1 ({limiar_classe_1:.2%})"
        ))
        fig.update_layout(
            title="Distribuição das Probabilidades das Classes",
            xaxis_title="Classes", yaxis_title="Probabilidade",
            yaxis=dict(range=[0, 1]), showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        if predictions_df['predict'][0] == 1:
            st.success("A Classe 1 - Baixa Sobrevida - foi predita com probabilidade maior que o limiar de 57,17%. Portanto, classe predita para o paciente em questão é de Baixa Sobrevida")
        else:
            st.warning("A Classe 0 - Longa Sobrevida - foi predita com probabilidade maior que o limiar de 42,83%. Portanto, classe predita para o paciente em questão é de Longa Sobrevida")

        st.markdown("### Explicação da Predição:")
        shap_values = model.predict_contributions(h2o_df)
        shap_df = shap_values.as_data_frame().drop(columns=["BiasTerm"], errors="ignore")
        shap_df_melted = shap_df.T.reset_index()
        shap_df_melted.columns = ["Feature", "Importance"]

        rename_dict = {
            "BMI_knn": "IMC", "HR_knn": "Frequência Cardíaca", "MBP_knn": "Pressão Média",
            "OS_knn": "Saturação de Oxigênio", "Weight_knn": "Peso", "Height_knn": "Altura",
            "TI_median": "Tempo entre Consulta e PS", "ECOG_median": "ECOG",
            "missing_bmi": "IMC ausente", "missing_ecog": "ECOG ausente",
            "Gender_binary": "Sexo", "TDR_binary": "Internação Recente", "Tendency_ordinal": "Tendência",
            "Age": "Idade", "Status_priority": "Prioridade Status", "ICD": "CID", "Status_Original": "Status Original"
        }
        shap_df_melted["Feature"] = shap_df_melted["Feature"].replace(rename_dict)
        shap_df_melted = shap_df_melted.reindex(shap_df_melted["Importance"].abs().sort_values(ascending=True).index)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=shap_df_melted["Feature"],
            x=shap_df_melted["Importance"],
            orientation='h',
            marker=dict(color=shap_df_melted["Importance"], colorscale="RdBu"),
        ))
        fig2.update_layout(
            title="Explicação da Predição (SHAP) - Paciente em avaliação",
            xaxis_title="Importância SHAP", yaxis_title="Feature",
            xaxis=dict(showgrid=True), yaxis=dict(showgrid=False),
            margin=dict(l=140, r=20, t=60, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.write("❌ Erro ao realizar predição...", str(e))