# ==============================================================
# Set up
# ==============================================================
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import h2o
import gdown
import plotly.graph_objects as go
from h2o.estimators import H2OGenericEstimator
import re

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

    with st.expander("Predição no PoC", expanded=True):
        st.page_link("app.py", label="Predição de Sobrevida", icon="📈")

    with st.expander("Explicação do Modelo", expanded=True):
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
# CONTEÚDO PRINCIPAL DO APP
# --------------------------------------------------------------
st.title("Predição no PoC 📈🎯")
st.write("Preencha os campos abaixo com os valores correspondentes às variáveis utilizadas no modelo preditivo.")

# ---------- caches utilitários ----------
@st.cache_data(show_spinner=False)
def load_cids_filtered_for_c():
    df = pd.read_csv("cids.csv")
    cols_lower = {c.lower(): c for c in df.columns}
    codigo_col = cols_lower.get("codigo") or "Codigo"
    desc_col   = cols_lower.get("descricao")

    # filtra apenas códigos que começam com "C"
    if codigo_col in df.columns:
        df[codigo_col] = df[codigo_col].astype(str)
        df_c = df[df[codigo_col].str.upper().str.startswith("C")].copy()
    else:
        df_c = pd.DataFrame(columns=[codigo_col])

    if not df_c.empty:
        if desc_col and desc_col in df_c.columns:
            options = (df_c[codigo_col] + " - " + df_c[desc_col].astype(str)).tolist()
        else:
            options = df_c[codigo_col].tolist()
        options = ["sem_cid - Não se aplica"] + options
    else:
        options = ["sem_cid - Não se aplica"]
    return options

@st.cache_data(show_spinner=False)
def load_encoding_maps():
    return joblib.load("encoding_maps.joblib")

@st.cache_data(show_spinner=False)
def load_scaler():
    return joblib.load("scaler.joblib")

# ---------- modelo H2O em cache ----------
@st.cache_resource(show_spinner=True)
def get_model():
    file_id = "1IEGIuHt1l8xwR_Jl5J_fuKf0h5Fkdwx2"  # ajuste se mudar
    url = f"https://drive.google.com/uc?id={file_id}"
    model_filename = "modelo_em_mojo.zip"
    if not os.path.exists(model_filename):
        gdown.download(url, model_filename, quiet=True)
    h2o.init()
    model = h2o.import_mojo(model_filename)
    return model

cid_options_c_only = load_cids_filtered_for_c()

# lista original (para encoding)
status_options_full = [
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

def status_display(s: str) -> str:
    # tira " - (Cor)" ou "(Cor)"
    return re.sub(r"\s*(-\s*)?\([^)]*\)", "", s).strip()

status_display_options = [status_display(s) for s in status_options_full]
status_display_to_full = {status_display(s): s for s in status_options_full}

# ========= WIDGETS (com keys para permitir reset) =========

# 1) Dados de nascimento
st.subheader("Dados de nascimento")
age = st.slider("Idade (anos)", min_value=0, max_value=120, value=60, step=1, key="age")
gender = st.radio("Sexo", options=["Masculino", "Feminino"], horizontal=True, key="gender")

st.markdown("---")

# 2) Dados vitais
st.subheader("Dados vitais")
sbp = st.slider("Pressão Arterial Sistólica (mmHg)", min_value=70, max_value=240, value=120, step=1, key="sbp")
dbp = st.slider("Pressão Arterial Diastólica (mmHg)", min_value=40, max_value=140, value=80, step=1, key="dbp")
mbp = round(dbp + (sbp - dbp) / 3.0, 1)  # PAM reativa
st.caption(f"PAM calculada automaticamente: **{mbp} mmHg**")

hr = st.slider("Frequência Cardíaca (bpm)", min_value=30, max_value=200, value=90, step=1, key="hr")
saot = st.slider("Saturação de Oxigênio (%)", min_value=50, max_value=100, value=97, step=1, key="saot")

st.markdown("---")

# 3) Antropometria
st.subheader("Antropometria")
missing_bmi = st.checkbox("Ausência de Antropometria", key="missing_bmi")

if not missing_bmi:
    height = st.slider("Altura (cm)", min_value=120, max_value=220, value=170, step=1, key="height")
    weight = st.slider("Peso (kg)", min_value=30, max_value=200, value=70, step=1, key="weight")
    bmi = round(weight / ((height / 100) ** 2), 1)
    st.caption(f"IMC calculado automaticamente: **{bmi} kg/m²**")
else:
    height, weight, bmi = 0.0, 0.0, 0.0

st.markdown("---")

# 4) Classificação de risco
st.subheader("Classificação de risco")
status_display_selected = st.selectbox("Status", options=status_display_options, index=0, key="status_display")
status_original = status_display_to_full[st.session_state["status_display"]]  # valor "full" p/ pipeline

prioridade_color = st.radio("Prioridade", options=["🟢 Verde", "🟡 Amarelo", "🔴 Vermelho"],
                            index=1, horizontal=True, key="prioridade_color")
priority_map_display_to_value = {
    "🟢 Verde": "Verde",
    "🟡 Amarelo": "Amarelo",
    "🔴 Vermelho": "Vermelho"
}
status_priority = priority_map_display_to_value[st.session_state["prioridade_color"]]

tendency = st.radio("Tendência clínica", options=tendency_options, horizontal=True, key="tendency")

st.markdown("---")

# 5) Doença neoplásica diagnosticada
st.subheader("Doença neoplásica diagnosticada")
icd = st.selectbox("CID-10 (apenas neoplasias – códigos iniciados por 'C')",
                   options=cid_options_c_only, index=0, key="icd")

st.markdown("---")

# 6) ECOG (condicional)
st.subheader("Escore Funcional (ECOG)")
missing_ecog = st.checkbox("Ausência de ECOG", key="missing_ecog")
if not missing_ecog:
    ecog = st.radio("ECOG", options=[0, 1, 2, 3, 4], index=0, horizontal=True, key="ecog")
else:
    ecog = 0.0

st.markdown("---")

# 7) Histórico da doença
st.subheader("Histórico da doença")
tdr = st.radio("Internação Recente", options=["Não", "Sim"], horizontal=True, key="tdr")
ti = st.slider("Tempo entre Última Consulta e PS (dias)", min_value=0, max_value=365, value=7, step=1, key="ti")

st.markdown("---")

# ========== Botões ==========
col1, col2 = st.columns(2)
submit_button = col1.button("Enviar", use_container_width=True)
reset_button  = col2.button("Nova simulação 🔁", type="secondary", use_container_width=True)

# handler do reset (limpa estado e reinicia)
if reset_button:
    keys_to_clear = [
        "age","gender","sbp","dbp","hr","saot",
        "missing_bmi","height","weight",
        "status_display","prioridade_color","tendency",
        "icd","missing_ecog","ecog","tdr","ti"
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# ---------- Exibição resumida ----------
if submit_button:
    pas = st.session_state["sbp"]
    pad = st.session_state["dbp"]
    pam_str = f"{pas}/{pad}/{mbp}"

    altura_val = 0 if st.session_state["missing_bmi"] else st.session_state["height"]
    peso_val   = 0 if st.session_state["missing_bmi"] else st.session_state["weight"]
    imc_val    = 0 if st.session_state["missing_bmi"] else bmi
    ecog_val   = None if st.session_state["missing_ecog"] else st.session_state.get("ecog", 0)

    st.success("Dados enviados com sucesso!")
    st.write({
        "Idade": st.session_state["age"],
        "Sexo": st.session_state["gender"],
        "PAS/PAD/PAM (mmHg)": pam_str,
        "Frequência Cardíaca (bpm)": st.session_state["hr"],
        "Saturação de Oxigênio (%)": st.session_state["saot"],
        "Ausência de Antropometria": st.session_state["missing_bmi"],
        "Altura (cm)": altura_val,
        "Peso (kg)": peso_val,
        "IMC (auto)": imc_val,
        "Status (exibição)": st.session_state["status_display"],
        "Status (interno p/ modelo)": status_original,
        "Prioridade (cor)": status_priority,
        "Tendência clínica": st.session_state["tendency"],
        "CID-10": st.session_state["icd"],
        "Ausência de ECOG": st.session_state["missing_ecog"],
        "ECOG": ecog_val,
        "Internação Recente": st.session_state["tdr"],
        "Tempo entre Última Consulta e PS (dias)": st.session_state["ti"],
    })

# ---------- DataFrame de entrada (mesmos nomes do pipeline) ----------
if submit_button:
    df_input = pd.DataFrame({
        "age": [st.session_state["age"]],
        "gender": [st.session_state["gender"]],
        "mbp": [float(mbp)],                              # PAM calculada reativamente
        "hr": [float(st.session_state["hr"])],
        "saot": [float(st.session_state["saot"])],
        "height": [0.0 if st.session_state["missing_bmi"] else float(st.session_state["height"])],
        "weight": [0.0 if st.session_state["missing_bmi"] else float(st.session_state["weight"])],
        "bmi": [0.0 if st.session_state["missing_bmi"] else float(bmi)],
        "icd": [st.session_state["icd"]],                # "Cxx - desc" ou "sem_cid - Não se aplica"
        "status_original": [status_original],            # valor completo (mapeado)
        "status_priority": [status_priority],            # Verde/Amarelo/Vermelho
        "ti": [float(st.session_state["ti"])],
        "tdr": [st.session_state["tdr"]],
        "tendency": [st.session_state["tendency"]],
        "ecog": [0.0 if st.session_state["missing_ecog"] else float(st.session_state.get("ecog", 0.0))],
        "missing_ecog": [bool(st.session_state["missing_ecog"])],
        "missing_bmi": [bool(st.session_state["missing_bmi"])],
    })

# ---------- ENCODINGS ----------
if submit_button:
    try:
        encoding_maps = load_encoding_maps()
        encoding_maps_cid = encoding_maps["ICD"]
        df_input["icd_processed"] = df_input["icd"].astype(str).str.split(" - ").str[0].str.lower()
        df_input.loc[~df_input["icd_processed"].str.startswith("c"), "icd_processed"] = "sem_cid"
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
if submit_button:
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

# ---------- PREDIÇÃO + DEVOLUTIVA CLÍNICA ----------
if submit_button:
    try:
        model = get_model()
        st.session_state['model'] = model

        h2o_df = h2o.H2OFrame(df_input_scaled)
        predictions = model.predict(h2o_df)
        predictions_df = predictions.as_data_frame()

        # Probabilidades e classe
        prob_longa  = float(predictions_df['p0'][0])
        prob_baixa  = float(predictions_df['p1'][0])
        classe_pred = int(predictions_df['predict'][0])

        # Limiar de decisão
        limiar_classe_0 = 0.4283
        limiar_classe_1 = 0.5717

        st.markdown("## Resultado da Predição:")

        # Gráfico de barras de probabilidade
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Classe 0 - Longa Sobrevida', 'Classe 1 - Baixa Sobrevida'],
            y=[prob_longa, prob_baixa],
            marker=dict(color=['green', 'red']),
            text=[f"{prob_longa:.2%}", f"{prob_baixa:.2%}"],
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

        # Resumo clínico mastigado
        st.markdown("## Resumo clínico para decisão")
        if classe_pred == 1:
            st.error(
                f"**Classe predita: Baixa Sobrevida (Classe 1)**\n\n"
                f"- Probabilidade estimada: **{prob_baixa:.1%}** (limiar de alerta: **{limiar_classe_1:.1%}**).\n"
                f"- Interpretação: o modelo indica **maior risco de baixa sobrevida** nessa admissão.\n"
                f"- Sinaliza necessidade de **avaliação clínica prioritária**, revisão de metas terapêuticas e plano de cuidado individualizado."
            )
        else:
            st.info(
                f"**Classe predita: Longa Sobrevida (Classe 0)**\n\n"
                f"- Probabilidade estimada: **{prob_longa:.1%}** (limiar: **{limiar_classe_0:.1%}**).\n"
                f"- Interpretação: o modelo indica **menor risco imediato** em comparação ao grupo de baixa sobrevida."
            )

        # SHAP / Contribuições (principais fatores)
        st.markdown("## Explicação da Predição:")

        shap_values = model.predict_contributions(h2o_df)
        shap_df = shap_values.as_data_frame().drop(columns=["BiasTerm"], errors="ignore")
        shap_df_melted = shap_df.T.reset_index()
        shap_df_melted.columns = ["Feature", "Importance"]

        rename_dict = {
            "BMI_knn": "IMC","HR_knn": "Frequência Cardíaca","MBP_knn": "Pressão Média",
            "OS_knn": "Saturação de Oxigênio","Weight_knn": "Peso","Height_knn": "Altura",
            "TI_median": "Tempo entre Consulta e PS","ECOG_median": "ECOG",
            "missing_bmi": "IMC ausente","missing_ecog": "ECOG ausente",
            "Gender_binary": "Sexo","TDR_binary": "Internação Recente","Tendency_ordinal": "Tendência",
            "Age": "Idade","Status_priority": "Prioridade Status","ICD": "CID","Status_Original": "Status Original"
        }
        shap_df_melted["Feature"] = shap_df_melted["Feature"].replace(rename_dict)

        # Top 5 fatores por |SHAP|
        top5 = (shap_df_melted.assign(abs_imp=lambda d: d["Importance"].abs())
                .sort_values("abs_imp", ascending=False)
                .head(5)[["Feature", "Importance"]])

        bullets = []
        for _, row in top5.iterrows():
            direcao = "↑ risco" if row["Importance"] > 0 else "↓ risco"
            bullets.append(f"- **{row['Feature']}** — impacto (SHAP): **{row['Importance']:.3f}** → *{direcao}*")

        st.markdown("**Principais fatores que influenciaram esta predição:**")
        st.markdown("\n".join(bullets))
        st.caption(
            "Observação: valores SHAP indicam contribuição relativa de cada variável para este caso. "
            "Sinal positivo tende a empurrar para a classe predita; sinal negativo, no sentido oposto."
        )

        # Gráfico SHAP
        shap_df_plot = shap_df_melted.sort_values("Importance", key=lambda s: s.abs(), ascending=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=shap_df_plot["Feature"],
            x=shap_df_plot["Importance"],
            orientation='h',
            marker=dict(color=shap_df_plot["Importance"], colorscale="RdBu"),
        ))
        fig2.update_layout(
            title="Explicação da Predição (SHAP) - Paciente em avaliação",
            xaxis_title="Importância SHAP", yaxis_title="Feature",
            xaxis=dict(showgrid=True), yaxis=dict(showgrid=False),
            margin=dict(l=140, r=20, t=60, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Dados usados (para auditoria clínica rápida)
        st.markdown("## Dados utilizados nesta avaliação")
        colA, colB, colC = st.columns(3)

        # colA
        with colA:
            st.write(f"**Idade:** {st.session_state['age']} anos")
            st.write(f"**Sexo:** {st.session_state['gender']}")
            st.write(f"**PAS/PAD/PAM:** {st.session_state['sbp']}/{st.session_state['dbp']}/{mbp} mmHg")
            st.write(f"**FC:** {st.session_state['hr']} bpm")

        # colB (evitar f-strings aninhadas)
        with colB:
            st.write(f"**SatO₂:** {st.session_state['saot']}%")
            if st.session_state["missing_bmi"]:
                altura_peso_txt = "-"
                imc_txt = "-"
            else:
                altura_peso_txt = f"{st.session_state['height']} cm / {st.session_state['weight']} kg"
                imc_txt = f"{bmi} kg/m²"
            st.write(f"**Altura/Peso:** {altura_peso_txt}")
            st.write(f"**IMC:** {imc_txt}")
            st.write(f"**ECOG:** {'ausente' if st.session_state['missing_ecog'] else st.session_state.get('ecog', 0)}")

        # colC
        with colC:
            st.write(f"**Status:** {st.session_state['status_display']}")
            st.write(f"**Prioridade:** {status_priority}")
            st.write(f"**Tendência:** {st.session_state['tendency']}")
            st.write(f"**CID:** {st.session_state['icd']}")
            st.write(f"**Internação recente:** {st.session_state['tdr']}")
            st.write(f"**Tempo desde última consulta:** {st.session_state['ti']} dias")

        # Lembrete ético
        st.caption(
            "Este modelo é **apoio à decisão clínica** e **não substitui** julgamento médico. "
            "Interprete sempre à luz do quadro clínico, preferências do paciente/família e diretrizes vigentes."
        )

    except Exception as e:
        st.write("❌ Erro ao realizar predição...", str(e))