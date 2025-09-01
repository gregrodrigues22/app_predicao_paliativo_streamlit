# explain.py
import os
import streamlit as st
import plotly.graph_objects as go
import gdown
import h2o

# =========================================================
# CONFIG INICIAL
# =========================================================
st.set_page_config(page_title="🧠 Explicação do Modelo", layout="wide")

# -------- util: procurar arquivos (logo/foto etc.)
ASSETS = "assets"
def first_existing(*names):
    for n in names:
        for base in (".", ASSETS):
            p = os.path.join(base, n)
            if os.path.exists(p):
                return p
    return None

LOGO = first_existing("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")

# =========================================================
# CABEÇALHO
# =========================================================
st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428);
                padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white; margin: 0;'>🧠 Explicação do Modelo</h1>
        <p style='color: white; font-size:16px; margin-top:8px;'>
            Predição de sobrevida em pacientes oncológicos na emergência –
            como o modelo foi construído, como performa e como interpretar suas saídas.
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
    unsafe_allow_html=True
)

# =========================================================
# MENU LATERAL
# =========================================================
with st.sidebar:
    if LOGO:
        st.image(LOGO, use_container_width=True)
    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.header("Menu")
    st.page_link("app.py",       label="Predição no PoC", icon="📈")
    st.page_link("explain.py",   label="Explicação do Modelo", icon="🧠")

    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.subheader("Conecte-se")
    st.markdown("""
- 💼 [LinkedIn](https://www.linkedin.com/in/gregorio-healthdata/)
- ▶️ [YouTube](https://www.youtube.com/@Patients2Python)
- 📸 [Instagram](https://www.instagram.com/patients2python/)
- 🌐 [Site](https://patients2python.com.br/)
- 🐙 [GitHub](https://github.com/gregrodrigues22)
- 👥💬 [Comunidade](https://chat.whatsapp.com/CBn0GBRQie5B8aKppPigdd)
- 🤝💬 [WhatsApp](https://patients2python.sprinthub.site/r/whatsapp-olz)
- 🎓 [Escola](https://app.patients2python.com.br/browse)
    """, unsafe_allow_html=True)

# =========================================================
# CONTEXTO CLÍNICO
# =========================================================
st.subheader("Contexto clínico e motivação")
st.markdown("""
A transição do cuidado **curativo para o paliativo** segue desafiadora na emergência oncológica.
Muitas vezes há **superestimação de sobrevida** e atrasos na discussão de metas terapêuticas.
O modelo busca **suportar decisões** com dados rotineiros da admissão (sinais vitais, prioridade/status,
diagnóstico por CID, ECOG da última consulta eletiva, idade, antropometria e histórico recente).
""")
st.info("**Uso clínico**: ferramenta de apoio — **não substitui** julgamento médico. Sempre integrar com avaliação clínica, preferências do paciente/família e diretrizes.")

# =========================================================
# CARREGAMENTO DO MODELO (CACHEADO)
# =========================================================
@st.cache_resource(show_spinner=True)
def load_mojo_model():
    file_id = "1IEGIuHt1l8xwR_Jl5J_fuKf0h5Fkdwx2"  # ID no Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    mojo_path = "modelo_em_mojo.zip"
    if not os.path.exists(mojo_path):
        gdown.download(url, mojo_path, quiet=True)
    h2o.init()
    return h2o.import_mojo(mojo_path)

with st.spinner("Carregando e preparando o modelo..."):
    try:
        model = load_mojo_model()
        st.success("✅ Modelo carregado com sucesso.")
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")
        st.stop()

# =========================================================
# MÉTRICAS GLOBAIS + CURVAS
# =========================================================
st.subheader("Desempenho global do modelo")

def _safe_call(obj, name, default=None, *args, **kwargs):
    try:
        fn = getattr(obj, name)
        return fn(*args, **kwargs)
    except Exception:
        return default

try:
    mp = model.model_performance()

    # — Métricas agregadas —
    auc      = _safe_call(mp, "auc")
    aucpr    = _safe_call(mp, "aucpr")
    logloss  = _safe_call(mp, "logloss")
    rmse     = _safe_call(mp, "rmse")
    mse      = _safe_call(mp, "mse")
    r2       = _safe_call(mp, "r2")

    # Cards
    cols = st.columns(6)
    cols[0].metric("AUC (ROC)", f"{auc:.3f}" if auc is not None else "—")
    cols[1].metric("AUCPR",     f"{aucpr:.3f}" if aucpr is not None else "—")
    cols[2].metric("LogLoss",   f"{logloss:.3f}" if logloss is not None else "—")
    cols[3].metric("RMSE",      f"{rmse:.3f}" if rmse is not None else "—")
    cols[4].metric("MSE",       f"{mse:.3f}" if mse is not None else "—")
    cols[5].metric("R²",        f"{r2:.3f}" if r2 is not None else "—")

    # — Curva ROC —
    st.markdown("#### Curva ROC")
    try:
        fpr, tpr = mp.roc()
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"ROC (AUC = {auc:.3f})" if auc is not None else "ROC"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Aleatório", line=dict(dash="dash")))
        fig_roc.update_layout(xaxis_title="Falsos Positivos (FPR)",
                              yaxis_title="Verdadeiros Positivos (TPR)",
                              template="plotly_white", height=420)
        st.plotly_chart(fig_roc, use_container_width=True)
    except Exception:
        st.warning("Não foi possível desenhar a ROC para este artefato.")

    # — Curva Precisão-Revocação (PR) —
    st.markdown("#### Curva Precisão-Revocação (PR)")
    try:
        # h2o disponibiliza precision(), recall() por threshold;
        # para traçar PR, buscamos arrays de precisão e recall
        prec_tbl = _safe_call(mp, "precision")
        rec_tbl  = _safe_call(mp, "recall")
        if prec_tbl is not None and rec_tbl is not None:
            precisions = [float(x) for x in prec_tbl["precision"]]
            recalls    = [float(x) for x in rec_tbl["recall"]]
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recalls, y=precisions, mode="lines+markers",
                                        name=f"PR (AUCPR = {aucpr:.3f})" if aucpr is not None else "PR"))
            fig_pr.update_layout(xaxis_title="Revocação (Sensibilidade)",
                                 yaxis_title="Precisão (PPV)",
                                 template="plotly_white", height=420)
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.info("Tabela de precisão/recall não disponível neste artefato.")
    except Exception:
        st.info("Não foi possível construir a curva PR com este artefato.")

    st.caption(
        "Interpretação: **AUC (ROC)** mede discriminação global; **AUCPR** é informativa em classes desbalanceadas. "
        "A ROC mostra trade-offs sensibilidade/especificidade; a PR mostra trade-offs precisão/revocação."
    )
except Exception as e:
    st.warning(f"Não foi possível calcular métricas globais: {e}")

# =========================================================
# MÉTRICAS POR LIMIAR (ÓTIMO F0.5) + MATRIZ DE CONFUSÃO
# =========================================================
st.subheader("Métricas no limiar operacional (ótimo F0.5)")

def get_metric_table(mp, name):
    tbl = _safe_call(mp, name)
    if tbl is None: 
        return None
    # h2o retorna dict-like com colunas; normalizamos
    try:
        thresholds = [float(x) for x in tbl["threshold"]]
    except Exception:
        thresholds = [float(x[0]) for x in tbl]  # fallback
    def col(colname, default=0.0):
        try:
            return [float(x) for x in tbl[colname]]
        except Exception:
            return [default]*len(thresholds)
    return thresholds, col

try:
    # tabelas por threshold
    f05_tbl = _safe_call(mp, "F0point5") or _safe_call(mp, "f0point5")
    f1_tbl  = _safe_call(mp, "f1")
    prec_tbl = _safe_call(mp, "precision")
    rec_tbl  = _safe_call(mp, "recall")
    acc_tbl  = _safe_call(mp, "accuracy")
    spec_tbl = _safe_call(mp, "specificity")

    # escolher threshold: max F0.5, senão max F1, senão 0.5
    if f05_tbl:
        ths  = [float(x) for x in f05_tbl["threshold"]]
        f05s = [float(x) for x in f05_tbl["f0point5"]]
        best_idx = max(range(len(f05s)), key=lambda i: f05s[i])
        best_th  = ths[best_idx]
        best_f05 = f05s[best_idx]
    elif f1_tbl:
        ths  = [float(x) for x in f1_tbl["threshold"]]
        f1s  = [float(x) for x in f1_tbl["f1"]]
        best_idx = max(range(len(f1s)), key=lambda i: f1s[i])
        best_th  = ths[best_idx]
        best_f05 = None
    else:
        best_th  = 0.5
        best_f05 = None

    # extrair métricas nesse threshold
    def metric_at(tbl, colname, th):
        if not tbl: return None
        ths = [float(x) for x in tbl["threshold"]]
        vals = [float(x) for x in tbl[colname]]
        # pega o mais próximo
        idx = min(range(len(ths)), key=lambda i: abs(ths[i]-th))
        return vals[idx]

    precision = metric_at(prec_tbl, "precision", best_th)
    recall    = metric_at(rec_tbl,  "recall",    best_th)
    accuracy  = metric_at(acc_tbl,  "accuracy",  best_th)
    specificity = metric_at(spec_tbl, "specificity", best_th)
    f1 = metric_at(f1_tbl, "f1", best_th) if f1_tbl else None

    c = st.columns(6)
    c[0].metric("Threshold", f"{best_th:.3f}")
    c[1].metric("Precisão (PPV)", f"{precision:.3f}" if precision is not None else "—")
    c[2].metric("Revocação (Sens.)", f"{recall:.3f}" if recall is not None else "—")
    c[3].metric("Especificidade", f"{specificity:.3f}" if specificity is not None else "—")
    c[4].metric("Acurácia", f"{accuracy:.3f}" if accuracy is not None else "—")
    c[5].metric("F0.5", f"{best_f05:.3f}" if best_f05 is not None else "—")

    # Matriz de confusão no limiar ótimo
    st.markdown("#### Matriz de confusão (no limiar selecionado)")
    try:
        # alguns artefatos trazem uma família de matrizes por threshold
        # tentamos pegar a do threshold mais próximo
        cm_list = _safe_call(mp, "confusion_matrix")
        if cm_list and "thresholds_and_metric_scores" in cm_list:
            # h2o às vezes expõe uma tabela; aqui optamos pela agregada se existir
            cm = _safe_call(mp, "confusion_matrix", None)
        else:
            cm = _safe_call(mp, "confusion_matrix", None, metrics="f0point5") or _safe_call(mp, "confusion_matrix", None)
        if cm is not None and hasattr(cm, "as_data_frame"):
            df_cm = cm.as_data_frame()
            # exibimos de forma enxuta se a tabela for grande
            try:
                st.dataframe(df_cm, use_container_width=True, hide_index=True)
            except Exception:
                st.write(df_cm)
        else:
            st.info("Confusion matrix não disponível para este artefato.")
    except Exception:
        st.info("Não foi possível recuperar a matriz de confusão para o limiar escolhido.")

    st.caption(
        "Limiar escolhido pelo **máximo F0.5** (quando disponível) para priorizar **precisão** e reduzir falso-positivo de Baixa Sobrevida. "
        "Revise periodicamente conforme capacidade assistencial."
    )
except Exception as e:
    st.warning(f"Não foi possível calcular métricas por limiar: {e}")

# =========================================================
# VARIÁVEIS MAIS RELEVANTES (SHAP)
# =========================================================
st.subheader("Variáveis mais relevantes e racional clínico")
st.markdown("""
- **ECOG** (última consulta eletiva) — maior peso; valores altos indicam pior prognóstico.  
- **CID** oncológico — sítio/estadiamento modulam risco basal.  
- **Sinais vitais** — **PAM**, **FC** e **SatO₂** refletem gravidade na admissão.  
- **TI** (tempo desde última consulta) e **TDR** (retorno/reinternação recente) indicam trajetória clínica.  
- **Idade/IMC** e **status/prioridade/tendência** complementam o estrato de risco.
""")

st.markdown("#### Visão global por SHAP")
shap_img = first_existing("shap_importance.png", "shap_importance.jpg", "shap_importance.webp")
if shap_img:
    st.image(os.path.join(ASSETS, "shap_importance.png") if "assets" in shap_img else shap_img,
             use_column_width=True, caption="Importância das variáveis explicativas (valores SHAP)")
else:
    st.warning("Imagem de SHAP não encontrada em assets/. Coloque 'assets/shap_importance.png'.")

st.caption(
    "Leitura: valores SHAP **positivos** empurram para **Baixa Sobrevida**; **negativos** para **Longa Sobrevida**. "
    "Maior dispersão horizontal = maior influência."
)

# =========================================================
# DESENVOLVIMENTO / LIMITAÇÕES / BOAS PRÁTICAS
# =========================================================
st.subheader("Desenvolvimento, validação e limitações")
st.markdown("""
- **H2O AutoML** com múltiplos algoritmos; **GBM** selecionado pelo equilíbrio AUC/AUCPR e baixo falso-positivo.  
- **Validação k-fold (k=5)**; limiar calibrado com **F0.5** (prioriza precisão).  
- Limitações: estudo **retrospectivo** de **centro único**; dependência de **dados de prontuário**; ausência de variáveis pode carregar **informação implícita** (p.ex., ECOG ausente).  
- Ética/Equidade: modelos podem refletir **viés** dos dados — uso **sob supervisão clínica**, com monitoramento e re-calibração.
""")

st.subheader("Boas práticas de uso no pronto-socorro")
st.markdown("""
1. Ativar **gatilho** para avaliação de **Cuidados Paliativos** quando a probabilidade de **Baixa Sobrevida** exceder o limiar operacional.  
2. **Confirmar** com exame clínico, história oncológica e preferências do paciente/família.  
3. **Documentar** discussão de metas e plano de cuidado (controle de sintomas, conforto, comunicação).  
4. **Monitorar** métricas locais (sens., esp., precisão, F0.5) e ajustar limiar conforme a capacidade assistencial.
""")

st.caption("Material para apoio à decisão clínica e educação. Não substitui o julgamento médico individualizado.")
