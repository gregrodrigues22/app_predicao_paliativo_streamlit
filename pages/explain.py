# pages/explain.py
import os
import streamlit as st
import plotly.graph_objects as go
import gdown
import h2o

# =======================
# CONFIG INICIAL
# =======================
st.set_page_config(page_title="🧠 Explicação do Modelo", layout="wide")

ASSETS = "assets"
def first_existing(*names):
    for n in names:
        for base in (".", ASSETS):
            p = os.path.join(base, n)
            if os.path.exists(p):
                return p
    return None

LOGO = first_existing("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")

# =======================
# CABEÇALHO
# =======================
st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428);
                padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white; margin: 0;'>🧠 Explicação do Modelo</h1>
        <p style='color: white; font-size:16px; margin-top:8px;'>
            Predição de sobrevida em pacientes oncológicos na emergência – como o modelo foi construído,
            como performa e como interpretar suas saídas.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Oculta a navegação padrão do Streamlit na sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# =======================
# MENU LATERAL
# =======================
with st.sidebar:
    if LOGO:
        st.image(LOGO, use_container_width=True)
    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.header("Menu")
    st.page_link("app.py", label="Predição no PoC", icon="📈")
    # use sempre o caminho com 'pages/' para evitar StreamlitPageNotFoundError
    st.page_link("pages/explain.py", label="Explicação do Modelo", icon="🧠")

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

# =======================
# CONTEXTO E OBJETIVO
# =======================
st.subheader("Contexto clínico e motivação")
st.markdown("""
A transição do cuidado **curativo para o paliativo** segue desafiadora na emergência oncológica.  
Com frequência há **superestimação da sobrevida**, atrasando discussões de metas e acesso oportuno a CP.
O modelo busca **apoiar decisões** usando variáveis **rotineiras da admissão** (sinais vitais, prioridade/status,
diagnóstico por CID, ECOG da última consulta eletiva, idade, antropometria e histórico recente).
""")
st.info("**Uso clínico**: ferramenta de apoio — **não substitui** julgamento médico. Integre sempre com exame, preferências do paciente/família e diretrizes.")

# =======================
# CARREGAMENTO DO MODELO (CACHE)
# =======================
@st.cache_resource(show_spinner=True)
def load_mojo_model():
    file_id = "1IEGIuHt1l8xwR_Jl5J_fuKf0h5Fkdwx2"
    url = f"https://drive.google.com/uc?id={file_id}"
    mojo_path = "modelo_em_mojo.zip"
    if not os.path.exists(mojo_path):
        gdown.download(url, mojo_path, quiet=True)
    h2o.init()
    return h2o.import_mojo(mojo_path)

with st.spinner("Carregando o modelo..."):
    try:
        model = load_mojo_model()
        st.success("✅ Modelo carregado com sucesso.")
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")
        st.stop()

# =======================
# MÉTRICAS GLOBAIS
# =======================
st.subheader("Desempenho global do modelo")

def _safe_call(obj, name, default=None, *args, **kwargs):
    try:
        fn = getattr(obj, name)
        return fn(*args, **kwargs)
    except Exception:
        return default

try:
    mp = model.model_performance()

    auc      = _safe_call(mp, "auc")
    aucpr    = _safe_call(mp, "aucpr")
    logloss  = _safe_call(mp, "logloss")
    rmse     = _safe_call(mp, "rmse")
    mse      = _safe_call(mp, "mse")
    r2       = _safe_call(mp, "r2")

    c = st.columns(6)
    c[0].metric("AUC (ROC)", f"{auc:.3f}" if auc is not None else "—")
    c[1].metric("AUCPR",     f"{aucpr:.3f}" if aucpr is not None else "—")
    c[2].metric("LogLoss",   f"{logloss:.3f}" if logloss is not None else "—")
    c[3].metric("RMSE",      f"{rmse:.3f}" if rmse is not None else "—")
    c[4].metric("MSE",       f"{mse:.3f}" if mse is not None else "—")
    c[5].metric("R²",        f"{r2:.3f}" if r2 is not None else "—")

    # ROC
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
        st.warning("Não foi possível construir a curva ROC com este artefato.")

    # PR (Precisão-Revocação)
    st.markdown("#### Curva Precisão-Revocação (PR)")
    try:
        # precision() e recall() podem vir como Frame-like ou dict-like dependendo do artefato
        prec_tbl = _safe_call(mp, "precision")
        rec_tbl  = _safe_call(mp, "recall")

        def _extract_col(tbl, key):
            # aceita dict, H2OFrame.as_data_frame(), lista de dicts etc.
            try:
                if hasattr(tbl, "as_data_frame"):
                    df = tbl.as_data_frame()
                    return df[key].astype(float).tolist()
                if isinstance(tbl, dict):
                    return [float(x) for x in tbl[key]]
                if isinstance(tbl, list):
                    return [float(row.get(key, 0)) for row in tbl]
            except Exception:
                pass
            return None

        precisions = _extract_col(prec_tbl, "precision")
        recalls    = _extract_col(rec_tbl,  "recall")

        if precisions and recalls and len(precisions) == len(recalls):
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recalls, y=precisions, mode="lines+markers",
                                        name=f"PR (AUCPR = {aucpr:.3f})" if aucpr is not None else "PR"))
            fig_pr.update_layout(xaxis_title="Revocação (Sensibilidade)",
                                 yaxis_title="Precisão (PPV)",
                                 template="plotly_white", height=420)
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.info("Não foi possível construir a curva PR com este artefato.")
    except Exception:
        st.info("Não foi possível construir a curva PR com este artefato.")

    st.caption("**AUC (ROC)** mede discriminação global; **AUCPR** é informativa em base desbalanceada. "
               "A ROC mostra trade-offs sens./esp.; a PR mostra trade-offs precisão/revocação.")
except Exception as e:
    st.warning(f"Não foi possível calcular métricas globais: {e}")

# =======================
# MÉTRICAS POR LIMIAR
# =======================
st.subheader("Métricas no limiar operacional (ótimo F0.5)")

try:
    f05_tbl = _safe_call(mp, "F0point5") or _safe_call(mp, "f0point5")
    f1_tbl  = _safe_call(mp, "f1")
    prec_tbl = _safe_call(mp, "precision")
    rec_tbl  = _safe_call(mp, "recall")
    acc_tbl  = _safe_call(mp, "accuracy")
    spec_tbl = _safe_call(mp, "specificity")

    def _get_pairs(tbl, value_col):
        """Retorna pares (threshold, valor) a partir de diferentes formatos."""
        if tbl is None:
            return []
        try:
            if hasattr(tbl, "as_data_frame"):
                df = tbl.as_data_frame()
                return list(zip(df["threshold"].astype(float), df[value_col].astype(float)))
            if isinstance(tbl, dict):
                ths = [float(x) for x in tbl["threshold"]]
                vals = [float(x) for x in tbl[value_col]]
                return list(zip(ths, vals))
            if isinstance(tbl, list):
                out = []
                for row in tbl:
                    th = float(row.get("threshold", 0))
                    val = float(row.get(value_col, 0))
                    out.append((th, val))
                return out
        except Exception:
            return []
        return []

    # escolhe threshold pelo maior F0.5; fallback F1; senão 0.5
    if f05_tbl:
        pairs = _get_pairs(f05_tbl, "f0point5")
        best_th, best_f05 = max(pairs, key=lambda x: x[1]) if pairs else (0.5, None)
    elif f1_tbl:
        pairs = _get_pairs(f1_tbl, "f1")
        best_th, best_f05 = (max(pairs, key=lambda x: x[1])[0], None) if pairs else (0.5, None)
    else:
        best_th, best_f05 = 0.5, None

    def _value_at(tbl, colname, th):
        pairs = _get_pairs(tbl, colname)
        if not pairs:
            return None
        # escolhe pelo threshold mais próximo
        th0, val0 = min(pairs, key=lambda x: abs(x[0] - th))
        return val0

    precision = _value_at(prec_tbl, "precision", best_th)
    recall    = _value_at(rec_tbl,  "recall",    best_th)
    accuracy  = _value_at(acc_tbl,  "accuracy",  best_th)
    specificity = _value_at(spec_tbl, "specificity", best_th)
    f1 = _value_at(f1_tbl, "f1", best_th) if f1_tbl else None

    k = st.columns(6)
    k[0].metric("Threshold", f"{best_th:.3f}")
    k[1].metric("Precisão (PPV)", f"{precision:.3f}" if precision is not None else "—")
    k[2].metric("Revocação (Sens.)", f"{recall:.3f}" if recall is not None else "—")
    k[3].metric("Especificidade", f"{specificity:.3f}" if specificity is not None else "—")
    k[4].metric("Acurácia", f"{accuracy:.3f}" if accuracy is not None else "—")
    k[5].metric("F0.5", f"{best_f05:.3f}" if best_f05 is not None else "—")

    # Matriz de confusão (tenta o agregado mais próximo do threshold)
    st.markdown("#### Matriz de confusão (no limiar selecionado)")
    try:
        cm = _safe_call(mp, "confusion_matrix")  # pode vir já agregada
        if cm is not None and hasattr(cm, "as_data_frame"):
            df_cm = cm.as_data_frame()
            st.dataframe(df_cm, use_container_width=True)
        else:
            st.info("Confusion matrix não disponível para este artefato.")
    except Exception:
        st.info("Não foi possível recuperar a matriz de confusão para o limiar escolhido.")

    st.caption("Limiar escolhido pelo **máximo F0.5** (quando disponível) para priorizar **precisão** e reduzir falso-positivo.")
except Exception as e:
    st.warning(f"Não foi possível calcular métricas por limiar: {e}")

# =======================
# VARIÁVEIS E SHAP
# =======================
st.subheader("Variáveis mais relevantes e racional clínico")
st.markdown("""
- **ECOG** (última consulta eletiva) — maior peso; valores altos indicam pior prognóstico.  
- **CID oncológico** — sítio/estadiamento modulam risco basal.  
- **Sinais vitais** — **PAM**, **FC** e **SatO₂** refletem gravidade na admissão.  
- **TI** (tempo desde última consulta) e **TDR** (retorno/reinternação recente) indicam trajetória clínica.  
- **Idade/IMC** e **status/prioridade/tendência** complementam o estrato de risco.
""")

st.markdown("#### Visão global por SHAP")
shap_img = first_existing("shap_importance.png", "shap_importance.jpg", "shap_importance.webp")
if shap_img:
    st.image(shap_img, use_container_width=True, caption="Importância das variáveis explicativas (valores SHAP)")
else:
    st.warning("Imagem de SHAP não encontrada em assets/. Coloque 'assets/shap_importance.png'.")

# --- EXPLICAÇÃO DIDÁTICA DOS SHAP (texto solicitado) ---
st.markdown("""
##### Como ler o gráfico SHAP

**O que o gráfico mostra?**  
Ele indica **como cada variável contribui** para as previsões do modelo. **Cada ponto** é um valor SHAP de uma observação.

**Elementos do gráfico**  
- **Eixo Y — Nome das variáveis**: listadas do topo para a base **em ordem de importância**. Mais no topo ⇒ maior influência.  
- **Eixo X — Valor SHAP**: magnitude e **direção do impacto** na previsão.  
  - Valores **positivos** deslocam a predição para **maior probabilidade** da **classe-alvo** (Baixa Sobrevida).  
  - Valores **negativos** deslocam para **menor probabilidade** da classe-alvo (Longa Sobrevida).  
- **Cores — Valor normalizado da variável**:  
  - **Azul** = valores **baixos**; **Rosa** = valores **altos**.  
  Isso ajuda a relacionar **nível da variável** ↔ **efeito no risco**.

**Como interpretar?**  
- Variáveis com **maior dispersão horizontal** (pontos mais espalhados no eixo X) têm **maior impacto**.  
- Se os pontos de uma variável estão predominantemente **à direita** (SHAP positivos), ela tende a **aumentar** a chance da classe predita.  
- Se estão **à esquerda** (SHAP negativos), tende a **reduzir** essa chance.  
- **Sobreposição de cores** indica relações possivelmente **não lineares**.

**Aplicações clínicas no nosso modelo**  
- **ECOG**: valores mais altos refletem pior estado funcional → **pior sobrevida**. É esperado que seja a variável mais influente.  
- **Internação/retorno recente (TDR)**: marca instabilidade clínica → **pior prognóstico**.  
- **CID (ICD)**: diferentes sítios/estádios têm pesos distintos no risco basal.  
- **Frequência Cardíaca (HR)**: alterações indicam gravidade fisiológica → influência relevante na admissão.  
- **Peso e Altura**: embora o IMC agregue a informação, **ainda aparece efeito adicional** de peso/altura.  
- **missing_ecog / missing_bmi**: a **ausência** de dados também carrega informação (p.ex., ECOG ausente pode refletir gravidade/fluxo).  
- **Idade**: maior idade costuma se associar a **pior prognóstico** em oncologia.  
- **Tempo desde a última consulta (TI)**:  
  - **Baixo TI** → pode indicar **agravamento rápido** → pior prognóstico;  
  - **Alto TI** → período estável mais longo → melhor prognóstico.
""")

# =======================
# DESENVOLVIMENTO / LIMITAÇÕES / BOAS PRÁTICAS
# =======================
st.subheader("Desenvolvimento, validação e limitações")
st.markdown("""
- **H2O AutoML** com múltiplos algoritmos; **GBM** escolhido pelo equilíbrio AUC/AUCPR e menor falso-positivo.  
- **Validação k-fold (k=5)**; limiar calibrado com **F0.5** (prioriza **precisão**).  
- Limitações: estudo **retrospectivo** de **centro único**; dependência de dados de prontuário; ausência de variáveis pode carregar **informação implícita** (ex.: ECOG ausente).  
- Ética/Equidade: modelos podem refletir **vieses**; uso **sob supervisão clínica**, com monitoramento e recalibração.
""")

st.subheader("Boas práticas de uso no pronto-socorro")
st.markdown("""
1. Utilize como **gatilho** para avaliação de **Cuidados Paliativos** quando a probabilidade de **Baixa Sobrevida** exceder o limiar operacional.  
2. **Confirme** com exame clínico, história oncológica e preferências do paciente/família.  
3. **Documente** decisões e plano de cuidado (controle de sintomas, conforto, comunicação).  
4. **Monitore** métricas locais (sens., esp., precisão, F0.5) e ajuste o limiar conforme a capacidade assistencial.
""")

st.caption("Material para **apoio à decisão clínica** e educação. Não substitui o julgamento médico individualizado.")