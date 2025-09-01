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

# =======================
# CONTEXTO E OBJETIVO
# =======================
st.subheader("Contexto clínico e motivação")
st.markdown("""
A transição do cuidado curativo para o paliativo continua sendo um desafio na prática clínica, especialmente para pacientes com câncer atendidos em serviços de urgência e emergência (ER).
Nesses cenários, marcados por informações clínicas fragmentadas e necessidade de decisões rápidas, médicos frequentemente superestimam a sobrevida dos pacientes, o que dificulta a adoção de cuidados paliativos.
""")

st.subheader("Substitutos")
st.markdown("""
Ao mesmo tempo, ferramentas tradicionais de predição de sobrevida disponíveis na literatura médica, como , Palliative Prognostic Score, são limitadas no contexto de urgências e emergências. 
Muitas dessas escalas exigem uma coleta extensa de dados ou dependem de exames laboratoriais e critérios subjetivos que podem não estar acessíveis no momento da admissão do paciente no pronto-socorro.
""")

st.subheader("Solução")
st.markdown("""
Este aplicativo utiliza um modelo de aprendizado de máquina desenvolvido a partir de dados reais de pacientes atendidos atendidos em pronto-socorro de hospital oncológico porta-fechada em São Paulo. 
O objetivo é prever a sobrevida de curto prazo (menos de seis meses) e longo prazo (mais de seis meses) de pacientes a partir de variáveis clínicas e demográficas disponíveis no momento da admissão na emergência.
""")

st.subheader("Metodologia")
st.markdown("""
O estudo adotou delineamento observacional retrospectivo, com base em 51.311 registros de pacientes oncológicos atendidos em pronto-socorro de hospital porta-fechada em São Paulo, restrito a pacientes previamente vinculados. 
Foram extraídos do prontuário eletrônico dados demográficos, clínicos e de sinais vitais. O target foi definido a partir de pareceres de paliativistas, totalizando 32.993 casos de longa sobrevida (label 0) e 18.318 de curta sobrevida (label 1). 
O pré-processamento incluiu tratamento de valores ausentes, normalização e codificação de variáveis. A amostra foi dividida em treino, validação e teste de forma estratificada, com uso de validação cruzada (5 folds) para maior robustez. 
Modelos supervisionados de machine learning foram aplicados e comparados a escalas prognósticas tradicionais. A avaliação privilegiou o F0.5 Score, métrica que enfatiza a precisão e reduz falsos positivos clinicamente relevantes. 
Além disso, o modelo foi validado em amostras com parecer de médicos especialistas, reforçando sua aderência ao julgamento clínico. 
Por fim, desenvolveu-se uma aplicação web em Streamlit para simulação de uso, ainda em fase experimental.
""")

st.subheader("Desafios e Aprendizados")
st.markdown("""
Os principais desafios envolveram o tratamento de dados incompletos, a definição de métricas adequadas e a interpretação de modelos complexos. 
Sobre aprendizados, observa-se que escalas tradicionais, como o Palliative Prognostic Score, mostram limitações de subjetividade e baixa aplicabilidade no pronto-socorro. 
O modelo desenvolvido destacou-se pela praticidade, ao usar variáveis rotineiras da admissão e disponibilizar os resultados em um app em Streamlit de fácil uso. 
Conclui-se que a IA pode apoiar a triagem precoce de pacientes para cuidados paliativos, oferecendo desempenho robusto e potencial de integração aos prontuários eletrônicos, com próximos passos voltados à validação multicêntrica.

""")

st.info("**Uso clínico**: ferramenta de apoio — **não substitui** julgamento médico. Integre sempre com exame, preferências do paciente/família e diretrizes.")

# =======================
# CARREGAMENTO DO MODELO (CACHEADO)
# =======================
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

# =======================
# MÉTRICAS GLOBAIS + CURVA ROC
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

    # Métricas agregadas (quando disponíveis no artefato)
    auc      = _safe_call(mp, "auc")
    aucpr    = _safe_call(mp, "aucpr")
    logloss  = _safe_call(mp, "logloss")
    rmse     = _safe_call(mp, "rmse")
    mse      = _safe_call(mp, "mse")
    r2       = _safe_call(mp, "r2")

    cols = st.columns(6)
    cols[0].metric("AUC (ROC)", f"{auc:.3f}" if auc is not None else "—")
    cols[1].metric("AUCPR",     f"{aucpr:.3f}" if aucpr is not None else "—")
    cols[2].metric("LogLoss",   f"{logloss:.3f}" if logloss is not None else "—")
    cols[3].metric("RMSE",      f"{rmse:.3f}" if rmse is not None else "—")
    cols[4].metric("MSE",       f"{mse:.3f}" if mse is not None else "—")
    cols[5].metric("R²",        f"{r2:.3f}" if r2 is not None else "—")

    # Curva ROC
    st.markdown("#### Curva ROC")
    try:
        fpr, tpr = mp.roc()
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"ROC (AUC = {auc:.3f})" if auc is not None else "ROC"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Aleatório", line=dict(dash="dash")
        ))
        fig_roc.update_layout(
            xaxis_title="Falsos Positivos (FPR)",
            yaxis_title="Verdadeiros Positivos (TPR)",
            template="plotly_white", height=420
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    except Exception:
        st.warning("Não foi possível desenhar a ROC para este artefato.")

    st.caption(
        "**AUC (ROC)** mede a capacidade de discriminar as classes em todos os limiares. "
        "Valores próximos a **1,0** indicam excelente discriminação; **0,5** equivale ao acaso."
    )
except Exception as e:
    st.warning(f"Não foi possível calcular/exibir métricas globais: {e}")

 # Curva ROC
    st.markdown("#### Curva ROC")
    try:
        fpr, tpr = mp.roc()
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"ROC (AUC = {auc:.3f})" if auc is not None else "ROC"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Aleatório", line=dict(dash="dash")
        ))
        fig_roc.update_layout(
            xaxis_title="Taxa de Falsos Positivos (FPR)",
            yaxis_title="Taxa de Verdadeiros Positivos (TPR)",
            template="plotly_white", height=420
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    except Exception:
        st.warning("Não foi possível desenhar a ROC para este artefato.")

    # Interpretação detalhada
    st.markdown("""
    **Interpretação da Curva ROC:**  
    A Curva ROC (Receiver Operating Characteristic) mostra o desempenho do modelo ao variar o limiar de decisão.  

    - O eixo **X** representa a **Taxa de Falsos Positivos (FPR)**, ou seja, a proporção de negativos incorretamente classificados como positivos.  
    - O eixo **Y** representa a **Taxa de Verdadeiros Positivos (TPR)**, a proporção de positivos corretamente identificados.  
    - A linha pontilhada cinza representa um **modelo aleatório**, enquanto a curva azul mostra o **modelo preditivo**.  
    - O valor de **AUC (Área sob a Curva)** indica a capacidade de o modelo distinguir entre as classes:  
        - Um **AUC próximo de 1,0** → modelo altamente discriminativo.  
        - Um **AUC de 0,5** → modelo sem capacidade preditiva (equivalente ao acaso).  

    Neste caso, um AUC de **{:.3f}** sugere que o modelo apresenta **excelente desempenho** na diferenciação entre longa e curta sobrevida.
    """.format(auc if auc else 0.0))
    
# =======================
# VARIÁVEIS IMPORTANTES + SHAP
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
    st.warning("Imagem de SHAP não encontrada em assets/. Coloque 'assets/shap_importance.png' no repositório.")

# Explicação detalhada do SHAP (texto solicitado)
st.markdown("""
##### Como ler o gráfico SHAP

**O que o gráfico mostra?**  
O gráfico de importância dos atributos via **SHAP** mostra **como cada variável contribui** para as previsões do modelo. **Cada ponto** representa um valor SHAP de uma observação do conjunto de dados.

**Elementos do Gráfico**  
- **Eixo Y — Nome das variáveis**: listadas do topo para a base, **em ordem de importância**. Quanto mais alto, maior a influência na predição.  
- **Eixo X — Valor SHAP**: indica a **magnitude** e a **direção** do impacto na predição.  
  - **Positivo**: desloca a predição para **maior probabilidade** da **classe-alvo** (Baixa Sobrevida).  
  - **Negativo**: desloca a predição para **menor probabilidade** da classe-alvo (Longa Sobrevida).  
- **Cores — Valor Normalizado da Variável**:  
  - **Azul** = valores **baixos**; **Rosa** = valores **altos** – ajuda a entender a relação **valor da variável ↔ efeito no risco**.

**Como interpretar?**  
- Variáveis com **maior dispersão horizontal** (pontos mais espalhados no eixo X) têm **maior impacto** nas predições.  
- Se os pontos de uma variável estão predominantemente **à direita** (SHAP positivos), ela tende a **aumentar** a chance da classe predita.  
- Se estão **à esquerda** (SHAP negativos), tende a **reduzir** essa chance.  
- **Sobreposição de cores** indica que a relação entre a variável e a predição pode ser **complexa**, não apenas linear.

**Aplicação do modelo**  
- **ECOG**: valores mais altos indicam pior estado funcional, **fortemente associados a menor sobrevida** — coerente com a literatura.  
- **Internação recente (TDR)**: indica **piora/instabilidade** clínica, associando-se a maior risco de óbito em curto prazo.  
- **CID (ICD)**: a diversidade de códigos sugere pesos distintos por sítio/estágio tumoral.  
- **Frequência Cardíaca (HR)**: indicador do estado fisiológico; alterações **elevam o risco**.  
- **Peso e Altura**: mesmo com IMC, observa-se **efeito adicional** de peso/altura.  
- **Ausências (missing_ecog / missing_bmi)**: a falta de informação **pode ser informativa** (p.ex., ECOG ausente por gravidade/fluxo).  
- **Idade**: maior idade tende a **pior prognóstico** em oncologia – alinhado à literatura.  
- **Tempo entre Última Consulta e PS (TI)**:  
  - **Baixo TI** → pode indicar **agravamento rápido** → pior prognóstico;  
  - **Alto TI** → período estável mais longo → melhor prognóstico.
""")

# =======================
# DESENVOLVIMENTO / LIMITAÇÕES / BOAS PRÁTICAS
# =======================
st.subheader("Desenvolvimento, validação e limitações")
st.markdown("""
- **H2O AutoML** com múltiplos algoritmos; **GBM** selecionado pelo equilíbrio **AUC/AUCPR** e baixo falso-positivo.  
- **Validação k-fold (k=5)**; calibração focada em **F0.5** (privilegia **precisão**).  
- **Limitações**: estudo retrospectivo, centro único, dependência de dados de prontuário; variáveis ausentes podem carregar **informação implícita** (ex.: ECOG ausente).  
- **Ética/Equidade**: modelos podem refletir **vieses**; uso **sob supervisão clínica** e com **monitoramento e recalibração**.
""")

st.subheader("Boas práticas de uso no pronto-socorro")
st.markdown("""
1. Utilize como **gatilho** para avaliação de **Cuidados Paliativos** quando a probabilidade de **Baixa Sobrevida** for elevada.  
2. **Confirme** com exame clínico, história oncológica e preferências do paciente/família.  
3. **Documente** decisões e plano de cuidado (controle de sintomas, conforto, comunicação).  
4. **Monitore** métricas locais (sens., esp., precisão) e ajuste políticas conforme capacidade assistencial.
""")

st.caption("Material para **apoio à decisão clínica** e educação. Não substitui o julgamento médico individualizado.")
