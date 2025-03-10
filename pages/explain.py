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
