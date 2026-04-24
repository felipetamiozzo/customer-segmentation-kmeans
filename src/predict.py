#src/predict.py
from pathlib import Path

import joblib
import pandas as pd

#Importa nosso Pipeline
from data_processing import (
    clean_data,
    create_features,
    remove_redundant_columns,
    standardize_categories,
)
from train_model import prepare_ml_data


#CONSTANTES DE NEGÓCIO

# Traduz a matemática dos clusters para a linguagem do negócio (Marketing/CRM)
CLUSTER_NAMES = {
    0: "VIPs Premium",
    1: "Grande Massa / Caçadores de Ofertas",
    2: "Classe Média Engajada",
    3: "Promissores",
}

#
#CARREGAMENTO DOS ARTEFATOS (Modelos e Transformadores)
#
def load_artifacts(models_dir: str = "models"):
    """
    Carrega todos os arquivos gerados no pipeline de treinamento.
    Usa Pathlib para garantir compatibilidade entre Windows, Linux e Nuvem.
    """
    models_path = Path(models_dir)

    model = joblib.load(models_path / "kmeans_model.pkl")
    scaler = joblib.load(models_path / "scaler.pkl")
    encoder = joblib.load(models_path / "encoder.pkl")
    feature_columns = joblib.load(models_path / "feature_columns.pkl")

    return model, scaler, encoder, feature_columns


#PRÉ-PROCESSAMENTO

def preprocess_new_data(
    df: pd.DataFrame,
    encoder,
    scaler,
    feature_columns,
):
    """
    Passa os dados novos (seja 1 cliente ou um lote de 10.000) pelo mesmo pipeline
    de regras de negócio usada no treinamento.
    """
    #Regras de Negócio (Feature Engineering)
    df = clean_data(df)
    df = create_features(df)
    df = remove_redundant_columns(df)
    df = standardize_categories(df)

    #Transformação Categórica -> Numérica
    #fit_encoder=False garante que usamos os padrões do passado (não reaprende)
    df_ml, _ = prepare_ml_data(
        df,
        encoder=encoder,
        fit_encoder=False,
    )

    #ALINHAMENTO DE COLUNAS (
    # Garante que a ordem e a quantidade de colunas do dado novo sejam EXATAMENTE
    # iguais ao que o K-Means espera ver. Se faltar uma coluna One-Hot, preenche com 0.
    df_ml = df_ml.reindex(columns=feature_columns, fill_value=0)

    #Padronização Matemática
    scaled_data = scaler.transform(df_ml)

    #Retorna tanto a matriz (para o modelo) quanto o DF legível (para a tela)
    return scaled_data, df


#API DE PREDIÇÃO (O que o Streamlit vai chamar)

def predict_cluster(
    df: pd.DataFrame,
    models_dir: str = "models",
) -> pd.DataFrame:
    """
    Recebe um DataFrame cru e devolve o DataFrame com a classificação final do Cluster.
    """
    #Busca as ferramentas 
    model, scaler, encoder, feature_columns = load_artifacts(models_dir)

    #Prepara o dado
    scaled_data, df_processed = preprocess_new_data(
        df=df,
        encoder=encoder,
        scaler=scaler,
        feature_columns=feature_columns,
    )

    #Faz a Previsão
    clusters = model.predict(scaled_data)

    #Formata a Resposta Final
    result = df_processed.copy()
    result["Cluster"] = clusters
    
    #Adiciona a coluna com o nome de negócio para facilitar a leitura na UI
    result["Cluster_Nome"] = result["Cluster"].map(CLUSTER_NAMES)

    return result