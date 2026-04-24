# src/train_model.py
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Importa noss pipeline de limpeza e criação de variáveis
from data_processing import run_processing_pipeline


#MAPEAMENTOS E CONSTANTES

# Ordinal Encoding: Transforma categorias textuais que possuem uma 
# ordem hierárquica clara em números que o algoritmo consegue pesar
EDUCATION_MAP = {
    "Básico": 0,
    "Graduação": 1,
    "Mestrado": 2,
    "Doutorado": 3,
}


#PREPARAÇÃO ESPECÍFICA PARA MACHINE LEARNING (Encoding)

def prepare_ml_data(
    df: pd.DataFrame,
    encoder: OneHotEncoder | None = None,
    fit_encoder: bool = True,
):
    """
    Converte variáveis categóricas em numéricas e lida com o OneHotEncoder.
    Recebe 'fit_encoder' para sabermos se estamos treinando (True) ou prevendo no Streamlit (False).
    """
    df_ml = df.copy()

    #Aplica o mapeamento ordinal na Educação
    df_ml["Education"] = df_ml["Education"].map(EDUCATION_MAP)

    # Configura o OneHotEncoder se ele não for passado
    if encoder is None:
        encoder = OneHotEncoder(
            drop="first", # Evita multicolinearidade (Dummy Variable Trap)
            sparse_output=False, # Retorna um array numpy padrão
            dtype=int,
            handle_unknown="ignore", # Protege o modelo em produção contra dados novos inesperados
        )

    # Se estivermos treinando, usamos fit_transform. Se for produção, apenas transform.
    if fit_encoder:
        encoded_data = encoder.fit_transform(df_ml[["Marital_Status"]])
    else:
        encoded_data = encoder.transform(df_ml[["Marital_Status"]])

    #Recupera os nomes das colunas criadas pelo Encoder (ex: Marital_Status_Com Parceiro)
    encoded_cols = encoder.get_feature_names_out(["Marital_Status"])

    #Transforma o array de volta em um DataFrame
    df_encoded = pd.DataFrame(
        encoded_data,
        columns=encoded_cols,
        index=df_ml.index,
    )

    #Remove a coluna original de texto e anexa as novas colunas binárias
    df_ml = pd.concat(
        [df_ml.drop(columns=["Marital_Status"]), df_encoded],
        axis=1,
    )

    return df_ml, encoder


#PIPELINE DE TREINAMENTO E EXPORTAÇÃO

def train_kmeans(
    input_path: str = "data/raw/customer_segmentation.csv",
    processed_path: str = "data/processed/customer_segmentation_processed.csv",
    models_dir: str = "models",
    n_clusters: int = 4, # Reflete a configuração ideal descoberta na EDA
    random_state: int = 42,
):
    """
    Orquestra todo o processo: puxa os dados, prepara, treina o modelo, 
    avalia a métrica (Silhouette) e exporta os artefatos (.pkl) para deploy.
    """
    #Usando pathlib para garantir que as pastas existam de forma segura e cross-platform
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)

    #Roda nosso pipeline de regras de negócio
    df_clean = run_processing_pipeline(
        input_path=input_path,
        output_path=processed_path,
    )

    #Converte categorias em números
    df_ml, encoder = prepare_ml_data(df_clean)

    #Padronização matemática
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_ml)

    #Mantemos como DataFrame para ter clareza de quais colunas estão indo para o modelo
    df_scaled = pd.DataFrame(
        scaled_data,
        columns=df_ml.columns,
        index=df_ml.index,
    )

    #Treinamento do K-Means
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    model.fit(df_scaled)

    #Avaliação do Modelo
    labels = model.labels_
    silhouette = silhouette_score(df_scaled, labels)

    #Salva o dataset final com a marcação de qual cluster cada cliente pertence
    df_segmented = df_clean.copy()
    df_segmented["Cluster"] = labels
    df_segmented.to_csv(
        "data/processed/customer_segmentation_clustered.csv",
        index=True,
    )

    #Salva todos os artefatos necessários para a aplicação Streamlit
    joblib.dump(model, models_path / "kmeans_model.pkl")
    joblib.dump(scaler, models_path / "scaler.pkl")
    joblib.dump(encoder, models_path / "encoder.pkl")
    
    #Salvar as colunas garante que a interface Web passe os dados na ordem exata
    joblib.dump(list(df_ml.columns), models_path / "feature_columns.pkl")

    print("Modelo treinado com sucesso!")
    print(f"Silhouette Score: {silhouette:.3f}")
    print("Arquivo segmentado salvo em: data/processed/customer_segmentation_clustered.csv")

    return model, scaler, encoder, df_segmented

if __name__ == "__main__":
    train_kmeans()