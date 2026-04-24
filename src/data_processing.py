# src/feature_engineering.py
import pandas as pd
import numpy as np


PRODUCT_COLUMNS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]

PURCHASE_COLUMNS = [
    "NumCatalogPurchases", "NumStorePurchases", "NumWebPurchases",
]

CAMPAIGN_COLUMNS = [
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
    "AcceptedCmp4", "AcceptedCmp5", "Response",
]


#FUNÇÕES DE EXTRAÇÃO E LIMPEZA (Data Cleaning)


def load_data(path: str) -> pd.DataFrame:
    """
    Carrega os dados brutos. 
    Usa type hinting (-> pd.DataFrame) para indicar que essa função SEMPRE retorna um dataframe.
    """
    
    return pd.read_csv(path, sep=None, engine="python")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza estrutural: remove nulos, formata datas e ajusta o índice.
    """
    
    df = df.copy()
    
    #Remove clientes que não preencheram a renda (como vimos, são apenas ~1% da base)
    df = df.dropna(subset=["Income"])
    
    #Converte texto para o tipo datetime do Pandas para podermos fazer contas com datas
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    
    #Transforma a coluna ID no índice (linha) do dataframe, pois ID não é variável matemática
    if "ID" in df.columns:
        df = df.set_index("ID")
        
    #Remove colunas que a sua EDA provou terem variância zero (não servem para o modelo)
    
    df = df.drop(columns=["Z_CostContact", "Z_Revenue"], errors="ignore")
    
    return df


#ENGENHARIA DE RECURSOS (Feature Engineering)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a inteligência de negócios transformando colunas brutas em KPIs (Gasto, Idade, etc).
    """
    df = df.copy()
    
    #Pega o ano e a data exata do cliente mais recente para usar como base de cálculo
    ano_referencia = df["Dt_Customer"].dt.year.max()
    data_referencia = df["Dt_Customer"].max()
    
    #Cálculo da Idade atual do cliente (Ano base - Ano de nascimento)
    df["Idade"] = ano_referencia - df["Year_Birth"]
    
    
    #Soma todas as colunas da lista na horizontal (linha por linha) 
    df["Gasto_Total"] = df[PRODUCT_COLUMNS].sum(axis=1)
    df["Total_Filhos"] = df["Kidhome"] + df["Teenhome"]
    df["Total_Compras"] = df[PURCHASE_COLUMNS].sum(axis=1)
    df["Total_Campanhas_Aceitas"] = df[CAMPAIGN_COLUMNS].sum(axis=1)
    
    #Calcula a diferença em dias para saber a 'maturidade' (fidelidade) do cliente
    df["Tempo_Cliente_Dias"] = (data_referencia - df["Dt_Customer"]).dt.days
    
    return df

def remove_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas originais que já foram transformadas para evitar Multicolinearidade 
    (dar peso em dobro para a mesma informação no algoritmo K-Means).
    """
    df = df.copy()
    redundant_columns = ["Year_Birth", "Dt_Customer", "Kidhome", "Teenhome"]
    
    #Junta a lista de redundantes com a lista de campanhas e apaga tudo de uma vez
    df = df.drop(columns=redundant_columns + CAMPAIGN_COLUMNS, errors="ignore")
    return df


#TRATAMENTO ESTATÍSTICO E CATEGÓRICO


def remove_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove anomalias extremas usando a matemática do Intervalo Interquartil (IQR).
    
    """
    df_clean = df.copy()
    
    #Se o usuário não passar uma lista de colunas, por padrão limpamos Income e Idade
    if columns is None:
        columns = ["Income", "Idade"]
        
    for col in columns:
        #Calcula o Primeiro Quartil (25%) e o Terceiro Quartil (75%)
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1 # A distância entre a massa principal de dados
        
        #Define os limites
        lower_limit = q1 - factor * iqr
        upper_limit = q3 + factor * iqr
        
        #Filtra o dataframe mantendo apenas quem está dentro dos limites
        df_clean = df_clean[df_clean[col].between(lower_limit, upper_limit)]
        
    return df_clean

def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa "lixos" de digitação e agrupa categorias para criar grupos maiores e com mais peso estatístico.
    """
    df = df.copy()
    
    #Padronização do Estado Civil (Reduz de várias categorias para apenas 2 agrupamentos lógicos)
    single_terms = ["Single", "Divorced", "Widow", "Alone", "YOLO", "Absurd"]
    partner_terms = ["Married", "Together"]
    
    
    conditions = [
        df["Marital_Status"].isin(single_terms),
        df["Marital_Status"].isin(partner_terms),
    ]
    choices = ["Solteiro", "Com Parceiro"]
    df["Marital_Status"] = np.select(conditions, choices, default="Solteiro")
    
    #Padronização de Escolaridade (Usa um dicionário para traduzir e agrupar nomes similares)
    df["Education"] = df["Education"].replace({
        "2n Cycle": "Mestrado",
        "Master": "Mestrado",
        "Graduation": "Graduação",
        "Basic": "Básico",
        "PhD": "Doutorado",
    })
    
    return df


#ORQUESTRAÇÃO (Função Principal / Pipeline)


def run_processing_pipeline(input_path: str, output_path: str = None):
    df = load_data(input_path)
    df = clean_data(df)
    df = create_features(df)
    df = remove_redundant_columns(df)
    df = remove_outliers_iqr(df)
    df = standardize_categories(df)

    if output_path:
        df.to_csv(output_path, index=True)

    return df