#app/streamlit_app.py
import pandas as pd
import plotly.express as px
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from predict import predict_cluster


#CONFIGURAÇÃO DA PÁGINA (primeiro comando do app)

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide", #Usa a tela inteira
)

#Constante de negócio
CLUSTER_NAMES = {
    0: "VIPs Premium",
    1: "Grande Massa / Caçadores de Ofertas",
    2: "Classe Média Engajada",
    3: "Promissores",
}


#CARREGAMENTO DE DADOS COM CACHE (Performance)

#@st.cache_data avisa ao Streamlit para ler o CSV apenas na primeira vez.
#Nas próximas interações (como clicar em filtros), ele puxa da memória RAM instantaneamente.
@st.cache_data
def load_segmented_data(
    path: str = "data/processed/customer_segmentation_clustered.csv",
):
    df = pd.read_csv(path)

    #Garante o tipo do dado e já mapeia o nome para os gráficos
    if "Cluster" in df.columns:
        df["Cluster"] = df["Cluster"].astype(int)
        df["Cluster_Nome"] = df["Cluster"].map(CLUSTER_NAMES)

    return df


#INTERFACE VISUAL (Header e Tratamento de Erros)

st.title("📊 Segmentação de Clientes com Machine Learning")
st.caption("Projeto de Data Science focado em estratégia de marketing orientada a dados")

st.markdown(
    "Dashboard interativo para segmentação de clientes utilizando Machine Learning "
    "não supervisionado (K-Means), com foco em estratégias de marketing orientadas a dados."
)

#Proteção de Produção: Se alguém rodar o app antes de treinar o modelo,
#ele não quebra com erro, mas exibe um aviso amigável ensinando o que fazer.
try:
    df = load_segmented_data()
except FileNotFoundError:
    st.error("Arquivo segmentado não encontrado. Execute primeiro: `python -m src.train_model`")
    st.stop() #Interrompe o script aqui para não dar erro nos gráficos abaixo


#KPIs GERAIS (Métricas de Topo)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Clientes", len(df))

with col2:
    st.metric("Clusters Identificados", df["Cluster"].nunique())

with col3:
    #Formatação limpa da moeda
    st.metric("Gasto Médio Geral", f"R$ {df['Gasto_Total'].mean():,.2f}")

with col4:
    st.metric("Cluster mais valioso", "VIPs Premium")
    
st.success("💎 O cluster VIPs Premium concentra os clientes de maior valor da base.")

st.divider()


#BARRA LATERAL (Filtros interativos)

st.sidebar.header("Filtros")


#Permite ao usuário focar a análise em nichos específicos (ex: só VIPs)
cluster_options = sorted(df["Cluster_Nome"].unique())

selected_clusters = st.sidebar.multiselect(
    "Selecione os clusters",
    options=cluster_options,
    default=cluster_options,
)

df_filtered = df[df["Cluster_Nome"].isin(selected_clusters)]


#GRÁFICOS INTERATIVOS (Plotly)

# Gráficos com Plotly são superiores ao Matplotlib para web


st.subheader("Distribuição dos Clientes por Cluster")

#Agrupamento para contar quantos clientes caíram em cada cluster
cluster_count = (
    df_filtered.groupby(["Cluster", "Cluster_Nome"])
    .size()
    .reset_index(name="Qtd_Clientes")
)

cluster_count = cluster_count.sort_values("Qtd_Clientes", ascending=False)

fig_count = px.bar(
    cluster_count,
    x="Cluster_Nome",
    y="Qtd_Clientes",
    color="Cluster_Nome",
    text="Qtd_Clientes",
    title="Quantidade de Clientes por Segmento",
)
st.plotly_chart(fig_count, use_container_width=True)


st.subheader("Renda vs. Gasto Total")
#Gráfico de dispersão é o coração da análise. 
#Adicionar o 'hover_data' para investigar outliers passando o mouse.
fig_scatter = px.scatter(
    df_filtered,
    x="Income",
    y="Gasto_Total",
    color="Cluster_Nome",
    hover_data=[
        "Idade",
        "Total_Filhos",
        "Total_Compras",
        "Tempo_Cliente_Dias",
    ],
    title="Mapeamento de Clientes por Renda e Gasto Total",
)
st.plotly_chart(fig_scatter, use_container_width=True)


#TABELAS ANALÍTICAS E EXPORTAÇÃO

st.subheader("Perfil Médio dos Segmentos")

#Tabela agregada focada no negócio (Como é o cliente médio de cada grupo?)
profile_cols = [
    "Income", "Gasto_Total", "Total_Compras",
    "Total_Filhos", "Idade", "Tempo_Cliente_Dias",
]

profile_table = (
    df_filtered.groupby(["Cluster", "Cluster_Nome"])[profile_cols]
    .mean()
    .round(2)
    .reset_index()
)
st.dataframe(profile_table, use_container_width=True)


st.subheader("📌 Estratégia Recomendada por Segmento")

st.markdown("""
### 👑 VIPs Premium
Clientes com maior renda, maior gasto total e alta frequência de compra.

**Estratégia recomendada:**  
Ofertas exclusivas, programa de fidelidade premium, produtos nobres e campanhas personalizadas.

---

### 🏷️ Grande Massa / Caçadores de Ofertas
Maior volume da base, porém com menor gasto médio e menor frequência de compra.

**Estratégia recomendada:**  
Cupons, descontos agressivos, campanhas promocionais e produtos de entrada.

---

### 🛒 Classe Média Engajada
Clientes com renda intermediária/alta, bom gasto total e alta frequência de compra.

**Estratégia recomendada:**  
Pacotes família, campanhas de custo-benefício, combos e ações de retenção.

---

### 🔍 Promissores
Grupo pequeno, com comportamento fora do padrão: renda menor, mas gasto e engajamento acima da massa.

**Estratégia recomendada:**  
Campanhas digitais, nutrição, análise individual e testes de ofertas personalizadas.
""")


st.subheader("🔮 Previsão de Segmento para Novo Cliente")

st.write(
    "Insira os dados de um novo cliente para estimar em qual segmento ele se encaixa."
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    renda = st.number_input("Renda anual", min_value=0, value=50000, step=1000)
    idade = st.number_input("Idade", min_value=18, max_value=100, value=35)
    educacao = st.selectbox(
        "Escolaridade",
        ["Básico", "Graduação", "Mestrado", "Doutorado"],
    )

with col_b:
    estado_civil = st.selectbox(
        "Estado civil",
        ["Solteiro", "Com Parceiro"],
    )
    filhos = st.number_input("Total de filhos/dependentes", min_value=0, max_value=5, value=1)
    recency = st.number_input("Dias desde a última compra", min_value=0, value=30)

with col_c:
    gasto_vinhos = st.number_input("Gasto com vinhos", min_value=0, value=300)
    gasto_carnes = st.number_input("Gasto com carnes", min_value=0, value=300)
    gasto_outros = st.number_input("Gasto em outros produtos", min_value=0, value=200)


if st.button("Prever Segmento"):
    ano_nascimento = 2024 - idade

    novo_cliente = pd.DataFrame({
        "ID": [999999],
        "Year_Birth": [ano_nascimento],
        "Education": [educacao],
        "Marital_Status": [estado_civil],
        "Income": [renda],
        "Kidhome": [filhos],
        "Teenhome": [0],
        "Dt_Customer": ["01-01-2024"],
        "Recency": [recency],
        "MntWines": [gasto_vinhos],
        "MntFruits": [gasto_outros / 5],
        "MntMeatProducts": [gasto_carnes],
        "MntFishProducts": [gasto_outros / 5],
        "MntSweetProducts": [gasto_outros / 5],
        "MntGoldProds": [gasto_outros / 5],
        "NumDealsPurchases": [2],
        "NumWebPurchases": [4],
        "NumCatalogPurchases": [3],
        "NumStorePurchases": [5],
        "NumWebVisitsMonth": [5],
        "AcceptedCmp3": [0],
        "AcceptedCmp4": [0],
        "AcceptedCmp5": [0],
        "AcceptedCmp1": [0],
        "AcceptedCmp2": [0],
        "Complain": [0],
        "Z_CostContact": [3],
        "Z_Revenue": [11],
        "Response": [0],
    })

    resultado = predict_cluster(novo_cliente)

    cluster_nome = resultado["Cluster_Nome"].iloc[0]
    cluster_numero = resultado["Cluster"].iloc[0]

    st.success(f"Cliente classificado no Cluster {cluster_numero}: **{cluster_nome}**")

st.subheader("📦 Distribuição do Gasto Total por Segmento")

fig_box = px.box(
    df_filtered,
    x="Cluster_Nome",
    y="Gasto_Total",
    color="Cluster_Nome",
    title="Distribuição do Gasto Total por Cluster",
)

st.plotly_chart(fig_box, use_container_width=True)

st.info(
    "O cluster 'Promissores' representa um grupo pequeno com comportamento atípico. Pode indicar clientes com potencial de crescimento, inconsistências na base ou necessidade de análise individual mais aprofundada."
)


#Exibe a base segmentada para o usuário auditar
st.subheader("📊 Base de Clientes Segmentada")
st.dataframe(df_filtered, use_container_width=True)

#Transforma o DataFrame em CSV diretamente na memória para download
csv = df_filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Baixar base filtrada em CSV",
    data=csv,
    file_name="clientes_segmentados.csv",
    mime="text/csv",
)