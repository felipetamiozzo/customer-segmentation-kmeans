# 📊 Segmentação de Clientes com Machine Learning

## 🚀 Visão Geral

Em um cenário altamente competitivo no varejo, campanhas de marketing genéricas ("one-size-fits-all") geram desperdício de orçamento e baixa conversão.

Este projeto utiliza **Machine Learning Não Supervisionado (K-Means)** para identificar padrões ocultos de comportamento e segmentar clientes em grupos estratégicos, permitindo decisões orientadas por dados (**Data-Driven Marketing**).

💡 O resultado é um sistema capaz de transformar dados brutos em **insights acionáveis**, aumentando eficiência de campanhas e potencializando o **Retorno sobre Investimento (ROI)**.

---

## 🎯 Problema de Negócio

Empresas frequentemente enfrentam:

* Baixa conversão em campanhas de marketing
* Desperdício de orçamento com público não qualificado
* Falta de personalização na comunicação com clientes
* Dificuldade em identificar clientes de alto valor

---

## 💡 Solução Proposta

Aplicação de um pipeline completo de Data Science para:

* Segmentar clientes com base em comportamento real
* Identificar perfis de consumo distintos
* Direcionar estratégias específicas por grupo
* Permitir previsão de segmento para novos clientes

---

## 📈 Impacto de Negócio (ROI)

Este projeto permite:

### 💰 Aumento de Receita

* Identificação de clientes **VIPs Premium** com maior ticket médio
* Possibilidade de campanhas exclusivas e aumento de LTV (Lifetime Value)

### 🎯 Otimização de Marketing

* Redução de desperdício com campanhas genéricas
* Segmentação precisa para campanhas direcionadas

### 📉 Redução de Custos

* Melhor alocação de budget em marketing
* Menor CAC (Custo de Aquisição de Cliente)

### 📊 Tomada de Decisão Estratégica

* Base de clientes segmentada e interpretável
* Insights claros para áreas de marketing e produto

---

## 🧠 Metodologia

O projeto segue boas práticas de Ciência de Dados:

### 🔍 1. Análise Exploratória (EDA)

* Tratamento de valores nulos
* Remoção de outliers
* Análise de distribuições

### ⚙️ 2. Feature Engineering

* Criação de variáveis como:

  * `Gasto_Total`
  * `Tempo_Cliente_Dias`
  * `Total_Compras`
  * `Total_Filhos`

### 🔄 3. Pré-processamento

* OneHotEncoder para variáveis categóricas
* StandardScaler para padronização

### 📐 4. Definição do K Ideal

* Método do Cotovelo (Elbow Method)
* Validação por inércia

### 🤖 5. Modelagem

* Algoritmo: **K-Means**
* Clusterização com base no comportamento do cliente

### 📉 6. Avaliação

* Silhouette Score: **0.21**
* Validação baseada em interpretabilidade de negócio

---

## 👥 Segmentos Identificados

### 👑 VIPs Premium (Alto Valor)

* Alta renda
* Maior gasto total
* Alta frequência de compra

📌 Estratégia:

* Programas de fidelidade premium
* Produtos exclusivos
* Personalização avançada

---

### 🏷️ Grande Massa / Caçadores de Ofertas

* Maior volume da base
* Baixo ticket médio
* Alta sensibilidade a preço

📌 Estratégia:

* Promoções agressivas
* Cupons e descontos
* Produtos de entrada

---

### 🛒 Classe Média Engajada

* Renda intermediária/alta
* Gasto consistente
* Alta recorrência

📌 Estratégia:

* Pacotes familiares
* Combos e ofertas
* Retenção

---

### 🔍 Promissores (Anomalias)

* Grupo pequeno
* Comportamento fora do padrão
* Alto potencial

📌 Estratégia:

* Campanhas digitais
* Nutrição de leads
* Análise individual

---

## 📊 Dashboard Interativo

O projeto inclui um dashboard desenvolvido com **Streamlit**, permitindo:

* Visualização dos clusters
* Análise de renda vs gasto
* Distribuição por segmento
* Perfil médio dos clientes
* Estratégias por cluster

### 🔮 Funcionalidade Extra (Diferencial)

✔ Previsão do segmento para novos clientes em tempo real

---

## 🛠️ Tecnologias Utilizadas

* Python
* Pandas / NumPy
* Scikit-learn
* Plotly
* Streamlit
* Joblib

---

## 📂 Estrutura do Projeto

```
customer-segmentation/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data_processing.py
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## ▶️ Como Executar

### 1. Criar ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Treinar o modelo

```bash
python src/train_model.py
```

### 4. Rodar o dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📌 Diferenciais do Projeto

✔ Pipeline completo de Data Science
✔ Aplicação prática orientada a negócio
✔ Dashboard interativo
✔ Previsão em tempo real
✔ Storytelling com foco em ROI
✔ Código modular (produção-ready)

---

## 👨‍💻 Autor

Felipe Tamiozzo

🔗 LinkedIn: https://linkedin.com/in/felipetamiozzo
🔗 GitHub: https://github.com/felipetamiozzo

---

## 📢 Conclusão

Este projeto demonstra como técnicas de Machine Learning podem ser aplicadas para resolver problemas reais de negócio, transformando dados em vantagem competitiva.

Mais do que segmentar clientes, o objetivo foi criar um **produto de dados**, capaz de gerar impacto direto em decisões estratégicas.
