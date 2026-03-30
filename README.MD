
## 📖 Visão Geral do Projeto
No cenário competitivo do varejo atual, campanhas de marketing genéricas ("one-size-fits-all") resultam em desperdício de orçamento e baixa conversão. O objetivo deste projeto é utilizar **Machine Learning Não Supervisionado** para descobrir padrões ocultos de comportamento de consumo e agrupar clientes em segmentos estratégicos.

Com base em dados transacionais e demográficos, o algoritmo identificou **4 perfis distintos de clientes**, permitindo que a equipe de Marketing substitua a intuição por decisões baseadas em dados (Data-Driven), direcionando ofertas e otimizando o Retorno sobre Investimento (ROI).

---

## 🛠️ Metodologia e Pipeline de Dados

O projeto foi estruturado seguindo as melhores práticas de Ciência de Dados:

1. **Análise Exploratória e Limpeza (EDA):** Tratamento de valores nulos, remoção de *outliers* anômalos e criação de novas variáveis (*Feature Engineering*), como `Tempo_Cliente_Dias` e `Gasto_Total`.
2. **Pré-processamento:** Transformação de variáveis categóricas (`OneHotEncoder`) e padronização de escalas (`StandardScaler`) para garantir que o algoritmo não fosse enviesado por grandezas numéricas diferentes (ex: Renda vs. Idade).
3. **Definição do K Ideal:** Utilização do **Método do Cotovelo (Elbow Method)** e validação da Inércia para determinar matematicamente a divisão em 4 clusters.
4. **Redução de Dimensionalidade:** Aplicação de **PCA (Principal Component Analysis)** para condensar as variáveis originais e permitir a visualização gráfica dos clusters em espaços 2D e 3D interativos.
5. **Modelagem Principal:** Treinamento do algoritmo **K-Means** utilizando as dimensões originais (preservando 100% da variância) para garantir precisão matemática na clusterização.
6. **Interpretabilidade do Modelo (XAI):** Treinamento de uma **Árvore de Decisão** sobre os clusters gerados para rankear o grau de importância das variáveis na decisão do algoritmo.

---

## 📊 Resultados e Perfis de Negócio (Os 4 Clusters)

O K-Means revelou perfeitamente as nuances da base de clientes. Os resultados foram cruzados com os dados reais para criar a seguinte tabela executiva de perfis:

* 👑 **Cluster 0 | VIPs Premium (A Mina de Ouro):** * **Perfil:** Alta renda e o maior ticket médio disparado (R$ 1.397,00). Compram com altíssima frequência e não possuem filhos dependentes em casa.
  * **Estratégia:** Exclusividade. Programas de fidelidade premium, vinhos e carnes nobres. Zero foco em descontos de massa.

* 🛒 **Cluster 2 | A Classe Média Engajada (O Motor da Loja):** * **Perfil:** Renda média/alta, gasto muito saudável (R$ 733,00) e alto engajamento. Geralmente possuem dependentes.
  * **Estratégia:** Campanhas de "pacotes família" e excelente custo-benefício. Sustentam o fluxo de caixa do dia a dia.

* 🏷️ **Cluster 1 | A Grande Massa (Caçadores de Ofertas):** * **Perfil:** Representam o maior volume da base (~50%), mas possuem o menor orçamento (Ticket médio < R$ 100,00). São mais jovens e possuem o maior número de filhos.
  * **Estratégia:** Queimas de estoque, cupons agressivos e produtos de entrada para garantir volume e retenção.

* 🔍 **Cluster 3 | Promissores (As Anomalias):** * **Perfil:** Grupo menor com renda declarada extremamente baixa, mas com engajamento e ticket médio (R$ 392,00) muito superiores à massa.
  * **Estratégia:** Investigação de perfil e nutrição com campanhas digitais para alavancar potencial.

---

## 🧠 Interpretabilidade: O Que Realmente Importa?

A análise de interpretabilidade da Árvore de Decisão trouxe um *insight* crucial para o negócio: **Demografia não dita o consumo nesta loja.**

Variáveis clássicas como Idade, Escolaridade e Estado Civil tiveram **0% de impacto** na formação dos clusters. O algoritmo priorizou exclusivamente o comportamento financeiro:
1. **Gasto_Total (27.3%)**
2. **Recency / Dias desde a última compra (23.2%)**
3. **Consumo de nichos específicos (ex: Peixes e Produtos Premium)**

---

## 📂 Estrutura do Repositório

```text
├── data/
│   ├── raw/                   # Dados originais e brutos
│   └── processed/             # Dados limpos e exportados (com as etiquetas dos clusters)
├── notebooks/
│   └── customer_segmentation.ipynb  # Notebook principal com toda a análise e modelagem
├── README.md                  # Documentação do projeto
````

## 👨‍💻 Autor

**Felipe Tamiozzo**

  * Estudante de Data Science em transição para a área de Dados.
  * Conecte-se comigo no [LinkedIn](https://linkedin.com/in/felipetamiozzo/)
  * Veja outros projetos no meu [GitHub](https://github.com/felipetamiozzo)

<!-- end list -->

```
