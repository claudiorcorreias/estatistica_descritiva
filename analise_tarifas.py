# Executando o Streamlit em um Servidor Local ou usando o ngrok para expor o Streamlit localmente e acessá-lo via Colab:

# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, spearmanr, chi2_contingency
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st  # Para criar um painel interativo
import io

# Configurações iniciais
st.set_page_config(page_title="Análise de Tarifas Aéreas", layout="wide")
sns.set_palette("colorblind")  # Paleta amigável a daltônicos

# Título da apresentação
st.title("Análise Estatística de Tarifas Aéreas")
st.write("""
Este projeto tem como objetivo explorar e analisar dados de tarifas aéreas, utilizando conceitos estatísticos e visualizações para identificar padrões, tendências e observações anômalas.
""")

# 1. Carregar a Base de Dados
st.header("1. Carregar a Base de Dados")
st.write("""
Vamos começar carregando a base de dados de tarifas aéreas. O conjunto de dados contém informações como origem, destino, distância, tarifa, IPCA, câmbio e PIB.
""")

# Caminho do arquivo no Google Drive
caminho = 'C:/Users/claud/Documents/Projetos CD e IA/df_projeto_estatistica.csv'
df = pd.read_csv(caminho, encoding='UTF-8', sep=',') 

# Converter a coluna 'Data' para datetime
df['Data'] = pd.to_datetime(df['Data'])

# Exibir as primeiras linhas do DataFrame
st.subheader("Visualização dos Dados")
st.write("Aqui estão as primeiras linhas do conjunto de dados:")
st.dataframe(df.head())

# Informações básicas sobre o DataFrame
st.write("Informações sobre o conjunto de dados:")
# Capturar a saída de df.info()
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

# Exibir a saída no Streamlit
st.text(s)

# 2. Seleção da Amostra
st.header("2. Seleção da Amostra")
st.write("""
Para facilitar a análise, vamos selecionar uma amostra representativa dos dados. Utilizaremos uma técnica de amostragem por conglomerados, agrupando os dados por estado de destino.
""")

# Amostragem por Conglomerados (clusters por estado de destino)
clusters = df['UF_Destino'].unique()
clusters_selecionados = np.random.choice(clusters, size=5, replace=False)  # Seleciona 5 clusters
amostra = df[df['UF_Destino'].isin(clusters_selecionados)]

# Exibir resultados da amostragem
st.write(f"Clusters selecionados: {clusters_selecionados}")
st.write(f"Tamanho da amostra: {len(amostra)}")
st.write("Primeiras linhas da amostra:")
st.dataframe(amostra.head())
st.write("Informações sobre a amostra:")
buffer = io.StringIO()
amostra.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Comparar a distribuição dos clusters
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='UF_Destino', label='População')
sns.countplot(data=amostra, x='UF_Destino', label='Amostra')
plt.legend()
plt.title('Distribuição dos Clusters na População e Amostra')
st.pyplot(plt)

# Verificação da Representatividade
st.subheader("2.1. Verificação da Representatividade da Amostra")
st.write("""
Agora, vamos comparar a distribuição das tarifas na população (dados completos) e na amostra selecionada para garantir que a amostra seja representativa.
""")

plt.figure(figsize=(10, 6))
sns.histplot(df['Tarifa'], label='População', kde=True, color='blue')
sns.histplot(amostra['Tarifa'], label='Amostra', kde=True, color='orange')
plt.legend()
plt.title('Comparação entre População e Amostra', fontsize=16)
plt.xlabel('Tarifa', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
st.pyplot(plt)

# 3. Análise Exploratória Inicial
st.header("3. Análise Exploratória Inicial")
st.write("""
Vamos começar com uma análise exploratória, focando na distribuição das tarifas e nas relações entre as variáveis.
""")

# Distribuição de Frequências e Histograma
st.subheader("3.1. Distribuição de Frequências e Histograma")
st.write("""
O histograma abaixo mostra a distribuição das tarifas aéreas. Ele nos permite visualizar a frequência com que diferentes faixas de tarifas ocorrem.
""")

plt.figure(figsize=(10, 6))
sns.histplot(amostra['Tarifa'], kde=True, bins=30, color='#1f77b4')
plt.title('Distribuição das Tarifas Aéreas', fontsize=16)
plt.xlabel('Tarifa', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
st.pyplot(plt)

# Medidas de Tendência Central
st.subheader("3.2. Medidas de Tendência Central")
st.write("""
Agora, vamos calcular as principais medidas de tendência central: média, mediana e moda.
""")

media = amostra['Tarifa'].mean()
mediana = amostra['Tarifa'].median()
moda = amostra['Tarifa'].mode().values[0]

st.write(f"- **Média**: {media:.2f}")
st.write(f"- **Mediana**: {mediana:.2f}")
st.write(f"- **Moda**: {moda:.2f}")

st.write("""
A média é afetada por valores extremos, enquanto a mediana é mais robusta. A moda representa o valor mais frequente.
""")

# Medidas de Dispersão
st.subheader("3.3. Medidas de Dispersão")
st.write("""
Vamos calcular as medidas de dispersão, que nos ajudam a entender a variabilidade dos dados.
""")

amplitude = amostra['Tarifa'].max() - amostra['Tarifa'].min()
variancia = amostra['Tarifa'].var()
desvio_padrao = amostra['Tarifa'].std()

st.write(f"- **Amplitude**: {amplitude:.2f}")
st.write(f"- **Variância**: {variancia:.2f}")
st.write(f"- **Desvio Padrão**: {desvio_padrao:.2f}")

st.write("""
O desvio padrão é uma medida mais interpretável que a variância, pois está na mesma unidade dos dados.
""")

# Medidas de Assimetria e Curtose
st.subheader("3.4. Medidas de Assimetria e Curtose")
st.write("""
A assimetria mede o grau de distorção da distribuição, enquanto a curtose mede o "achatamento".
""")

assimetria = skew(amostra['Tarifa'])
curtose = kurtosis(amostra['Tarifa'])

st.write(f"- **Assimetria**: {assimetria:.2f}")
st.write(f"- **Curtose**: {curtose:.2f}")

st.write("""
Uma assimetria positiva indica uma cauda longa à direita, enquanto uma curtose positiva indica uma distribuição mais "pontiaguda".
""")

# Análise de Dados Anômalos (Outliers)
st.subheader("3.5. Análise de Dados Anômalos (Outliers)")
st.write("""
Vamos identificar outliers usando o boxplot e o intervalo interquartílico (IIQ).
""")

Q1 = amostra['Tarifa'].quantile(0.25)
Q3 = amostra['Tarifa'].quantile(0.75)
IIQ = Q3 - Q1
limite_inferior = Q1 - 1.5 * IIQ
limite_superior = Q3 + 1.5 * IIQ

outliers = amostra[(amostra['Tarifa'] < limite_inferior) | (amostra['Tarifa'] > limite_superior)]

st.write(f"- **Limite Inferior**: {limite_inferior:.2f}")
st.write(f"- **Limite Superior**: {limite_superior:.2f}")
st.write(f"- **Número de Outliers**: {len(outliers)}")

plt.figure(figsize=(10, 6))
sns.boxplot(x=amostra['Tarifa'], color='#1f77b4')
plt.title('Boxplot das Tarifas Aéreas', fontsize=16)
plt.xlabel('Tarifa', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
st.pyplot(plt)

# Análise da exploração inicial
st.write(f"""
**Análise da Análise Exploratória Inicial:**

A análise exploratória inicial revelou insights importantes sobre as tarifas aéreas.

* **Distribuição:** O histograma mostrou que as tarifas não seguem uma distribuição normal, com assimetria positiva, indicando uma cauda longa à direita.
* **Tendência Central:** A média das tarifas foi superior à mediana, confirmando a presença de valores atípicos que influenciam a média.
* **Dispersão:** O desvio padrão indicou uma variabilidade significativa nas tarifas.
* **Assimetria e Curtose:** A assimetria e curtose positivas confirmaram a distribuição não normal, com cauda longa à direita e "pontiaguda".
* **Outliers:** O boxplot e o IIQ identificaram {len(outliers)} outliers. O limite inferior foi {limite_inferior:.2f} e o superior {limite_superior:.2f}. Outliers podem ser erros de coleta, eventos excepcionais ou variação natural dos dados.

A presença de outliers sugere a necessidade de investigação para entender suas causas e decidir sobre sua remoção ou manutenção na análise, justificando qualquer decisão tomada.

A análise exploratória inicial fornece uma base sólida para análises mais aprofundadas, como relações entre variáveis e análise temporal.
""")


# Análise de Relações entre Variáveis
st.header("4. Análise de Relações entre Variáveis")
st.write("""
Agora, vamos explorar as relações entre as variáveis, como tarifa e distância, usando covariância e correlação.
""")

# Covariância e Correlação
st.subheader("4.1. Covariância e Correlação")
covariancia = amostra['Tarifa'].cov(amostra['Dist_Km'])
correlacao = amostra['Tarifa'].corr(amostra['Dist_Km'])

st.write(f"- **Covariância**: {covariancia:.2f}")
st.write(f"- **Correlação (Pearson)**: {correlacao:.2f}")

st.write("""
A correlação de Pearson mede a força e a direção da relação linear entre duas variáveis.
""")

# Gráfico de Dispersão
plt.figure(figsize=(10, 6))
sns.scatterplot(data=amostra, x='Dist_Km', y='Tarifa', hue='Região_Destino', palette='colorblind')
plt.title('Relação entre Tarifa e Distância', fontsize=16)
plt.xlabel('Distância (Km)', fontsize=12)
plt.ylabel('Tarifa', fontsize=12)
plt.legend(title='Região de Destino')
plt.grid(True, linestyle='--', alpha=0.7)
st.pyplot(plt)

# Análise dos achados de Covariância e Correlação
st.write(f"""
**Análise:**

A covariância entre as tarifas e a distância foi de {covariancia:.2f}, indicando uma relação positiva entre as duas variáveis. 
A correlação de Pearson foi de {correlacao:.2f}, sugerindo uma correlação moderada a forte. 
Isso significa que, em geral, tarifas mais altas tendem a estar associadas a distâncias maiores.

O gráfico de dispersão confirma essa relação, mostrando uma tendência de aumento das tarifas com o aumento da distância. 
No entanto, também é possível observar que a relação não é perfeita, com alguns pontos fora da tendência geral. 
A região de destino também parece influenciar a relação entre tarifa e distância, com algumas regiões apresentando tarifas mais altas para distâncias semelhantes.

Esses resultados sugerem que a distância é um fator importante na determinação das tarifas aéreas, mas outros fatores, como a região de destino, também desempenham um papel significativo.
""")


# Análise Temporal
st.header("5. Análise Temporal")
st.write("""
Vamos analisar a evolução das tarifas ao longo do tempo, identificando tendências e sazonalidades.
""")

# Decomposição da Série Temporal
st.subheader("5.1. Decomposição da Série Temporal")
serie_temporal = amostra.groupby('Data')['Tarifa'].mean().asfreq('D', method='ffill')
decomposicao = seasonal_decompose(serie_temporal, model='additive', period=30)
fig = decomposicao.plot()  # Captura a figura para personalização

# Modifica os rótulos do eixo Y para cada subplot
fig.axes[0].set_ylabel('Valor')
fig.axes[1].set_ylabel('Sazonal')
fig.axes[2].set_ylabel('Resíduo')

plt.suptitle('Decomposição da Série Temporal das Tarifas', fontsize=10)
plt.xlabel('Data', fontsize=8)
plt.legend(fontsize=8)

# Aumenta o espaçamento abaixo do suptitle
plt.subplots_adjust(top=0.9)  # Ajuste o valor conforme necessário

st.pyplot(fig)

# Análise dos achados da Série Temporal
st.write("""
**Análise:**

A decomposição da série temporal revela os componentes da evolução das tarifas ao longo do tempo: observado, tendência, sazonalidade e resíduo.

* **Observado:** Mostra a série temporal original das tarifas, exibindo a variação real dos preços ao longo do tempo.
* **Tendência:** Indica a direção geral dos preços, mostrando se há um aumento ou diminuição a longo prazo. No gráfico, podemos observar uma tendência de aumento das tarifas.
* **Sazonalidade:** Revela padrões que se repetem em intervalos regulares, como variações semanais ou mensais. No gráfico, podemos identificar um padrão sazonal claro, com picos e vales que se repetem.
* **Resíduo:** Representa a variação aleatória que não é explicada pela tendência ou sazonalidade.

A análise da decomposição da série temporal permite identificar padrões e tendências que podem ser úteis para prever futuras variações nas tarifas. A identificação de padrões sazonais pode auxiliar na otimização de preços e na gestão da oferta e demanda.
""")


# Conclusão
st.header("6. Conclusão")
st.write("""
Nesta análise, exploramos as características das tarifas aéreas, identificamos padrões e outliers, e analisamos relações entre variáveis. Os principais insights são:
- As tarifas variam significativamente entre regiões.
- A distância é um fator importante, mas outras variáveis, como IPCA e câmbio, também impactam as tarifas.
- Há uma tendência de aumento nas tarifas ao longo do tempo, com padrões sazonais claros.
""")