import plotly.express as px  # Biblioteca para criar gráficos interativos
from sklearn.preprocessing import StandardScaler  # Ferramenta para padronizar dados
import pandas as pd  # Biblioteca para manipular dados em tabelas
import matplotlib.pyplot as plt  # Biblioteca para gerar gráficos
from scipy.cluster.hierarchy import dendrogram, linkage  # Ferramentas para clustering hierárquico
from sklearn.cluster import AgglomerativeClustering  # Algoritmo de agrupamento hierárquico

# Carrega os dados do arquivo CSV contendo informações dos clientes
base_cartao = pd.read_csv('credit_card_clients.csv', header=1)

# Cria uma nova coluna somando os valores das faturas dos últimos 6 meses
base_cartao['BILL_TOTAL'] = (
    base_cartao['BILL_AMT1'] + base_cartao['BILL_AMT2'] + 
    base_cartao['BILL_AMT3'] + base_cartao['BILL_AMT4'] + 
    base_cartao['BILL_AMT5'] + base_cartao['BILL_AMT6']
)

# Seleciona as colunas 'Limite de crédito' e 'Total de faturas'
x_cartao = base_cartao.iloc[:, [1, 25]].values

# Normaliza os dados para deixar na mesma escala
scaler_cartao = StandardScaler()
x_cartao = scaler_cartao.fit_transform(x_cartao)

# Cria o dendrograma para visualizar a estrutura dos clusters
dendrograma = dendrogram(linkage(x_cartao, method='ward'))
plt.show()

# Aplica o algoritmo de agrupamento hierárquico com 3 clusters
hc_cartao = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
rotulos = hc_cartao.fit_predict(x_cartao)

# Plota os clientes agrupados em um gráfico de dispersão
grafico = px.scatter(x=x_cartao[:, 0], y=x_cartao[:, 1], color=rotulos)
grafico.show()