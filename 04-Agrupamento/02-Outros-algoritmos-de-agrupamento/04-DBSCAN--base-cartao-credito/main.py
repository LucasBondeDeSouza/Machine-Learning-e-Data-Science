import plotly.express as px  # Biblioteca para criar gráficos interativos
from sklearn.preprocessing import StandardScaler  # Normaliza os dados para evitar distorções
import pandas as pd  # Biblioteca para manipular dados em tabelas
from scipy.cluster.hierarchy import dendrogram, linkage  # Ferramentas para clustering hierárquico
from sklearn.cluster import DBSCAN  # Algoritmo de agrupamento baseado em densidade

# Carrega os dados do arquivo CSV contendo informações dos clientes
base_cartao = pd.read_csv('credit_card_clients.csv', header=1)

# Cria uma nova coluna somando os valores das faturas dos últimos 6 meses
base_cartao['BILL_TOTAL'] = (
    base_cartao['BILL_AMT1'] + base_cartao['BILL_AMT2'] + 
    base_cartao['BILL_AMT3'] + base_cartao['BILL_AMT4'] + 
    base_cartao['BILL_AMT5'] + base_cartao['BILL_AMT6']
)

# Seleciona as colunas 'Limite de crédito' e 'Total de faturas' para análise
x_cartao = base_cartao.iloc[:, [1, 25]].values

# Normaliza os dados para padronizar a escala
scaler_cartao = StandardScaler()
x_cartao = scaler_cartao.fit_transform(x_cartao)

# Aplica o algoritmo DBSCAN para identificar grupos com base na densidade
dbscan_cartao = DBSCAN(eps=0.37, min_samples=5)
rotulos = dbscan_cartao.fit_predict(x_cartao)  # Gera os rótulos de cluster para cada cliente

# Cria um gráfico de dispersão para visualizar os clusters formados
grafico = px.scatter(x=x_cartao[:, 0], y=x_cartao[:, 1], color=rotulos)
grafico.show()