import plotly.express as px  # Biblioteca para criar gráficos interativos
import plotly.graph_objects as go  # Biblioteca para gráficos personalizados
import numpy as np  # Biblioteca para manipulação de arrays e cálculos matemáticos
from sklearn.preprocessing import StandardScaler  # Normaliza os dados para evitar distorções
import matplotlib.pyplot as plt  # Biblioteca para gerar gráficos
from scipy.cluster.hierarchy import dendrogram, linkage  # Ferramentas para clustering hierárquico
from sklearn.cluster import AgglomerativeClustering  # Algoritmo de agrupamento hierárquico

# Dados representando idade e salário de diferentes pessoas
base_salario = np.array([
    [20, 1000], [27, 1200], [21, 2900], [37, 1850], [46, 900],
    [53, 950], [55, 2000], [47, 2100], [52, 3000], [32, 5900],
    [39, 4100], [41, 5100], [39, 7000], [48, 5000], [48, 6500]
])

# Padroniza os dados para que todas as variáveis tenham a mesma escala
scaler_salario = StandardScaler()
base_salario = scaler_salario.fit_transform(base_salario)

# Cria um dendrograma para visualizar a formação dos clusters
dendrograma = dendrogram(linkage(base_salario, method='ward'))
plt.title('Dendrograma')  # Título do gráfico
plt.xlabel('Pessoas')  # Eixo X representa as pessoas
plt.ylabel('Distância')  # Eixo Y mostra a distância entre os clusters
plt.show()

# Aplica o algoritmo de agrupamento hierárquico, dividindo os dados em 3 grupos
hc_salario = AgglomerativeClustering(n_clusters=3, linkage='ward', metric='euclidean')
rotulos = hc_salario.fit_predict(base_salario)  # Gera os rótulos dos clusters para cada pessoa

# Gera um gráfico de dispersão colorido com base nos grupos identificados
grafico = px.scatter(x=base_salario[:, 0], y=base_salario[:, 1], color=rotulos)
grafico.show()