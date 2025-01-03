import plotly.express as px  # Biblioteca para criar gráficos interativos
import numpy as np  # Biblioteca para manipulação de arrays e cálculos matemáticos
from sklearn.preprocessing import StandardScaler  # Normaliza os dados para evitar distorções
from sklearn.cluster import DBSCAN  # Algoritmo de agrupamento baseado em densidade

# Conjunto de dados representando idade e salário de diferentes pessoas
base_salario = np.array([
    [20, 1000], [27, 1200], [21, 2900], [37, 1850], [46, 900],
    [53, 950], [55, 2000], [47, 2100], [52, 3000], [32, 5900],
    [39, 4100], [41, 5100], [39, 7000], [48, 5000], [48, 6500]
])

# Normaliza os dados para padronizar a escala das variáveis
scaler_salario = StandardScaler()
base_salario = scaler_salario.fit_transform(base_salario)

# Aplica o algoritmo DBSCAN para encontrar grupos (clusters)
dbscan_salario = DBSCAN(eps=0.95, min_samples=2)  # Define o raio e o número mínimo de pontos por cluster
dbscan_salario.fit(base_salario)

# Obtém os rótulos dos clusters identificados (-1 indica pontos não agrupados)
rotulos = dbscan_salario.labels_

# Cria um gráfico de dispersão mostrando os clusters
grafico = px.scatter(x=base_salario[:, 0], y=base_salario[:, 1], color=rotulos)
grafico.show()