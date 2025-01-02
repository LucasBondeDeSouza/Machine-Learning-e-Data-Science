import plotly.express as px  # Biblioteca para criar gráficos interativos
import plotly.graph_objects as go  # Biblioteca para criar gráficos mais personalizados
import numpy as np  # Biblioteca para manipulação de arrays e cálculos matemáticos
from sklearn.preprocessing import StandardScaler  # Ferramenta para padronizar dados
from sklearn.cluster import KMeans  # Algoritmo de agrupamento

# Define os dados de idade (x) e salário (y) manualmente
x = [20, 27, 21, 37, 46, 53, 55, 47, 52, 32, 39, 41, 39, 48, 48]  # Idades
y = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500]  # Salários

# Cria um gráfico de dispersão para visualizar a relação entre idade e salário
grafico = px.scatter(x=x, y=y)  # Gráfico com idades no eixo X e salários no eixo Y
grafico.show()  # Exibe o gráfico

# Converte os dados de idade e salário em um array 2D para processamento
base_salario = np.array([
    [20, 1000], [27, 1200], [21, 2900], [37, 1850], [46, 900],
    [53, 950], [55, 2000], [47, 2100], [52, 3000], [32, 5900],
    [39, 4100], [41, 5100], [39, 7000], [48, 5000], [48, 6500]
])

# Padroniza os dados (transforma para ter média 0 e desvio padrão 1)
scaler_salario = StandardScaler()  # Inicializa o escalador
base_salario = scaler_salario.fit_transform(base_salario)  # Aplica a padronização

# Configura o modelo K-Means para dividir os dados em 3 grupos (clusters)
kmeans_salario = KMeans(n_clusters=3)  # Cria o modelo com 3 clusters
kmeans_salario.fit(base_salario)  # Ajusta o modelo aos dados padronizados

# Obtém os centroides dos clusters encontrados
centroides = kmeans_salario.cluster_centers_  # Coordenadas centrais de cada grupo

# Mostra os centroides na escala original (antes da padronização)
print(scaler_salario.inverse_transform(kmeans_salario.cluster_centers_))

# Rótulos de cada ponto, indicando a qual cluster ele pertence
rotulos = kmeans_salario.labels_

# Cria um gráfico de dispersão para os pontos agrupados
grafico1 = px.scatter(x=base_salario[:, 0], y=base_salario[:, 1], color=rotulos)  # Grupos coloridos
# Cria um gráfico para os centroides dos clusters
grafico2 = px.scatter(x=centroides[:, 0], y=centroides[:, 1], size=[12, 12, 12])  # Centroides com tamanho maior
# Combina os dois gráficos em um só
grafico3 = go.Figure(data=grafico1.data + grafico2.data)
grafico3.show()  # Exibe o gráfico combinado