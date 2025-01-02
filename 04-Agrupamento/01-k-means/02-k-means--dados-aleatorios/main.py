import plotly.express as px  # Biblioteca para criar gráficos interativos
import plotly.graph_objects as go  # Biblioteca para criar gráficos mais personalizados
from sklearn.cluster import KMeans  # Algoritmo de agrupamento
from sklearn.datasets import make_blobs  # Função para criar dados de exemplo com "blobs"

# Cria 200 pontos de dados aleatórios divididos em 5 grupos
x_random, y_random = make_blobs(n_samples = 200, centers = 5, random_state = 3)

# Cria um gráfico de dispersão (pontos) com os dados gerados
grafico = px.scatter(x = x_random[:, 0], y = x_random[:, 1])
grafico.show()  # Exibe o gráfico

# Aplica o algoritmo KMeans para agrupar os dados em 5 grupos (clusters)
kmeans_blobs = KMeans(n_clusters = 5)
kmeans_blobs.fit(x_random)  # Treina o modelo com os dados

# Previsão dos rótulos dos grupos para cada ponto de dados
rotulos = kmeans_blobs.predict(x_random)

# Obtém as coordenadas dos centros dos grupos
centroides = kmeans_blobs.cluster_centers_

# Cria o gráfico de dispersão com cores diferentes para cada grupo (rótulos)
grafico1 = px.scatter(x = x_random[:, 0], y = x_random[:, 1], color = rotulos)

# Cria um gráfico para mostrar os centros dos grupos (centroides)
grafico2 = px.scatter(x = centroides[:, 0], y = centroides[:, 1], size = [5, 5, 5, 5, 5])

# Combina os dois gráficos (pontos e centroides) e exibe
grafico3 = go.Figure(data = grafico1.data + grafico2.data)
grafico3.show()