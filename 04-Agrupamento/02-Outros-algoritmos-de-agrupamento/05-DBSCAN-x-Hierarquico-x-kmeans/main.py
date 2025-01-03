import plotly.express as px  # Biblioteca para criar gráficos interativos
from sklearn.cluster import DBSCAN  # Algoritmo de agrupamento baseado em densidade
from sklearn import datasets  # Biblioteca para gerar conjuntos de dados artificiais
from sklearn.cluster import KMeans  # Algoritmo de agrupamento baseado em centros
from sklearn.cluster import AgglomerativeClustering  # Algoritmo de agrupamento hierárquico

# Gera um conjunto de dados em formato de lua crescente com 1500 pontos e um pouco de ruído
x_random, y_random = datasets.make_moons(n_samples=1500, noise=0.09)

# Plota os dados sem agrupamento
grafico = px.scatter(x=x_random[:, 0], y=x_random[:, 1])
grafico.show()


# K-Means (Divide os dados em 2 grupos com base na média dos pontos)
kmeans = KMeans(n_clusters=2)
rotulos = kmeans.fit_predict(x_random)  # Ajusta e atribui rótulos aos grupos
grafico = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos)
grafico.show()


# Agrupamento Hierárquico (Agrupa os pontos com base na proximidade entre eles)
hc = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
rotulos = hc.fit_predict(x_random)  # Ajusta e atribui rótulos aos grupos
grafico = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos)
grafico.show()


# DBSCAN (Agrupa os pontos com base na densidade, detectando outliers)
dbscan = DBSCAN(eps=0.1)  # Define a distância máxima para considerar pontos vizinhos
rotulos = dbscan.fit_predict(x_random)  # Ajusta e atribui rótulos aos grupos
grafico = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos)
grafico.show()