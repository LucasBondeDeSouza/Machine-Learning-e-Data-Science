import plotly.express as px  # Biblioteca para criar gráficos interativos
from sklearn.cluster import KMeans  # Algoritmo de agrupamento
from sklearn.preprocessing import StandardScaler  # Ferramenta para padronizar dados
import pandas as pd  # Biblioteca para manipular dados em tabelas
import numpy as np  # Biblioteca para trabalhar com arrays e operações matemáticas

# Carrega os dados do arquivo CSV que contém informações de clientes de cartão de crédito
# 'header=1' indica que o cabeçalho começa na segunda linha
base_cartao = pd.read_csv('credit_card_clients.csv', header=1)

# Cria uma nova coluna que soma os valores de todas as faturas dos últimos 6 meses
base_cartao['BILL_TOTAL'] = (
    base_cartao['BILL_AMT1'] + base_cartao['BILL_AMT2'] + 
    base_cartao['BILL_AMT3'] + base_cartao['BILL_AMT4'] + 
    base_cartao['BILL_AMT5'] + base_cartao['BILL_AMT6']
)

# Seleciona as colunas que serão usadas para o agrupamento (idade e total de faturas)
x_cartao = base_cartao.iloc[:, [1, 25]].values

# Padroniza os dados para que todos fiquem na mesma escala
scaler_cartao = StandardScaler()
x_cartao = scaler_cartao.fit_transform(x_cartao)

# Calcula o WCSS (soma dos quadrados dentro dos clusters) para diferentes números de clusters
wcss = []  # Lista para armazenar os valores de WCSS
for i in range(1, 11):  # Testa de 1 a 10 clusters
    kmeans_cartao = KMeans(n_clusters=i, random_state=0)
    kmeans_cartao.fit(x_cartao)
    wcss.append(kmeans_cartao.inertia_)  # Adiciona o WCSS para cada número de clusters

# Cria um gráfico para ajudar a identificar o "cotovelo" (melhor número de clusters)
grafico = px.line(x=range(1, 11), y=wcss, title="Método do Cotovelo", labels={'x': 'Número de Clusters', 'y': 'WCSS'})
grafico.show()

# Escolhe 4 clusters com base no gráfico do cotovelo e realiza o agrupamento
kmeans_cartao = KMeans(n_clusters=4, random_state=0)
rotulos = kmeans_cartao.fit_predict(x_cartao)  # Calcula os rótulos (a qual cluster cada cliente pertence)

# Cria um gráfico de dispersão para visualizar os clientes agrupados
grafico = px.scatter(x=x_cartao[:, 0], y=x_cartao[:, 1], color=rotulos, title="Agrupamento de Clientes", labels={'x': 'Idade (padronizada)', 'y': 'Total de Faturas (padronizado)'})
grafico.show()

# Adiciona os rótulos (clusters) como uma nova coluna na tabela original
lista_clientes = np.column_stack((base_cartao, rotulos))
print(lista_clientes)  # Exibe a tabela com os clientes e seus respectivos clusters

# Ordena os clientes com base nos clusters (última coluna)
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]
print(lista_clientes)  # Exibe a tabela ordenada por clusters