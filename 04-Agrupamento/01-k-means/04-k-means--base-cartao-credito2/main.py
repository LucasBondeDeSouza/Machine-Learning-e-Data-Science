import plotly.express as px  # Biblioteca para criar gráficos interativos
from sklearn.cluster import KMeans  # Algoritmo de agrupamento
from sklearn.preprocessing import StandardScaler  # Ferramenta para padronizar dados
import pandas as pd  # Biblioteca para manipular dados em tabelas
from sklearn.decomposition import PCA  # Ferramenta para reduzir a dimensionalidade dos dados

# Carrega os dados de clientes de cartão de crédito de um arquivo CSV
# 'header=1' indica que a primeira linha do arquivo é ignorada
base_cartao = pd.read_csv('credit_card_clients.csv', header=1)

# Cria uma nova coluna que soma as faturas dos últimos 6 meses para cada cliente
base_cartao['BILL_TOTAL'] = (
    base_cartao['BILL_AMT1'] + base_cartao['BILL_AMT2'] + 
    base_cartao['BILL_AMT3'] + base_cartao['BILL_AMT4'] + 
    base_cartao['BILL_AMT5'] + base_cartao['BILL_AMT6']
)

# Seleciona colunas específicas para análise (idade, limite, faturas, total de faturas)
x_cartao_mais = base_cartao.iloc[:, [1, 2, 3, 4, 5, 25]].values

# Padroniza os dados para que todas as variáveis fiquem na mesma escala
scaler_cartao = StandardScaler()
x_cartao_mais = scaler_cartao.fit_transform(x_cartao_mais)

# Lista para armazenar o WCSS (soma total das distâncias dentro dos clusters)
wcss = []
for i in range(1, 11):  # Testa diferentes números de clusters (de 1 a 10)
    kmeans_cartao_mais = KMeans(n_clusters=i, random_state=0)  # Configura o modelo K-Means
    kmeans_cartao_mais.fit(x_cartao_mais)  # Ajusta o modelo aos dados
    wcss.append(kmeans_cartao_mais.inertia_)  # Armazena o WCSS para cada quantidade de clusters

# Plota o gráfico do "cotovelo" para identificar o número ideal de clusters
grafico = px.line(x=range(1, 11), y=wcss)
grafico.show()

# Define 2 clusters como o número ideal (com base no gráfico do cotovelo)
kmeans_cartao_mais = KMeans(n_clusters=2, random_state=0)
rotulos = kmeans_cartao_mais.fit_predict(x_cartao_mais)  # Agrupa os clientes e retorna os rótulos dos clusters

# Reduz a dimensionalidade dos dados para 2 componentes principais para visualização
pca = PCA(n_components=2)
x_cartao_mais_pca = pca.fit_transform(x_cartao_mais)

# Plota os dados reduzidos com os clusters coloridos
grafico = px.scatter(x=x_cartao_mais_pca[:, 0], y=x_cartao_mais_pca[:, 1], color=rotulos)
grafico.show()