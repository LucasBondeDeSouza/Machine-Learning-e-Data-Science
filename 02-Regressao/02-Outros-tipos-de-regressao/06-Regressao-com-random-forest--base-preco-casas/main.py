import pandas as pd  # Biblioteca para manipulação de dados em tabelas (DataFrames)
from sklearn.model_selection import train_test_split  # Função para dividir os dados em partes de treino e teste
from sklearn.metrics import mean_absolute_error  # Função para calcular o erro das previsões feitas
from sklearn.ensemble import RandomForestRegressor  # Modelo de Regressão de Floresta Aleatória (usado para prever)

# Carrega os dados do arquivo CSV com preços de casas (o arquivo deve estar no mesmo diretório)
base_casas = pd.read_csv('house_prices.csv')

# Define as variáveis de entrada (x) e saída (y) para o modelo
x_casas = base_casas.iloc[:, 3:19].values  # Dados como tamanho da casa, localização, etc. (informações que ajudam a prever o preço)
y_casas = base_casas.iloc[:, 2].values  # Preços das casas (o que queremos prever)

# Divide os dados em duas partes: 70% para treinamento do modelo e 30% para testar a precisão
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0  # O 30% é destinado ao teste do modelo
)

# Cria o modelo de Floresta Aleatória, que vai usar 100 árvores para prever os preços
regressor_random_forest_casas = RandomForestRegressor(n_estimators=100)
# Treina o modelo usando os dados de treinamento
regressor_random_forest_casas.fit(x_casas_treinamento, y_casas_treinamento)

# Avalia o modelo e imprime a precisão sobre os dados de treinamento
print(regressor_random_forest_casas.score(x_casas_treinamento, y_casas_treinamento))
# Avalia o modelo e imprime a precisão sobre os dados de teste (que o modelo nunca viu antes)
print(regressor_random_forest_casas.score(x_casas_teste, y_casas_teste))

# Faz previsões de preços para as casas no conjunto de teste
previsoes = regressor_random_forest_casas.predict(x_casas_teste)

# Calcula e imprime o erro médio absoluto das previsões feitas (quanto o modelo errou, em média)
print(mean_absolute_error(y_casas_teste, previsoes))