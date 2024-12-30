import pandas as pd  # Biblioteca para manipulação de dados em tabelas (DataFrames)
from sklearn.model_selection import train_test_split  # Divide os dados em treino e teste
from sklearn.metrics import mean_absolute_error  # Calcula o erro absoluto médio
from sklearn.tree import DecisionTreeRegressor  # Modelo de Árvore de Decisão para regressão

# Carrega os dados do arquivo CSV com preços de casas
base_casas = pd.read_csv('house_prices.csv')

# Define as variáveis de entrada (x) e saída (y)
x_casas = base_casas.iloc[:, 3:19].values  # Dados como tamanho, localização, etc.
y_casas = base_casas.iloc[:, 2].values  # Preços das casas (o que queremos prever)

# Divide os dados em treinamento (70%) e teste (30%)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0  # Separação dos dados com 30% para teste
)

# Cria o modelo de regressão com árvore de decisão
regressor_arvore_casas = DecisionTreeRegressor()

# Treina o modelo com os dados de treinamento
regressor_arvore_casas.fit(x_casas_treinamento, y_casas_treinamento)

# Avalia o modelo no conjunto de treinamento
print(regressor_arvore_casas.score(x_casas_treinamento, y_casas_treinamento))  # Acurácia no treino
# Avalia o modelo no conjunto de teste
print(regressor_arvore_casas.score(x_casas_teste, y_casas_teste))  # Acurácia no teste

# Faz previsões usando o modelo treinado
previsoes = regressor_arvore_casas.predict(x_casas_teste)

# Calcula o erro médio das previsões (quanto o modelo errou em média)
print(mean_absolute_error(y_casas_teste, previsoes))  # Mostra o erro médio das previsões