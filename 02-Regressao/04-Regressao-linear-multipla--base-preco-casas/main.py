import pandas as pd  # Biblioteca para manipulação de tabelas (DataFrames)
from sklearn.model_selection import train_test_split  # Divide os dados em treino e teste
from sklearn.linear_model import LinearRegression  # Modelo de Regressão Linear
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Métricas de erro

# Carrega os dados de preços de casas a partir de um arquivo CSV
base_casas = pd.read_csv('house_prices.csv')

# Separa as colunas de características (variáveis independentes) e a coluna de preços (variável dependente)
x_casas = base_casas.iloc[:, 3:19].values  # Características (dados que vão ajudar a prever o preço)
y_casas = base_casas.iloc[:, 2].values  # Preço das casas (o que queremos prever)

# Divide os dados em 70% para treinamento e 30% para teste
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0
)

# Cria o modelo de regressão linear múltipla (para trabalhar com várias características)
regressor_multiplo_casas = LinearRegression()
# Treina o modelo com os dados de treinamento
regressor_multiplo_casas.fit(x_casas_treinamento, y_casas_treinamento)

# Exibe o valor do intercepto (b0) da reta de regressão
print(regressor_multiplo_casas.intercept_)
# Exibe os coeficientes (b1, b2, ...) para cada característica usada para prever o preço
print(regressor_multiplo_casas.coef_)

# Exibe a precisão do modelo nos dados de treinamento e teste (quanto o modelo se ajusta aos dados)
print(regressor_multiplo_casas.score(x_casas_treinamento, y_casas_treinamento))
print(regressor_multiplo_casas.score(x_casas_teste, y_casas_teste))

# Faz previsões sobre os dados de teste (onde o modelo tenta adivinhar os preços das casas)
previsoes = regressor_multiplo_casas.predict(x_casas_teste)

# Exibe o erro médio absoluto entre o valor real e o valor previsto (quanto, em média, o modelo errou)
print(mean_absolute_error(y_casas_teste, previsoes))