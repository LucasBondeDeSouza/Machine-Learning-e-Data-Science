# Importação das bibliotecas necessárias
import pickle  # Para carregar arquivos serializados
import numpy as np  # Para manipulação de arrays e matrizes
from sklearn.linear_model import LogisticRegression  # Modelo de regressão logística

# Carrega os dados do arquivo 'risco_credito.pkl', que contém as variáveis de entrada (X) e as saídas (y)
with open('risco_credito.pkl', 'rb') as f:
    x_risco_credito, y_risco_credito = pickle.load(f)

# Remove algumas linhas específicas (índices 2, 7 e 11) dos dados de entrada e saída.
# Isso pode ser útil para remover outliers ou entradas problemáticas.
x_risco_credito = np.delete(x_risco_credito, [2, 7, 11], axis=0)
y_risco_credito = np.delete(y_risco_credito, [2, 7, 11], axis=0)

# Criação do modelo de regressão logística
# O parâmetro `random_state` garante reprodutibilidade nos resultados.
logistic_risco_credito = LogisticRegression(random_state=1)

# Treina o modelo com os dados de entrada (x_risco_credito) e suas respectivas saídas (y_risco_credito)
logistic_risco_credito.fit(x_risco_credito, y_risco_credito)

# Coeficientes do modelo (não utilizados diretamente aqui, mas podem ser úteis para análise)
# print(logistic_risco_credito.intercept_)  # Intercepto (bias) do modelo
# print(logistic_risco_credito.coef_)       # Coeficientes de cada atributo no modelo

# Previsões com o modelo treinado
# Testamos o modelo com duas novas instâncias de dados:
# [0, 0, 1, 2]: história boa, dívida alta, garantias nenhuma, renda > 35
# [2, 0, 0, 0]: história ruim, dívida alta, garantias adequada, renda < 15
previsoes = logistic_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])

print(previsoes)