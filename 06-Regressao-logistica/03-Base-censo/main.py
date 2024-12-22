# Importação das bibliotecas necessárias
import pickle  # Para carregar arquivos serializados
from sklearn.linear_model import LogisticRegression  # Modelo de regressão logística
from sklearn.metrics import accuracy_score, classification_report  # Métricas de avaliação

# Carrega os dados do arquivo 'census.pkl', que contém os dados de treinamento e teste.
with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

# Criação do modelo de regressão logística
# O parâmetro `random_state` garante reprodutibilidade nos resultados.
logistic_census = LogisticRegression(random_state=1)

# Treina o modelo com os dados de entrada (x_census_treinamento) e suas respectivas saídas (y_census_treinamento)
logistic_census.fit(x_census_treinamento, y_census_treinamento)

# Coeficientes do modelo (não utilizados diretamente aqui, mas podem ser úteis para análise)
# print(logistic_census.intercept_)  # Intercepto (bias) do modelo
# print(logistic_census.coef_)       # Coeficientes de cada atributo no modelo

# Faz previsões no conjunto de teste usando o modelo treinado
previsoes = logistic_census.predict(x_census_teste)

# Avalia o desempenho do modelo usando as previsões geradas e os valores reais
# A acurácia é a proporção de previsões corretas
print(accuracy_score(y_census_teste, previsoes))

# O classification_report fornece uma análise mais detalhada com métricas como precisão, recall, f1-score
print(classification_report(y_census_teste, previsoes))