# Importação das bibliotecas necessárias
import pickle  # Para carregar arquivos serializados
from sklearn.linear_model import LogisticRegression  # Modelo de regressão logística
from sklearn.metrics import accuracy_score, classification_report  # Métricas de avaliação

# Carrega os dados do arquivo 'credit.pkl', que contém os dados de treinamento e teste.
with open('credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Criação do modelo de regressão logística
# O parâmetro `random_state` garante reprodutibilidade nos resultados.
logistic_credit = LogisticRegression(random_state=1)

# Treina o modelo com os dados de entrada (x_credit_treinamento) e suas respectivas saídas (y_credit_treinamento)
logistic_credit.fit(x_credit_treinamento, y_credit_treinamento)

# Coeficientes do modelo (não utilizados diretamente aqui, mas podem ser úteis para análise)
# print(logistic_credit.intercept_)  # Intercepto (bias) do modelo
# print(logistic_credit.coef_)       # Coeficientes de cada atributo no modelo

# Faz previsões no conjunto de teste usando o modelo treinado
previsoes = logistic_credit.predict(x_credit_teste)

# Avalia o desempenho do modelo usando as previsões geradas e os valores reais
# A acurácia é a proporção de previsões corretas
print(accuracy_score(y_credit_teste, previsoes))

# O classification_report fornece uma análise mais detalhada com métricas como precisão, recall, f1-score
print(classification_report(y_credit_teste, previsoes))