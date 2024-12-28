from sklearn.model_selection import GridSearchCV  # Ferramenta para testar várias combinações de parâmetros
from sklearn.neural_network import MLPClassifier  # Modelo de Classificador Multi-layer Perceptron (Rede Neural)
import pickle  # Biblioteca para carregar e salvar arquivos
import numpy as np  # Biblioteca para trabalhar com arrays e cálculos numéricos

# Abre o arquivo 'credit.pkl' que contém os dados e carrega as variáveis de treino e teste
with open('../credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Junta os dados de treino e teste em um único conjunto para entrada (X) e saída (Y)
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Junta as entradas (features)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Junta os alvos (labels)

# Define os parâmetros que serão testados para o modelo de rede neural MLP
parametros = {
    'activation': ['relu', 'logistic', 'tanh'],  # Funções de ativação para as camadas da rede neural
    'solver': ['adam', 'sgd'],  # Algoritmos para otimizar a rede (Adam e Stochastic Gradient Descent)
    'batch_size': [10, 56]  # Tamanho dos lotes de dados usados para treinar a rede
}

# Configura o GridSearchCV para testar várias combinações de parâmetros no modelo MLP
grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=parametros)

# Treina o modelo com as combinações de parâmetros e os dados completos
grid_search.fit(x_credit, y_credit)

# Obtém os melhores parâmetros e a melhor pontuação de desempenho do modelo
melhores_parametros = grid_search.best_params_  # Parâmetros que obtiveram o melhor desempenho
melhor_resultado = grid_search.best_score_  # Melhor resultado obtido na validação cruzada

# Exibe os resultados: os melhores parâmetros e o desempenho correspondente
print(melhores_parametros)
print(melhor_resultado)