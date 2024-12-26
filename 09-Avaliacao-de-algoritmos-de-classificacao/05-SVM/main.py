from sklearn.model_selection import GridSearchCV  # Ferramenta para testar várias combinações de parâmetros
from sklearn.svm import SVC  # Modelo de Classificador Support Vector Machine (SVM)
import pickle  # Biblioteca para carregar e salvar arquivos
import numpy as np  # Biblioteca para trabalhar com arrays e cálculos numéricos

# Abre o arquivo 'credit.pkl' que contém os dados e carrega as variáveis de treino e teste
with open('../credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Junta os dados de treino e teste em um único conjunto para entrada (X) e saída (Y)
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Junta as entradas (features)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Junta os alvos (labels)

# Define os parâmetros que serão testados para o modelo SVM
parametros = {
    'tol': [0.001, 0.0001, 0.00001],  # Tolerância para a precisão (quanto menor, mais preciso)
    'C': [1.0, 1.5, 2.0],  # Parâmetro de regularização (quanto maior, mais flexível o modelo)
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']  # Tipos de kernel (função que transforma dados para o modelo SVM)
}

# Configura o GridSearchCV para testar várias combinações de parâmetros no modelo SVM
grid_search = GridSearchCV(estimator=SVC(), param_grid=parametros)

# Treina o modelo com as combinações de parâmetros e os dados completos
grid_search.fit(x_credit, y_credit)

# Obtém os melhores parâmetros e a melhor pontuação de desempenho do modelo
melhores_parametros = grid_search.best_params_  # Parâmetros que obtiveram o melhor desempenho
melhor_resultado = grid_search.best_score_  # Melhor resultado obtido na validação cruzada

# Exibe os resultados: os melhores parâmetros e o desempenho correspondente
print(melhores_parametros)
print(melhor_resultado)