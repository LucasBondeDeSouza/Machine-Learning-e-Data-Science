from sklearn.model_selection import GridSearchCV  # Ferramenta para encontrar os melhores parâmetros de um modelo
from sklearn.ensemble import RandomForestClassifier  # Modelo de Random Forest
import pickle  # Biblioteca para carregar e salvar arquivos
import numpy as np  # Biblioteca para trabalhar com arrays e cálculos numéricos

# Abre o arquivo 'credit.pkl' que contém os dados e os carrega em variáveis de treino e teste
with open('../credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Junta os dados de treino e teste em conjuntos únicos para entrada (X) e saída (Y)
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Combina dados de entrada
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Combina dados de saída

# Define os parâmetros que serão testados para o modelo Random Forest
parametros = {
    'criterion': ['gini', 'entropy'],  # Critério para avaliar a qualidade das divisões (Gini ou Entropia)
    'n_estimators': [10, 40, 100, 150],  # Quantidade de árvores na floresta
    'min_samples_split': [2, 5, 10],  # Mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 5, 10],  # Mínimo de amostras em um nó folha
}

# Configura o GridSearchCV para testar várias combinações de parâmetros
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)

# Treina o modelo com as combinações de parâmetros e os dados completos
grid_search.fit(x_credit, y_credit)

# Obtém os melhores parâmetros e o melhor desempenho do modelo
melhores_parametros = grid_search.best_params_  # Os parâmetros ideais encontrados
melhor_resultado = grid_search.best_score_  # A melhor pontuação média alcançada

# Exibe os resultados: os melhores parâmetros e o desempenho correspondente
print(melhores_parametros)
print(melhor_resultado)