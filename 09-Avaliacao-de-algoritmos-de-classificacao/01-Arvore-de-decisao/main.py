from sklearn.model_selection import GridSearchCV  # Importa ferramenta para buscar os melhores parâmetros de um modelo
from sklearn.tree import DecisionTreeClassifier  # Importa modelo de árvore de decisão
import pickle  # Importa biblioteca para carregar e salvar arquivos
import numpy as np  # Importa biblioteca para trabalhar com arrays e operações numéricas

# Abre o arquivo 'credit.pkl' que contém os dados e carrega em variáveis de treino e teste
with open('../credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Combina os dados de treino e teste em um único conjunto (X para entrada e Y para saída)
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Junta os dados de entrada
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Junta os dados de saída

# Define os parâmetros que queremos testar para o modelo de árvore de decisão
parametros = {
    'criterion': ['gini', 'entropy'],  # Critérios para medir qualidade de divisão (gini ou entropia)
    'splitter': ['best', 'random'],  # Estratégia para dividir os nós da árvore
    'min_samples_split': [2, 5, 10],  # Mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 5, 10],  # Mínimo de amostras em uma folha (nó final)
}

# Configura a busca pelos melhores parâmetros usando GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros)

# Treina o modelo com todas as combinações de parâmetros usando os dados
grid_search.fit(x_credit, y_credit)

# Obtém os melhores parâmetros e o melhor resultado de desempenho do modelo
melhores_parametros = grid_search.best_params_  # Melhores opções de parâmetros
melhor_resultado = grid_search.best_score_  # Melhor precisão média obtida durante a validação

# Mostra os melhores parâmetros e o melhor resultado
print(melhores_parametros)
print(melhor_resultado)