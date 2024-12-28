from sklearn.tree import DecisionTreeClassifier  # Modelo de Árvore de Decisão
from sklearn.ensemble import RandomForestClassifier  # Modelo de Random Forest
from sklearn.neighbors import KNeighborsClassifier  # Modelo K-Nearest Neighbors
from sklearn.linear_model import LogisticRegression  # Modelo de Regressão Logística
from sklearn.svm import SVC  # Modelo de Máquinas de Vetores de Suporte
from sklearn.neural_network import MLPClassifier  # Modelo de Rede Neural Multi-layer Perceptron
from sklearn.model_selection import cross_val_score, KFold  # Ferramentas para validação cruzada
import pickle  # Biblioteca para carregar e salvar arquivos
import numpy as np  # Biblioteca para manipulação de arrays e cálculos numéricos
import pandas as pd

# Abre o arquivo 'credit.pkl' que contém os dados e os carrega em variáveis de treino e teste
with open('../credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Junta os dados de treino e teste em um único conjunto (X para entrada e Y para saída)
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Dados de entrada (features)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Dados de saída (labels)

# Inicializa listas para armazenar os resultados de cada modelo
resultados_arvore = []  # Resultados para o modelo de Árvore de Decisão
resultados_random_forest = []  # Resultados para o modelo Random Forest
resultados_knn = []  # Resultados para o modelo KNN
resultados_logistica = []  # Resultados para o modelo de Regressão Logística
resultados_svm = []  # Resultados para o modelo SVM
resultados_rede_neural = []  # Resultados para o modelo de Rede Neural

# Executa 30 iterações para cada modelo com validação cruzada
for i in range(30):
    # Configura a validação cruzada K-Fold com 10 divisões (folds), embaralhando os dados
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    # Modelo: Árvore de Decisão
    arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
    scores = cross_val_score(arvore, x_credit, y_credit, cv=kfold)  # Valida o modelo
    resultados_arvore.append(scores.mean())  # Salva a média das pontuações

    # Modelo: Random Forest
    random_forest = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=10)
    scores = cross_val_score(random_forest, x_credit, y_credit, cv=kfold)  # Valida o modelo
    resultados_random_forest.append(scores.mean())  # Salva a média das pontuações

    # Modelo: K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier()  # Instancia o modelo com os parâmetros padrão
    scores = cross_val_score(knn, x_credit, y_credit, cv=kfold)  # Valida o modelo
    resultados_knn.append(scores.mean())  # Salva a média das pontuações

    # Modelo: Regressão Logística
    logistica = LogisticRegression(C=1.0, solver='lbfgs', tol=0.0001)
    scores = cross_val_score(logistica, x_credit, y_credit, cv=kfold)  # Valida o modelo
    resultados_logistica.append(scores.mean())  # Salva a média das pontuações

    # Modelo: Máquinas de Vetores de Suporte (SVM)
    svm = SVC(kernel='rbf', C=2.0)  # Configura o kernel RBF e o parâmetro de regularização
    scores = cross_val_score(svm, x_credit, y_credit, cv=kfold)  # Valida o modelo
    resultados_svm.append(scores.mean())  # Salva a média das pontuações

    # Modelo: Rede Neural Multi-layer Perceptron
    rede_neural = MLPClassifier(activation='relu', batch_size=56, solver='adam')  # Configura a rede neural
    scores = cross_val_score(rede_neural, x_credit, y_credit, cv=kfold)  # Valida o modelo
    resultados_rede_neural.append(scores.mean())  # Salva a média das pontuações


# Cria um DataFrame para organizar os resultados de todas as iterações
resultados = pd.DataFrame({
    'Arvore': resultados_arvore, 
    'Random Forest': resultados_random_forest,
    'KNN': resultados_knn,
    'Logistica': resultados_logistica,
    'SVM': resultados_svm,
    'Rede Neural': resultados_rede_neural
})

# Exibe os resultados detalhados de cada modelo por iteração
print(resultados)

# Exibe estatísticas descritivas (média, desvio padrão, etc.) dos resultados
print(resultados.describe())

# Calcula e exibe a variância dos resultados por modelo
print(resultados.var())

# Calcula e exibe o desvio padrão dos resultados por modelo
print(resultados.std())