from sklearn.tree import DecisionTreeClassifier  # Modelo de Árvore de Decisão
from sklearn.ensemble import RandomForestClassifier  # Modelo Random Forest
from sklearn.neighbors import KNeighborsClassifier  # Modelo K-Nearest Neighbors
from sklearn.linear_model import LogisticRegression  # Modelo de Regressão Logística
from sklearn.svm import SVC  # Modelo de Máquina de Vetores de Suporte
from sklearn.neural_network import MLPClassifier  # Modelo de Rede Neural Multi-layer Perceptron
from sklearn.model_selection import cross_val_score, KFold  # Validação cruzada e separação de dados
import pickle  # Para salvar e carregar arquivos
import numpy as np  # Biblioteca para manipulação de arrays
import pandas as pd  # Manipulação de tabelas de dados
from scipy.stats import shapiro  # Teste de normalidade
import seaborn as sns  # Visualização de dados

# Carrega os dados salvos no arquivo 'credit.pkl'
with open('../credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Junta os dados de treino e teste para entrada (x) e saída (y)
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Concatena as features
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Concatena os labels

# Listas para guardar os resultados de cada modelo
resultados_arvore = []  # Resultados da Árvore de Decisão
resultados_random_forest = []  # Resultados do Random Forest
resultados_knn = []  # Resultados do KNN
resultados_logistica = []  # Resultados da Regressão Logística
resultados_svm = []  # Resultados do SVM
resultados_rede_neural = []  # Resultados da Rede Neural

# Faz 30 rodadas de validação cruzada para cada modelo
for i in range(30):
    # Divide os dados em 10 partes (folds) aleatórias para validação cruzada
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    # Árvore de Decisão
    arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
    scores = cross_val_score(arvore, x_credit, y_credit, cv=kfold)  # Mede o desempenho com validação cruzada
    resultados_arvore.append(scores.mean())  # Salva a média dos scores

    # Random Forest
    random_forest = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=10)
    scores = cross_val_score(random_forest, x_credit, y_credit, cv=kfold)
    resultados_random_forest.append(scores.mean())

    # K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier()  # Configura o modelo com parâmetros padrão
    scores = cross_val_score(knn, x_credit, y_credit, cv=kfold)
    resultados_knn.append(scores.mean())

    # Regressão Logística
    logistica = LogisticRegression(C=1.0, solver='lbfgs', tol=0.0001)  # Configura os parâmetros
    scores = cross_val_score(logistica, x_credit, y_credit, cv=kfold)
    resultados_logistica.append(scores.mean())

    # SVM (Máquinas de Vetores de Suporte)
    svm = SVC(kernel='rbf', C=2.0)  # Define o kernel RBF e a regularização
    scores = cross_val_score(svm, x_credit, y_credit, cv=kfold)
    resultados_svm.append(scores.mean())

    # Rede Neural (MLP)
    rede_neural = MLPClassifier(activation='relu', batch_size=56, solver='adam')  # Configura a rede neural
    scores = cross_val_score(rede_neural, x_credit, y_credit, cv=kfold)
    resultados_rede_neural.append(scores.mean())

# Organiza os resultados em uma tabela (DataFrame)
resultados = pd.DataFrame({
    'Arvore': resultados_arvore,
    'Random Forest': resultados_random_forest,
    'KNN': resultados_knn,
    'Logistica': resultados_logistica,
    'SVM': resultados_svm,
    'Rede Neural': resultados_rede_neural
})

# Teste de normalidade para os resultados de cada modelo
print(
    shapiro(resultados_arvore), 
    shapiro(resultados_random_forest), 
    shapiro(resultados_knn), 
    shapiro(resultados_logistica), 
    shapiro(resultados_svm), 
    shapiro(resultados_rede_neural)
)

# Visualiza a distribuição dos resultados de cada modelo
sns.displot(resultados_arvore, kind='kde')  # Distribuição dos resultados da Árvore de Decisão
sns.displot(resultados_random_forest, kind='kde')  # Random Forest
sns.displot(resultados_knn, kind='kde')  # KNN
sns.displot(resultados_logistica, kind='kde')  # Regressão Logística
sns.displot(resultados_svm, kind='kde')  # SVM
sns.displot(resultados_rede_neural, kind='kde')  # Rede Neural