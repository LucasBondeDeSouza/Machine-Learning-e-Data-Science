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
from scipy.stats import f_oneway  # Teste ANOVA para comparar grupos
from statsmodels.stats.multicomp import MultiComparison  # Comparação múltipla para teste estatístico

# Carrega os dados pré-processados (treinamento e teste) salvos em um arquivo.
with open('../credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Junta os dados de treinamento e teste em um único conjunto para análise.
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Dados de entrada
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Saídas correspondentes

# Cria listas para armazenar os resultados de diferentes modelos de aprendizado de máquina.
resultados_arvore = []
resultados_random_forest = []
resultados_knn = []
resultados_logistica = []
resultados_svm = []
resultados_rede_neural = []

# Executa 30 rodadas de validação cruzada para avaliar a precisão dos modelos.
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)  # Divide os dados em 10 partes (folds).

    # Avaliação da Árvore de Decisão
    arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
    scores = cross_val_score(arvore, x_credit, y_credit, cv=kfold)  # Calcula a precisão em cada rodada.
    resultados_arvore.append(scores.mean())  # Salva a média da precisão.

    # Avaliação do Random Forest
    random_forest = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=10)
    scores = cross_val_score(random_forest, x_credit, y_credit, cv=kfold)
    resultados_random_forest.append(scores.mean())

    # Avaliação do K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier()
    scores = cross_val_score(knn, x_credit, y_credit, cv=kfold)
    resultados_knn.append(scores.mean())

    # Avaliação da Regressão Logística
    logistica = LogisticRegression(C=1.0, solver='lbfgs', tol=0.0001)
    scores = cross_val_score(logistica, x_credit, y_credit, cv=kfold)
    resultados_logistica.append(scores.mean())

    # Avaliação da Máquina de Vetores de Suporte (SVM)
    svm = SVC(kernel='rbf', C=2.0)
    scores = cross_val_score(svm, x_credit, y_credit, cv=kfold)
    resultados_svm.append(scores.mean())

    # Avaliação da Rede Neural (MLP)
    rede_neural = MLPClassifier(activation='relu', batch_size=56, solver='adam')
    scores = cross_val_score(rede_neural, x_credit, y_credit, cv=kfold)
    resultados_rede_neural.append(scores.mean())

# Organiza os resultados de todas as rodadas em uma tabela para análise.
resultados = pd.DataFrame({
    'Arvore': resultados_arvore,
    'Random Forest': resultados_random_forest,
    'KNN': resultados_knn,
    'Logistica': resultados_logistica,
    'SVM': resultados_svm,
    'Rede Neural': resultados_rede_neural
})

# Teste ANOVA para verificar se os modelos possuem diferenças significativas nos resultados.
p = f_oneway(
    resultados_arvore,
    resultados_random_forest,
    resultados_knn,
    resultados_logistica,
    resultados_svm,
    resultados_rede_neural
)

# Define um nível de significância (alpha) e avalia a hipótese nula.
alpha = 0.05
if p <= alpha:
    print('Hipótese nula rejeitada. Dados são diferentes')  # Há diferença entre os modelos.
else:
    print('Hipótese alternativa rejeitada. Resultados são iguais')  # Sem diferenças significativas.

# Prepara os resultados para comparações detalhadas entre os algoritmos.
resultados_algoritmos = {
    'accuracy': np.concatenate([
        resultados_arvore,
        resultados_random_forest,
        resultados_knn,
        resultados_logistica,
        resultados_svm,
        resultados_rede_neural
    ]),
    'algoritmo': np.repeat(
        ['arvore', 'random_forest', 'knn', 'logistica', 'svm', 'rede_neural'],
        30
    )
}

resultados_df = pd.DataFrame(resultados_algoritmos)
print(resultados_df)  # Exibe a tabela com as precisões e os nomes dos algoritmos.

# Realiza o teste Tukey HSD para comparações estatísticas detalhadas.
compara_algoritmos = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo'])
teste_estatistico = compara_algoritmos.tukeyhsd()
print(teste_estatistico)  # Exibe os resultados do teste estatístico.

# Exibe a média dos resultados de cada modelo.
print(resultados.mean())