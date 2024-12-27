import pickle  # Biblioteca usada para salvar e carregar arquivos
import numpy as np  # Biblioteca usada para manipulação de arrays (listas de números)
from sklearn.neural_network import MLPClassifier  # Importa a Rede Neural Multi-layer Perceptron (MLP)
from sklearn.tree import DecisionTreeClassifier  # Importa o modelo de Árvore de Decisão
from sklearn.svm import SVC  # Importa o modelo de Máquinas de Vetores de Suporte (SVM)

# Carrega os dados que foram pré-processados e salvos em um arquivo 'credit.pkl'
with open('../credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Junta os dados de treinamento e teste em um único conjunto de dados para análise.
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)  # Junta as características (dados de entrada)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)  # Junta as respostas (o que esperamos que o modelo aprenda)

# Cria e treina o modelo de Rede Neural (MLP)
classificador_rede_neural = MLPClassifier(activation='relu', batch_size=56, solver='adam')  # Define a Rede Neural
classificador_rede_neural.fit(x_credit, y_credit)  # Treina a Rede Neural com os dados

# Cria e treina o modelo de Árvore de Decisão
classificador_arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')  # Define a Árvore de Decisão
classificador_arvore.fit(x_credit, y_credit)  # Treina a Árvore de Decisão com os dados

# Cria e treina o modelo de Máquinas de Vetores de Suporte (SVM)
classificador_svm = SVC(C=2.0, kernel='rbf')  # Define o modelo SVM com um kernel 'rbf' e parâmetro C
classificador_svm.fit(x_credit, y_credit)  # Treina o modelo SVM com os dados

# Salva os modelos treinados em arquivos para que possam ser carregados e utilizados novamente
pickle.dump(classificador_rede_neural, open('rede_neural_finalizado.sav', 'wb'))  # Salva o modelo de Rede Neural
pickle.dump(classificador_arvore, open('arvore_finalizado.sav', 'wb'))  # Salva o modelo de Árvore de Decisão
pickle.dump(classificador_svm, open('svm_finalizado.sav', 'wb'))  # Salva o modelo de SVM