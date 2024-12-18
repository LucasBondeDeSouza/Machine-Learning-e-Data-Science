from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree
import matplotlib.pyplot as plt

# Carregando os dados do arquivo pickle
with open('risco_credito.pkl', 'rb') as f:
    x_risco_credito, y_risco_credito = pickle.load(f)


# Criando e treinando um modelo de árvore de decisão para classificar o risco de crédito
# O critério 'entropy' é usado para calcular os ganhos de informação
arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')
arvore_risco_credito.fit(x_risco_credito, y_risco_credito)


# Lista dos nomes das características usadas como previsores no modelo
# Cada índice corresponde a uma característica no conjunto de dados
previsores = ['história', 'dívida', 'garantias', 'renda']


# Configurando a visualização da árvore de decisão
# O gráfico terá 8 polegadas de largura e 6 polegadas de altura
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))  # Tamanho reduzido


# Plotando a árvore de decisão com as seguintes configurações:
# - Nomes das características e classes fornecidos
# - Nós preenchidos com cores para representar as classes
# - Tamanho da fonte ajustado para 8 para facilitar a visualização em gráficos menores
tree.plot_tree(
    arvore_risco_credito, 
    feature_names=previsores, 
    class_names=arvore_risco_credito.classes_, 
    filled=True,
    ax=eixos,
    fontsize=8  # Reduzindo o tamanho da fonte
)

# Exibindo a árvore
# plt.show()


# Simulando
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
previsoes = arvore_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(previsoes)