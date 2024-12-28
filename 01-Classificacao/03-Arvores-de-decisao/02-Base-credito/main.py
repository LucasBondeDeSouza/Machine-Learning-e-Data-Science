from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Carregando os dados de treinamento e teste a partir de um arquivo pickle
# O arquivo 'credit.pkl' deve conter as variáveis: 
# x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste
with open('credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Criando um modelo de árvore de decisão com o critério 'entropy' para medir a qualidade da divisão
# 'random_state=0' garante resultados reproduzíveis
avore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)

# Treinando o modelo de árvore de decisão com os dados de treinamento
avore_credit.fit(x_credit_treinamento, y_credit_treinamento)

# Previsões comentadas, mas podem ser ativadas para obter as predições com os dados de teste
previsoes = avore_credit.predict(x_credit_teste)

# Lista com os nomes das variáveis preditoras (características) do conjunto de dados
# Essa lista será usada para rotular os nós na visualização da árvore
previsores = ['income', 'age', 'loan']

# Configurando a visualização gráfica da árvore de decisão
# Cria uma figura com dimensões 8x6 polegadas para o gráfico
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))  # Tamanho do gráfico ajustado

# Gerando o gráfico da árvore de decisão:
tree.plot_tree(
    avore_credit, 
    feature_names=previsores, # - feature_names: Nomes das características preditoras
    class_names=['0', '1'],  # - class_names: Classes alvo convertidas para strings ('0' e '1' representam as classes)
    filled=True, # - filled: Nós preenchidos com cores para representar as classes
    fontsize=8  # Ajustando o tamanho da fonte
)
# Salvando o gráfico gerado em um arquivo de imagem
fig.savefig('arvore_credit.png')
# plt.show()

# Cálculo da acurácia e relatório de classificação
# Esses cálculos estão comentados, mas podem ser ativados para análise do desempenho do modelo
print(accuracy_score(y_credit_teste, previsoes))  # Exibe a acurácia do modelo
print(classification_report(y_credit_teste, previsoes))  # Exibe métricas detalhadas (precisão, recall, F1-score)