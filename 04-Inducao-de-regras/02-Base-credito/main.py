# Importando o módulo Orange, uma biblioteca para aprendizado de máquina,
# focada em tarefas como classificação, avaliação e manipulação de dados.
import Orange
import Orange.classification
import Orange.evaluation

# Carregando a base de dados 'credit_data_regras.csv', que será usada para treinar e testar o modelo.
# Essa base contém informações sobre crédito e suas respectivas classificações (ex.: risco baixo, moderado ou alto).
base_credito = Orange.data.Table('credit_data_regras.csv')

# Dividindo a base de dados em duas partes:
# - 25% dos dados para teste (verificar a precisão do modelo).
# - 75% dos dados para treinamento (ensinar o modelo a classificar corretamente).
base_dividida = Orange.evaluation.testing.sample(base_credito, n=0.25)

# Separando os dados em conjuntos de treinamento e teste.
base_treinamento = base_dividida[1]  # 75% dos dados usados para treinar o modelo.
base_teste = base_dividida[0]        # 25% dos dados usados para testar o modelo.

# Criando um aprendiz de regras baseado no algoritmo CN2.
# O CN2 gera um conjunto de regras compreensíveis para prever a classe de um exemplo.
cn2 = Orange.classification.rules.CN2Learner()

# Treinando o modelo com a base de treinamento.
# O resultado será um conjunto de regras aprendidas com os dados.
regras_credit = cn2(base_treinamento)


# Abaixo temos um código comentado que, se ativado, exibe todas as regras aprendidas pelo modelo.
# Essas regras explicam como as decisões estão sendo tomadas.
# for regras in regras_credit.rule_list:
#     print(regras)


# Avaliando o desempenho do modelo.
# O `TestOnTestData` aplica o modelo treinado (base_treinamento) nos dados de teste (base_teste)
# para verificar a precisão das previsões.
previsoes = Orange.evaluation.testing.TestOnTestData(
    base_treinamento,  # Dados usados para treinar o modelo.
    base_teste,        # Dados usados para testar o modelo.
    [lambda testdata: regras_credit]  # Modelo de previsão a ser avaliado.
)

# Calculando e exibindo a precisão do modelo.
# `CA` (Classification Accuracy) é a porcentagem de exemplos corretamente classificados no teste.
print(Orange.evaluation.CA(previsoes))