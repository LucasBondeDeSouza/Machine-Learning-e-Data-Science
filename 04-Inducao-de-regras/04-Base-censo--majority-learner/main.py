# Importando as bibliotecas necessárias do Orange, que é uma ferramenta para análise de dados e aprendizado de máquina.
import Orange
import Orange.classification
import Orange.evaluation
from collections import Counter  # Usado para contar quantas vezes algo aparece

# Carregando a base de dados chamada 'census_regras.csv', que contém informações de censos (dados demográficos).
base_census = Orange.data.Table('census_regras.csv')

# Criando um modelo que sempre escolhe a classe mais comum, chamado "MajorityLearner".
majority = Orange.classification.MajorityLearner()

# Usando uma função para testar como o modelo se sai com os dados, comparando as previsões com os dados reais.
previsoes = Orange.evaluation.testing.TestOnTestData(base_census, base_census, [majority])

# Mostrando o resultado da avaliação do modelo. O "CA" significa "Accuracy", ou seja, a precisão do modelo.
print(Orange.evaluation.CA(previsoes))

# Esta parte está comentada, mas se ativada, ela mostraria a classe (resultado esperado) de cada registro na base de dados.
#for registro in base_census:
#    print(registro.get_class())

# Contando quantas vezes cada classe (resultado esperado) aparece nos dados de censos.
print(Counter(str(registro.get_class()) for registro in base_census))