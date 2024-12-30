import plotly.express as px  # Biblioteca para criar gráficos interativos
import pandas as pd  # Biblioteca para manipular dados em tabelas (DataFrames)
from sklearn.tree import DecisionTreeRegressor  # Modelo de Árvore de Decisão para regressão
import numpy as np  # Biblioteca para operações matemáticas e arrays

# Carrega os dados do arquivo CSV 'plano_saude.csv' para análise
base_plano_saude = pd.read_csv('plano_saude.csv')

# Separa os dados de entrada (idade) e saída (custo do plano de saúde)
x_plano_saude = base_plano_saude.iloc[:, 0:1].values  # Idade (variável independente)
y_plano_saude = base_plano_saude.iloc[:, 1].values  # Custo do plano (variável dependente)

# Cria o modelo de árvore de decisão e treina com os dados
regressor_arvore_saude = DecisionTreeRegressor()
regressor_arvore_saude.fit(x_plano_saude, y_plano_saude)

# Faz previsões com o modelo para os dados de entrada
previsoes = regressor_arvore_saude.predict(x_plano_saude)

# Cria um gráfico para visualizar os dados reais e as previsões
grafico = px.scatter(x=x_plano_saude.ravel(), y=y_plano_saude)  # Pontos reais (idade vs custo)
grafico.add_scatter(x=x_plano_saude.ravel(), y=previsoes, name='Regressão')  # Linha de previsão
grafico.show()

# Gera novos valores para uma visualização mais detalhada da previsão
x_teste_arvore = np.arange(min(x_plano_saude), max(x_plano_saude), 0.1)  # Valores detalhados de idade
x_teste_arvore = x_teste_arvore.reshape(-1, 1)  # Reshape para o formato esperado pelo modelo

# Atualiza o gráfico com a nova previsão detalhada
grafico = px.scatter(x=x_plano_saude.ravel(), y=y_plano_saude)  # Pontos reais
grafico.add_scatter(x=x_plano_saude.ravel(), y=regressor_arvore_saude.predict(x_teste_arvore), name='Regressão')  # Linha detalhada
grafico.show()

# Faz uma previsão para uma idade específica (exemplo: 40 anos)
print(regressor_arvore_saude.predict([[40]]))  # Retorna o custo previsto para 40 anos