import plotly.express as px  # Biblioteca para criar gráficos interativos
import seaborn as sns  # Biblioteca para visualização de dados
import matplotlib.pyplot as plt  # Biblioteca para criar gráficos estáticos
import pandas as pd  # Biblioteca para manipulação de dados em tabelas (DataFrames)
import numpy as np  # Biblioteca para manipulação de arrays numéricos
from sklearn.linear_model import LinearRegression  # Importa o modelo de Regressão Linear
from yellowbrick.regressor import ResidualsPlot  # Biblioteca para gráficos de resíduos de regressão

# Carrega os dados do arquivo CSV 'plano_saude.csv' para um DataFrame
base_plano_saude = pd.read_csv('plano_saude.csv')

# Separa os dados em variáveis independentes (x) e dependentes (y)
x_plano_saude = base_plano_saude.iloc[:, 0].values  # Primeira coluna como variável x (ex: idade)
y_plano_saude = base_plano_saude.iloc[:, 1].values  # Segunda coluna como variável y (ex: custo)

# Reshape da variável x para que tenha a forma correta para o modelo de regressão
x_plano_saude = x_plano_saude.reshape(-1, 1)

# Cria o modelo de regressão linear
regressor_plano_saude = LinearRegression()

# Treina o modelo com os dados de x e y
regressor_plano_saude.fit(x_plano_saude, y_plano_saude)

# Faz previsões com o modelo treinado
previsoes = regressor_plano_saude.predict(x_plano_saude)

# Cria um gráfico de dispersão (scatter) com os dados reais
grafico = px.scatter(x = x_plano_saude.ravel(), y = y_plano_saude)

# Adiciona a linha de regressão ao gráfico
grafico.add_scatter(x = x_plano_saude.ravel(), y = previsoes, name = 'Regressão')
grafico.show()

# A linha abaixo faz uma previsão manual para o valor de x = 18
print(regressor_plano_saude.intercept_ + regressor_plano_saude.coef_ * 18)
print(regressor_plano_saude.predict([[18]]))

# Avalia o modelo, mostrando o quanto ele se ajusta aos dados
print(regressor_plano_saude.score(x_plano_saude, y_plano_saude))

# Cria o gráfico de resíduos para verificar a precisão da regressão
visualizador = ResidualsPlot(regressor_plano_saude)
visualizador.fit(x_plano_saude, y_plano_saude)  # Ajusta o gráfico aos dados
visualizador.poof()  # Exibe o gráfico de resíduos