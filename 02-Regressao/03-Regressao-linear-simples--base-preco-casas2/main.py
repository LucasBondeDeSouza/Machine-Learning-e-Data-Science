import plotly.express as px  # Biblioteca para criar gráficos interativos
import plotly.graph_objects as go  # Necessário para combinar gráficos interativos
import seaborn as sns  # Biblioteca para visualização de dados
import matplotlib.pyplot as plt  # Biblioteca para criar gráficos simples
import pandas as pd  # Biblioteca para manipulação de tabelas (DataFrames)
import numpy as np  # Biblioteca para cálculos matemáticos e manipulação de arrays
from sklearn.model_selection import train_test_split  # Divide os dados em treino e teste
from sklearn.linear_model import LinearRegression  # Modelo de Regressão Linear
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Métricas de erro

# Carrega os dados de preços de casas
base_casas = pd.read_csv('house_prices.csv')

# Seleciona apenas colunas com valores numéricos para análise de correlação
numeric_cols = base_casas.select_dtypes(include=['float64', 'int64'])

# Cria um gráfico para mostrar a correlação entre as colunas numéricas (comentado)
# figura = plt.figure(figsize=(20, 20))
# sns.heatmap(numeric_cols.corr(), annot=True)  # Exibe a matriz de correlação com valores
# plt.show()

# Define a coluna de entrada (x) e a saída (y)
x_casas = base_casas.iloc[:, 5:6].values  # Exemplo: área ou número de quartos (coluna de entrada)
y_casas = base_casas.iloc[:, 2].values  # Preço das casas (coluna de saída)

# Divide os dados em treino (70%) e teste (30%)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0
)

# Cria o modelo de regressão linear e treina com os dados de treino
regressor_simples_casas = LinearRegression()
regressor_simples_casas.fit(x_casas_treinamento, y_casas_treinamento)

# Obtém o coeficiente b0 (intercepto) e b1 (inclinação da reta) da regressão (comentado)
# print(regressor_simples_casas.intercept_)  # Valor inicial da linha de regressão
# print(regressor_simples_casas.coef_)  # Inclinação da linha

# Faz previsões com os dados de teste
previsoes_teste = regressor_simples_casas.predict(x_casas_teste)

# Calcula o erro absoluto médio (diferença média entre valores reais e previstos)
print(mean_absolute_error(y_casas_teste, previsoes_teste))

# Calcula o erro quadrático médio (média dos erros elevados ao quadrado)
print(mean_squared_error(y_casas_teste, previsoes_teste))

# Calcula a raiz do erro quadrático médio (indica o erro em unidades reais)
print(np.sqrt(mean_squared_error(y_casas_teste, previsoes_teste)))

# Cria um gráfico de dispersão com os valores reais
grafico1 = px.scatter(x=x_casas_teste.ravel(), y=y_casas_teste)

# Cria uma linha representando a reta da regressão
grafico2 = px.line(x=x_casas_teste.ravel(), y=previsoes_teste)
grafico2.data[0].line.color = 'red'  # Altera a cor da linha para vermelho

# Combina os gráficos de dispersão e linha em um único gráfico
grafico3 = go.Figure(data=grafico1.data + grafico2.data)
grafico3.show()  # Exibe o gráfico combinado