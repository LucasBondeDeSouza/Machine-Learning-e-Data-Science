import plotly.express as px  # Biblioteca para criar gráficos interativos
import pandas as pd  # Biblioteca para manipulação de dados em tabelas (DataFrames)
from sklearn.linear_model import LinearRegression  # Importa o modelo de Regressão Linear
from sklearn.preprocessing import PolynomialFeatures  # Para criar features polinomiais

# Carrega os dados de um arquivo CSV para análise
base_plano_saude = pd.read_csv('plano_saude.csv')

# Separa a coluna de idade (independente) e custo (dependente)
x_plano_saude = base_plano_saude.iloc[:, 0:1].values  # Idades
y_plano_saude = base_plano_saude.iloc[:, 1].values  # Custos

# Transforma os dados de idade em uma forma polinomial (ex.: x, x^2)
poly = PolynomialFeatures(degree=2)  # Define um modelo de 2ª ordem (parábola)
x_plano_saude_poly = poly.fit_transform(x_plano_saude)  # Aplica a transformação

# Treina o modelo de Regressão Linear com os dados polinomiais
regressor_saude_polinomial = LinearRegression()
regressor_saude_polinomial.fit(x_plano_saude_poly, y_plano_saude)

# Gera previsões de custo com base nas idades
previsoes = regressor_saude_polinomial.predict(x_plano_saude_poly)

# Cria um gráfico interativo para visualizar os dados reais e as previsões
grafico = px.scatter(x=x_plano_saude[:, 0], y=y_plano_saude)  # Pontos reais (idade x custo)
grafico.add_scatter(x=x_plano_saude[:, 0], y=previsoes, name='Regressão')  # Linha da previsão
grafico.show()  # Exibe o gráfico