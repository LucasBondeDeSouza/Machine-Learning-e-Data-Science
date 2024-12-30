import plotly.express as px  # Biblioteca para criar gráficos interativos
import pandas as pd  # Biblioteca para manipulação de dados em tabelas (DataFrames)
from sklearn.linear_model import LinearRegression  # Modelo de Regressão Linear
from sklearn.preprocessing import PolynomialFeatures  # Criação de características polinomiais
from sklearn.model_selection import train_test_split  # Divide os dados em treino e teste
from sklearn.metrics import mean_absolute_error  # Calcula o erro absoluto médio

# Carrega os dados do arquivo CSV com preços de casas
base_casas = pd.read_csv('house_prices.csv')

# Define as variáveis de entrada (x) e saída (y)
x_casas = base_casas.iloc[:, 3:19].values  # Dados como tamanho, localização, etc.
y_casas = base_casas.iloc[:, 2].values  # Preços das casas (o que queremos prever)

# Divide os dados em treinamento (70%) e teste (30%)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0
)

# Cria características polinomiais de grau 2 (ajuda a capturar relações não lineares)
poly = PolynomialFeatures(degree=2)
x_casas_treinamento_poly = poly.fit_transform(x_casas_treinamento)  # Ajusta dados de treino
x_casas_teste_poly = poly.transform(x_casas_teste)  # Ajusta dados de teste

# Cria o modelo de Regressão Linear e treina com os dados polinomiais
regressor_casas_poly = LinearRegression()
regressor_casas_poly.fit(x_casas_treinamento_poly, y_casas_treinamento)

# Exibe a precisão (R²) do modelo no conjunto de treinamento e teste
print(regressor_casas_poly.score(x_casas_treinamento_poly, y_casas_treinamento))  # Precisão no treino
print(regressor_casas_poly.score(x_casas_teste_poly, y_casas_teste))  # Precisão no teste

# Faz previsões com o modelo nos dados de teste
previsoes = regressor_casas_poly.predict(x_casas_teste_poly)

# Calcula o erro absoluto médio (quanto, em média, o modelo errou nas previsões)
print(mean_absolute_error(y_casas_teste, previsoes))