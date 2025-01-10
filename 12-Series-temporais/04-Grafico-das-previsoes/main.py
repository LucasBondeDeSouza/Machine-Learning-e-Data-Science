# Importa bibliotecas necessárias para manipulação de dados e análise de séries temporais
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima  # Para modelar a série temporal

# Carrega os dados do arquivo CSV, tratando a coluna 'Month' como datas e definindo como índice da tabela
dataset = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')

# Converte o índice para o formato correto de data (ano-mês)
dataset.index = pd.to_datetime(dataset.index, format='%Y-%m')

# Extrai a série temporal de número de passageiros (número de passageiros por mês)
time_series = dataset['#Passengers']

# Separa os dados em conjunto de treinamento (130 primeiros meses) e teste (últimos meses)
train = time_series[: 130]  # Dados de treinamento
test = time_series[130 :]   # Dados de teste

# Cria o modelo ARIMA para prever os dados de treinamento
model2 = auto_arima(train, suppress_warnings=True)  # Ajuste automático do modelo ARIMA

# Faz previsões para os próximos 14 períodos (meses) usando o modelo ajustado
prediction = pd.DataFrame(model2.predict(n_periods = 14), index = test.index)
prediction.columns = ['passengers_predictions']  # Atribui nome à coluna de previsões
print(prediction)

# Plota os dados de treinamento, teste e previsões para visualização
plt.figure(figsize=(8, 5))
plt.plot(train, label = 'Training')  # Dados de treinamento
plt.plot(test, label = 'Test')  # Dados de teste
plt.plot(prediction, label = 'Predictions')  # Previsões feitas pelo modelo
plt.legend()  # Exibe a legenda
plt.show()  # Exibe o gráfico