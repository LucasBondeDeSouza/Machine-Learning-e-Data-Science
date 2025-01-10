# Importa bibliotecas necessárias para manipulação de dados e análise de séries temporais
import pandas as pd
from pmdarima.arima import auto_arima  # Para modelar a série temporal

# Carrega os dados do arquivo CSV, tratando a coluna 'Month' como datas e definindo como índice da tabela
dataset = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')

# Converte o índice para o formato correto de data (ano-mês)
dataset.index = pd.to_datetime(dataset.index, format='%Y-%m')

# Extrai a série temporal de número de passageiros (número de passageiros por mês)
time_series = dataset['#Passengers']

# Cria um modelo ARIMA automaticamente para prever a série temporal (ordem 2,1,2 significa que o modelo considera 2 termos autoregressivos, 1 diferença e 2 termos de média móvel)
model = auto_arima(time_series, order = (2, 1, 2))

# Faz a previsão para os próximos 24 meses (2 anos)
predictions = model.predict(n_periods = 24)

# Exibe as previsões geradas pelo modelo
print(predictions)