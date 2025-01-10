import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

# Carrega os dados do arquivo CSV, convertendo a coluna 'Month' para datas e definindo como índice
dataset = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')

# Converte o índice para o formato de data correto
dataset.index = pd.to_datetime(dataset.index, format='%Y-%m')

# Seleciona a série temporal de número de passageiros (número de passageiros por mês)
time_series = dataset['#Passengers']

