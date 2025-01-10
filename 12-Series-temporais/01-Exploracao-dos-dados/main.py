import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Carrega os dados do arquivo CSV, convertendo a coluna 'Month' para datas e definindo como índice
dataset = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')

# Converte o índice para o formato de data correto
dataset.index = pd.to_datetime(dataset.index, format='%Y-%m')

# Seleciona a série temporal de número de passageiros
time_series = dataset['#Passengers']

# Exibe o segundo valor da série
print(time_series[1])

# Exibe o número de passageiros em fevereiro de 1949
print(time_series['1949-02'])

# Outra forma de acessar o mesmo dado usando datetime
print(time_series[datetime(1949, 2, 1)])

# Exibe os dados de janeiro a julho de 1950
print(time_series['1950-01-01': '1950-07-31'])

# Exibe os dados até julho de 1950
print(time_series[: '1950-07-31'])

# Exibe os dados do ano de 1950
print(time_series['1950'])

# Exibe a última data disponível na série
print(time_series.index.max())

# Exibe a primeira data disponível na série
print(time_series.index.min())

# Plota a série temporal completa
plt.plot(time_series)
plt.show()

# Agrega os dados por ano e soma os valores, depois plota o gráfico
time_series_ano = time_series.resample('A').sum()
plt.plot(time_series_ano)
plt.show()

# Agrega os dados por mês (independente do ano) e soma os valores, depois plota o gráfico
time_series_mes = time_series.groupby([lambda x: x.month]).sum()
plt.plot(time_series_mes)
plt.show()

# Filtra os dados do ano de 1960 e plota o gráfico
time_series_datas = time_series['1960-01-01' : '1960-12-01']
plt.plot(time_series_datas)
plt.show()