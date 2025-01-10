import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Carrega os dados do arquivo CSV, convertendo a coluna 'Month' para datas e definindo como índice
dataset = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')

# Converte o índice para o formato de data correto
dataset.index = pd.to_datetime(dataset.index, format='%Y-%m')

# Seleciona a série temporal de número de passageiros (número de passageiros por mês)
time_series = dataset['#Passengers']

# Decompõe a série temporal em 3 partes: tendência, sazonalidade e erro (aleatório)
decomposicao = seasonal_decompose(time_series)

# Extrai a parte da tendência (movimento de longo prazo)
tendencia = decomposicao.trend

# Extrai a parte sazonal (padrões que se repetem a cada ano)
sazonal = decomposicao.seasonal

# Extrai a parte aleatória (erro ou variação não explicada)
aleatorio = decomposicao.resid

# Plota a tendência (comportamento de longo prazo)
plt.plot(tendencia)
plt.show()

# Plota a sazonalidade (padrões de curto prazo)
plt.plot(sazonal)
plt.show()

# Plota o erro aleatório (variações inesperadas)
plt.plot(aleatorio)
plt.show()