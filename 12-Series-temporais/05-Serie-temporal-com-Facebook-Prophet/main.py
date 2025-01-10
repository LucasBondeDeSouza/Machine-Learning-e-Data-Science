import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt

# Carrega os dados do arquivo CSV
dataset = pd.read_csv('page_wikipedia.csv')

# Exibe estatísticas resumidas dos dados
print(dataset.describe())

# Gera histogramas das colunas numéricas
print(dataset.hist())
dataset.hist()
plt.show()

# Renomeia as colunas para o formato esperado pelo Prophet ('ds' = datas, 'y' = valores)
dataset = dataset[['date', 'views']].rename(columns={'date': 'ds', 'views': 'y'})

# Ordena os dados por data
dataset = dataset.sort_values(by='ds')

# Cria e treina o modelo Prophet
model = Prophet()
model.fit(dataset)

# Cria um conjunto de datas futuras para previsão (90 dias à frente)
future = model.make_future_dataframe(periods=90)

# Gera previsões para as datas futuras
forecast = model.predict(future)

# Exibe as primeiras e as últimas 90 previsões
print(forecast.head())
print(forecast.tail(90))


# Gera gráficos das previsões
model.plot(forecast, xlabel='Date', ylabel='Views')
plt.show()

# Exibe os componentes da previsão (tendência, sazonalidade, etc.)
model.plot_components(forecast)
plt.show()

# Gráficos interativos usando Plotly
plot_plotly(model, forecast)
plt.show()

plot_components_plotly(model, forecast)
plt.show()