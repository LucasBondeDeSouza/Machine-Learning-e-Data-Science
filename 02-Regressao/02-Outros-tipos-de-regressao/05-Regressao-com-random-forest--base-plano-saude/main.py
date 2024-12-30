import plotly.express as px  # Biblioteca para criar gráficos interativos
import pandas as pd  # Biblioteca para manipular dados em tabelas (DataFrames)
import numpy as np  # Biblioteca para operações matemáticas e arrays
from sklearn.ensemble import RandomForestRegressor  # Modelo de Regressão de Floresta Aleatória

# Carrega os dados do arquivo CSV 'plano_saude.csv' para análise
base_plano_saude = pd.read_csv('plano_saude.csv')

# Separa os dados de entrada (idade) e saída (custo do plano de saúde)
x_plano_saude = base_plano_saude.iloc[:, 0:1].values  # Idade (variável independente)
y_plano_saude = base_plano_saude.iloc[:, 1].values  # Custo do plano (variável dependente)

# Cria o modelo de Floresta Aleatória e o treina com os dados de idade e custo
regressor_random_forest_saude = RandomForestRegressor(n_estimators=10)  # 10 árvores na floresta
regressor_random_forest_saude.fit(x_plano_saude, y_plano_saude)

# Avalia a performance do modelo nos dados de entrada
print(regressor_random_forest_saude.score(x_plano_saude, y_plano_saude))  # Mostra o R² (indicador de precisão do modelo)

# Gera novos valores para uma visualização mais detalhada da previsão
x_teste_arvore = np.arange(min(x_plano_saude), max(x_plano_saude), 0.1)  # Valores detalhados de idade (passo de 0.1)
x_teste_arvore = x_teste_arvore.reshape(-1, 1)  # Ajusta o formato dos dados para o modelo

# Cria o gráfico, mostrando os dados reais e a linha de previsão do modelo
grafico = px.scatter(x=x_plano_saude.ravel(), y=y_plano_saude)  # Plota os pontos reais de idade e custo
grafico.add_scatter(x=x_teste_arvore.ravel(), y=regressor_random_forest_saude.predict(x_teste_arvore), name='Regressão')  # Plota a linha da previsão
grafico.show()  # Exibe o gráfico

# Faz uma previsão do custo do plano de saúde para uma pessoa de 40 anos
print(regressor_random_forest_saude.predict([[40]]))  # Previsão do custo para 40 anos