import plotly.express as px  # Biblioteca para criar gráficos interativos
import pandas as pd  # Biblioteca para manipular dados em tabelas (DataFrames)
from sklearn.preprocessing import StandardScaler  # Escala os dados para facilitar o treinamento
from sklearn.neural_network import MLPRegressor  # Modelo de Rede Neural para Regressão

# Carrega os dados do arquivo CSV 'plano_saude.csv' para análise
base_plano_saude = pd.read_csv('plano_saude.csv')

# Separa os dados de entrada (idade) e saída (custo do plano de saúde)
x_plano_saude = base_plano_saude.iloc[:, 0:1].values  # Idade (variável independente)
y_plano_saude = base_plano_saude.iloc[:, 1].values  # Custo do plano (variável dependente)

# Escala os dados de entrada (idade) para manter os valores em uma faixa padrão
scaler_x = StandardScaler()
x_plano_saude_scaled = scaler_x.fit_transform(x_plano_saude)

# Escala os dados de saída (custo) para facilitar a convergência do modelo
scaler_y = StandardScaler()
y_plano_saude_scaled = scaler_y.fit_transform(y_plano_saude.reshape(-1, 1))

# Cria e treina um modelo de Rede Neural para prever o custo do plano de saúde
regressor_rna_saude = MLPRegressor(max_iter=1000)  # Define o modelo com até 1000 iterações
regressor_rna_saude.fit(x_plano_saude_scaled, y_plano_saude_scaled.ravel())  # Treina o modelo com os dados escalados

# Exibe a pontuação do modelo (o quão bem ele se ajusta aos dados)
print(regressor_rna_saude.score(x_plano_saude_scaled, y_plano_saude_scaled))

# Cria um gráfico interativo para visualizar os dados reais e a previsão do modelo
grafico = px.scatter(x=x_plano_saude_scaled.ravel(), y=y_plano_saude_scaled.ravel())  # Plota os dados reais
grafico.add_scatter(x=x_plano_saude_scaled.ravel(), y=regressor_rna_saude.predict(x_plano_saude_scaled), name="Regressão")  # Adiciona as previsões
grafico.show()  # Mostra o gráfico

# Faz uma previsão para um novo valor de idade (exemplo: idade 40 anos)
novo = [[40]]  # Define a nova entrada
novo = scaler_x.transform(novo)  # Escala o valor para o mesmo padrão dos dados de entrada
# Faz a previsão, converte o resultado escalado de volta ao valor original e exibe
print(scaler_y.inverse_transform(regressor_rna_saude.predict(novo).reshape(-1, 1)))