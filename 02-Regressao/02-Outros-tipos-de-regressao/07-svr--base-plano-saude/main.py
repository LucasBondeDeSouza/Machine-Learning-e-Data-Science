import plotly.express as px  # Biblioteca para criar gráficos interativos
import pandas as pd  # Biblioteca para manipular dados em tabelas (DataFrames)
from sklearn.svm import SVR  # Modelo de Regressão com Máquina de Vetores de Suporte
from sklearn.preprocessing import StandardScaler  # Escala os dados para facilitar o treinamento

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

# **Regressão com SVR usando kernel linear**
regressor_svr_saude_linear = SVR(kernel='linear')  # Configura um modelo com kernel linear
regressor_svr_saude_linear.fit(x_plano_saude, y_plano_saude)  # Treina o modelo com os dados originais

# Cria um gráfico mostrando os dados reais e a linha de regressão linear
grafico = px.scatter(x=x_plano_saude.ravel(), y=y_plano_saude.ravel())  # Gráfico com os dados reais
grafico.add_scatter(
    x=x_plano_saude.ravel(),
    y=regressor_svr_saude_linear.predict(x_plano_saude),
    name='Regressão'  # Adiciona a linha de previsão do modelo linear
)
grafico.show()

# **Regressão com SVR usando kernel polinomial**
regressor_svr_saude_poly = SVR(kernel='poly', degree=4)  # Configura um modelo com kernel polinomial (grau 4)
regressor_svr_saude_poly.fit(x_plano_saude, y_plano_saude)  # Treina o modelo com os dados originais

# Cria um gráfico mostrando os dados reais e a linha de regressão polinomial
grafico = px.scatter(x=x_plano_saude.ravel(), y=y_plano_saude.ravel())  # Gráfico com os dados reais
grafico.add_scatter(
    x=x_plano_saude.ravel(),
    y=regressor_svr_saude_poly.predict(x_plano_saude),
    name='Regressão'  # Adiciona a linha de previsão do modelo polinomial
)
grafico.show()

# **Regressão com SVR usando kernel RBF (Radial Basis Function)**
regressor_svr_saude_rbf = SVR(kernel='rbf')  # Configura um modelo com kernel RBF
regressor_svr_saude_rbf.fit(x_plano_saude_scaled, y_plano_saude_scaled.ravel())  # Treina com os dados escalados

# Cria um gráfico mostrando os dados escalados e a linha de regressão RBF
grafico = px.scatter(x=x_plano_saude_scaled.ravel(), y=y_plano_saude_scaled.ravel())  # Gráfico com os dados escalados
grafico.add_scatter(
    x=x_plano_saude_scaled.ravel(),
    y=regressor_svr_saude_rbf.predict(x_plano_saude_scaled),
    name='Regressão'  # Adiciona a linha de previsão do modelo RBF
)
grafico.show()

# **Previsão para um novo valor de idade (exemplo: 40 anos)**
novo = [[40]]  # Idade para previsão
novo = scaler_x.transform(novo)  # Escala o valor para manter o mesmo padrão dos dados de entrada
print(novo)  # Mostra o valor escalado da idade (40 anos)