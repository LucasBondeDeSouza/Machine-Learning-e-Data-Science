import pandas as pd  # Biblioteca para manipular dados em tabelas
from sklearn.model_selection import train_test_split  # Função para dividir dados em treino e teste
from sklearn.metrics import mean_absolute_error  # Calcula o erro médio absoluto
from sklearn.preprocessing import StandardScaler  # Escala os dados para padronização
from sklearn.svm import SVR  # Modelo de Regressão baseado em Máquina de Vetores de Suporte

# Carrega os dados de preços de casas do arquivo CSV
base_casas = pd.read_csv('house_prices.csv')

# Define as variáveis de entrada (x) e saída (y)
x_casas = base_casas.iloc[:, 3:19].values  # Características da casa (tamanho, localização, etc.)
y_casas = base_casas.iloc[:, 2].values  # Preços das casas (valor que queremos prever)

# Divide os dados em treino (70%) e teste (30%)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0  # Define proporção e garante repetibilidade
)

# Escala os dados de entrada e saída para valores padronizados
scaler_x_casas = StandardScaler()
x_casas_treinamento_scaled = scaler_x_casas.fit_transform(x_casas_treinamento)  # Escala os dados de treino
scaler_y_casas = StandardScaler()
y_casas_treinamento_scaled = scaler_y_casas.fit_transform(y_casas_treinamento.reshape(-1, 1))  # Escala saída de treino

x_casas_teste_scaled = scaler_x_casas.transform(x_casas_teste)  # Escala dados de teste
y_casas_teste_scaled = scaler_y_casas.transform(y_casas_teste.reshape(-1, 1))  # Escala saída de teste

# Cria o modelo de regressão usando kernel RBF
regressor_svr_casas = SVR(kernel='rbf')
regressor_svr_casas.fit(x_casas_treinamento_scaled, y_casas_treinamento_scaled.ravel())  # Treina o modelo

# Exibe a precisão (score) do modelo nos dados de treino e teste
print(regressor_svr_casas.score(x_casas_treinamento_scaled, y_casas_treinamento_scaled))  # Precisão no treino
print(regressor_svr_casas.score(x_casas_teste_scaled, y_casas_teste_scaled))  # Precisão no teste

# Faz previsões nos dados de teste
previsoes = regressor_svr_casas.predict(x_casas_teste_scaled)

# Converte os valores escalados de volta para a escala original
y_casas_teste_inverse = scaler_y_casas.inverse_transform(y_casas_teste_scaled)  # Valores reais do teste
privisoes_inverse = scaler_y_casas.inverse_transform(previsoes.reshape(-1, 1))  # Previsões convertidas

# Calcula o erro absoluto médio entre valores reais e previstos
print(mean_absolute_error(y_casas_teste_inverse, privisoes_inverse))  # Avalia a precisão