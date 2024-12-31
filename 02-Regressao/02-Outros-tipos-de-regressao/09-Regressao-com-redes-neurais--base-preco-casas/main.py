import pandas as pd  # Biblioteca para manipular dados em tabelas
from sklearn.model_selection import train_test_split  # Divide os dados em treino e teste
from sklearn.metrics import mean_absolute_error  # Mede o erro médio entre previsões e valores reais
from sklearn.preprocessing import StandardScaler  # Padroniza os dados para facilitar o aprendizado
from sklearn.neural_network import MLPRegressor  # Rede Neural para prever valores numéricos

# Carrega os dados de um arquivo CSV contendo informações de casas
base_casas = pd.read_csv('house_prices.csv')

# Define os dados de entrada (características das casas) e saída (preço das casas)
x_casas = base_casas.iloc[:, 3:19].values  # Características da casa, como tamanho e localização
y_casas = base_casas.iloc[:, 2].values  # Preço da casa (o que queremos prever)

# Divide os dados em 70% para treinamento e 30% para teste
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0  # Define o tamanho das divisões e garante repetibilidade
)

# Padroniza os dados de entrada (x) e saída (y) para manter as variáveis na mesma escala
scaler_x_casas = StandardScaler()  # Cria um escalador para os dados de entrada
x_casas_treinamento_scaled = scaler_x_casas.fit_transform(x_casas_treinamento)  # Ajusta e transforma os dados de treino
scaler_y_casas = StandardScaler()  # Cria um escalador para os dados de saída
y_casas_treinamento_scaled = scaler_y_casas.fit_transform(y_casas_treinamento.reshape(-1, 1))  # Ajusta e transforma a saída

# Escala os dados de teste com base no ajuste dos dados de treino
x_casas_teste_scaled = scaler_x_casas.transform(x_casas_teste)
y_casas_teste_scaled = scaler_y_casas.transform(y_casas_teste.reshape(-1, 1))

# Cria e treina uma Rede Neural com duas camadas ocultas, cada uma com 9 neurônios
regressor_rna_casas = MLPRegressor(max_iter=1000, hidden_layer_sizes=(9, 9))  # Configura o modelo
regressor_rna_casas.fit(x_casas_treinamento_scaled, y_casas_treinamento_scaled.ravel())  # Treina a Rede Neural

# Mostra a precisão do modelo nos dados de treino e de teste
print(regressor_rna_casas.score(x_casas_treinamento_scaled, y_casas_treinamento_scaled))  # Precisão nos dados de treino
print(regressor_rna_casas.score(x_casas_teste_scaled, y_casas_teste_scaled))  # Precisão nos dados de teste

# Faz previsões com os dados de teste
previsoes = regressor_rna_casas.predict(x_casas_teste_scaled)

# Desfaz a padronização dos dados para comparar previsões e valores reais
y_casas_teste_inverse = scaler_y_casas.inverse_transform(y_casas_teste_scaled)  # Valor real desescalado
previsoes_inverse = scaler_y_casas.inverse_transform(previsoes.reshape(-1, 1))  # Previsões desescaladas

# Calcula o erro médio absoluto entre as previsões e os valores reais
print(mean_absolute_error(y_casas_teste_inverse, previsoes_inverse))  # Mostra o erro médio absoluto