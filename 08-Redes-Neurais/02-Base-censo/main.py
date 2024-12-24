# Importamos as bibliotecas necessárias
import pickle  # Usado para carregar dados salvos no formato binário
from sklearn.neural_network import MLPClassifier  # Rede neural do tipo perceptron multicamada
from sklearn.metrics import accuracy_score, classification_report  # Métricas para avaliar o modelo

# Carregando os dados
with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

# Configuração da rede neural
# Estamos criando um modelo de rede neural usando MLPClassifier. Aqui estão os parâmetros usados:
# - max_iter=1000: Número máximo de iterações que o modelo realizará durante o treinamento.
# - verbose=True: Exibe informações no console sobre o progresso do treinamento.
# - tol=0.000010: Tolerância para o critério de parada. O treinamento para quando a melhora for menor que esse valor.
# - hidden_layer_sizes=(55, 55): Configuração das camadas ocultas. Aqui temos 2 camadas ocultas, cada uma com 55 neurônios.
rede_neural_census = MLPClassifier(
    max_iter=1000,
    verbose=True,
    tol=0.000010,
    hidden_layer_sizes=(55, 55)
)

# Treinamento do modelo
rede_neural_census.fit(x_census_treinamento, y_census_treinamento)

# Realizando previsões
previsoes = rede_neural_census.predict(x_census_teste)

# Avaliação do modelo: Métrica de precisão
print(accuracy_score(y_census_teste, previsoes))

# Avaliação do modelo: Relatório detalhado
print(classification_report(y_census_teste, previsoes))