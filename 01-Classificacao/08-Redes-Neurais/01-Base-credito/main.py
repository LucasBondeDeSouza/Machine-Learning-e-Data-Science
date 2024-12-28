# Importamos as bibliotecas necessárias
import pickle  # Usado para carregar dados salvos no formato binário
from sklearn.neural_network import MLPClassifier  # Rede neural do tipo perceptron multicamada
from sklearn.metrics import accuracy_score, classification_report  # Métricas para avaliar o modelo

# Carregando os dados
with open('credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Configuração da rede neural
# Estamos criando um modelo de rede neural usando MLPClassifier. Aqui estão os parâmetros usados:
# - max_iter=1500: Número máximo de iterações que o modelo realizará durante o treinamento.
# - verbose=True: Exibe informações no console sobre o progresso do treinamento.
# - tol=0.0000100: Tolerância para o critério de parada. O treinamento para quando a melhora for menor que esse valor.
# - solver='adam': Algoritmo usado para otimizar os pesos da rede. "Adam" é eficiente para grandes conjuntos de dados.
# - activation='relu': Função de ativação usada nos neurônios. "ReLU" ajuda a lidar com problemas de não-linearidade.
# - hidden_layer_sizes=(2, 2): Configuração das camadas ocultas. Aqui temos 2 camadas ocultas, cada uma com 2 neurônios.
rede_neural_credit = MLPClassifier(
    max_iter=1500,
    verbose=True,
    tol=0.0000100,
    solver='adam',
    activation='relu',
    hidden_layer_sizes=(2, 2)
)

# Treinamento do modelo
rede_neural_credit.fit(x_credit_treinamento, y_credit_treinamento)

# Realizando previsões
previsoes = rede_neural_credit.predict(x_credit_teste)

# Avaliação do modelo: Métrica de precisão
print(accuracy_score(y_credit_teste, previsoes))

# Avaliação do modelo: Relatório detalhado
print(classification_report(y_credit_teste, previsoes))