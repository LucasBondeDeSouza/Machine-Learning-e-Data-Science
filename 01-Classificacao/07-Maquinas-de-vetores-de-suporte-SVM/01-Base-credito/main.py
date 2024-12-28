# Importação das bibliotecas necessárias
import pickle  # Para carregar arquivos serializados
from sklearn.svm import SVC  # Modelo de Support Vector Classifier (SVC)
from sklearn.metrics import accuracy_score, classification_report  # Métricas de avaliação

# Carrega os dados do arquivo 'credit.pkl', que contém os dados de treinamento e teste.
with open('credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Criação do modelo SVC (Support Vector Classifier) com o kernel radial basis function (rbf)
# O parâmetro `C` controla a regularização do modelo (quanto maior C, mais rígida a margem).
# O parâmetro `random_state` garante reprodutibilidade nos resultados.
svm_credit = SVC(kernel='rbf', random_state=1, C=2.0)

# Treina o modelo com os dados de entrada (x_credit_treinamento) e suas respectivas saídas (y_credit_treinamento)
svm_credit.fit(x_credit_treinamento, y_credit_treinamento)

# Faz previsões no conjunto de teste usando o modelo treinado
previsoes = svm_credit.predict(x_credit_teste)

# Avalia o desempenho do modelo usando as previsões geradas e os valores reais
# A acurácia é a proporção de previsões corretas
print(accuracy_score(y_credit_teste, previsoes))

# O classification_report fornece uma análise mais detalhada com métricas como precisão, recall, f1-score
print(classification_report(y_credit_teste, previsoes))