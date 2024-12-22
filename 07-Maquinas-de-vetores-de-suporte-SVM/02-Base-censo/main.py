# Importação das bibliotecas necessárias
import pickle  # Para carregar arquivos serializados
from sklearn.svm import SVC  # Modelo de Support Vector Classifier (SVC)
from sklearn.metrics import accuracy_score, classification_report  # Métricas de avaliação

# Carrega os dados do arquivo 'census.pkl', que contém os dados de treinamento e teste.
with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

# Criação do modelo SVC (Support Vector Classifier) com o kernel radial basis function (rbf)
# O parâmetro `C` controla a regularização do modelo (quanto maior C, mais rígida a margem).
# O parâmetro `random_state` garante reprodutibilidade nos resultados.
svm_census = SVC(kernel='linear', random_state=1, C=1.0)

# Treina o modelo com os dados de entrada (x_census_treinamento) e suas respectivas saídas (y_census_treinamento)
svm_census.fit(x_census_treinamento, y_census_treinamento)

# Faz previsões no conjunto de teste usando o modelo treinado
previsoes = svm_census.predict(x_census_teste)

# Avalia o desempenho do modelo usando as previsões geradas e os valores reais
# A acurácia é a proporção de previsões corretas
print(accuracy_score(y_census_teste, previsoes))

# O classification_report fornece uma análise mais detalhada com métricas como precisão, recall, f1-score
print(classification_report(y_census_teste, previsoes))