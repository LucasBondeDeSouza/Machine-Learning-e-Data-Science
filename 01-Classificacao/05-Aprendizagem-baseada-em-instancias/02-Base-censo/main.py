# O código começa importando algumas ferramentas para ajudar a trabalhar com dados e fazer previsões.
import pickle  # "pickle" é usado para carregar e salvar informações em arquivos.
from sklearn.neighbors import KNeighborsClassifier  # Este é um algoritmo chamado kNN (k-vizinhos mais próximos).
from sklearn.metrics import accuracy_score, classification_report  # Ferramentas para medir o quão boas são as previsões.

# Aqui estamos abrindo um arquivo onde estão armazenados os dados que vamos usar para treinar o modelo.
with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

# Agora, estamos criando um modelo kNN que vai aprender a fazer previsões.
knn_census = KNeighborsClassifier(n_neighbors=10)  
# 'n_neighbors=10' significa que o modelo vai olhar para os 10 vizinhos mais próximos para decidir a previsão.

# Aqui, o modelo "aprende" com os dados de treinamento. 
# Isso significa que ele vai usar as informações de 'x_census_treinamento' para fazer previsões com base em 'y_census_treinamento'.
knn_census.fit(x_census_treinamento, y_census_treinamento)

# Agora que o modelo aprendeu, ele vai tentar fazer previsões com os dados de teste.
previsoes = knn_census.predict(x_census_teste)  # O modelo faz previsões com base nas informações de 'x_census_teste'.

# O código vai mostrar o quão boas foram as previsões do modelo:
print(accuracy_score(y_census_teste, previsoes))  # Isso vai mostrar a "precisão" do modelo, ou seja, o quanto ele acertou.

# O "classification_report" dá um relatório mais detalhado sobre as previsões, incluindo acertos e erros.
print(classification_report(y_census_teste, previsoes))