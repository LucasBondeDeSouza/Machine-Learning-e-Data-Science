# O código começa importando algumas ferramentas para ajudar a trabalhar com dados e fazer previsões.
import pickle  # "pickle" é usado para carregar e salvar informações em arquivos.
from sklearn.neighbors import KNeighborsClassifier  # Este é um algoritmo chamado kNN (k-vizinhos mais próximos).
from sklearn.metrics import accuracy_score, classification_report  # Ferramentas para medir o quão boas são as previsões.

# Aqui estamos abrindo um arquivo onde estão armazenados os dados que vamos usar para treinar o modelo.
with open('credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Agora, estamos criando um modelo kNN que vai aprender a fazer previsões.
knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)  
# 'n_neighbors=5' significa que o modelo vai olhar para os 5 vizinhos mais próximos para decidir a previsão.
# 'metric' e 'p' são ajustes que ajudam a calcular as distâncias entre os dados.

# Aqui, o modelo "aprende" com os dados de treinamento. 
# Isso significa que ele vai usar as informações de 'x_credit_treinamento' para fazer previsões com base em 'y_credit_treinamento'.
knn_credit.fit(x_credit_treinamento, y_credit_treinamento)

# Agora que o modelo aprendeu, ele vai tentar fazer previsões com os dados de teste.
previsoes = knn_credit.predict(x_credit_teste)  # O modelo faz previsões com base nas informações de 'x_credit_teste'.

# O código vai mostrar o quão boas foram as previsões do modelo:
print(accuracy_score(y_credit_teste, previsoes))  # Isso vai mostrar a "precisão" do modelo, ou seja, o quanto ele acertou.

# O "classification_report" dá um relatório mais detalhado sobre as previsões, incluindo acertos e erros.
print(classification_report(y_credit_teste, previsoes))