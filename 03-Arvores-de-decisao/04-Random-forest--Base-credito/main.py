# Importa as bibliotecas necessárias
from sklearn.ensemble import RandomForestClassifier  # Para criar o modelo de Random Forest
import pickle  # Para carregar os dados salvos
from sklearn.metrics import accuracy_score, classification_report  # Para medir a qualidade do modelo

# Carrega os dados de um arquivo chamado 'credit.pkl'
with open('credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Cria o modelo de Random Forest
# n_estimators=40: Usa 40 árvores no modelo
# criterion='entropy': Critério usado para construir as árvores
random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)

# Treina o modelo com os dados de treinamento
random_forest_credit.fit(x_credit_treinamento, y_credit_treinamento)

# Faz previsões com os dados de teste
previsoes = random_forest_credit.predict(x_credit_teste)

# Mostra a precisão (percentual de acertos) do modelo
print(accuracy_score(y_credit_teste, previsoes))

# Mostra métricas detalhadas sobre as previsões
print(classification_report(y_credit_teste, previsoes))