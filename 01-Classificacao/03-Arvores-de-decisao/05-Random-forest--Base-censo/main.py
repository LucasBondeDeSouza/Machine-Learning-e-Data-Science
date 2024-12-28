# Importa as bibliotecas necessárias
from sklearn.ensemble import RandomForestClassifier  # Para criar o modelo de Random Forest
import pickle  # Para carregar os dados salvos
from sklearn.metrics import accuracy_score, classification_report  # Para medir a qualidade do modelo

# Carrega os dados de um arquivo chamado 'census.pkl'
with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

# Cria o modelo de Random Forest
# n_estimators=40: Usa 100 árvores no modelo
# criterion='entropy': Critério usado para construir as árvores
random_forest_census = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

# Treina o modelo com os dados de treinamento
random_forest_census.fit(x_census_treinamento, y_census_treinamento)

# Faz previsões com os dados de teste
previsoes = random_forest_census.predict(x_census_teste)

# Mostra a precisão (percentual de acertos) do modelo
print(accuracy_score(y_census_teste, previsoes))

# Mostra métricas detalhadas sobre as previsões
print(classification_report(y_census_teste, previsoes))