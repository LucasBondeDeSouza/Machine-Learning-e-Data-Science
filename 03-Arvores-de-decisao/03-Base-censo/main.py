# Importação de bibliotecas necessárias
from sklearn.tree import DecisionTreeClassifier  # Para criar e treinar o modelo de árvore de decisão
import pickle  # Para carregar arquivos serializados (dados pré-processados e salvos)
from sklearn.metrics import accuracy_score, classification_report  # Para avaliar o desempenho do modelo

# Carregando os dados previamente processados e salvos em um arquivo 'census.pkl'
# Este arquivo contém os conjuntos de treinamento e teste já preparados
with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

# Criando um modelo de árvore de decisão
# criterion='entropy': Define que o modelo usará a métrica de entropia para decidir como dividir os dados
# random_state=0: Garante que os resultados sejam reproduzíveis (mesmo comportamento ao rodar várias vezes)
arvore_census = DecisionTreeClassifier(criterion='entropy', random_state=0)

# Treinando o modelo com os dados de treinamento
# x_census_treinamento: Conjunto de características (informações dos dados)
# y_census_treinamento: Conjunto de rótulos (classes correspondentes às características)
arvore_census.fit(x_census_treinamento, y_census_treinamento)

# Fazendo previsões com o modelo treinado usando o conjunto de teste
# x_census_teste: Conjunto de características que o modelo nunca viu antes
previsoes = arvore_census.predict(x_census_teste)

# Avaliando a precisão do modelo
# accuracy_score: Mede a porcentagem de previsões corretas
print(accuracy_score(y_census_teste, previsoes))

# Exibindo métricas detalhadas de avaliação do modelo
# classification_report: Mostra métricas como precisão, revocação e F1-Score para cada classe
print(classification_report(y_census_teste, previsoes))