# Importação das bibliotecas necessárias
import pickle # Para carregar os dados salvos em arquivos
from sklearn.naive_bayes import GaussianNB # Modelo de classificação Naive Bayes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # Métricas de avaliação

# 1. Carregamento dos dados pré-processados
# O arquivo 'census.pkl' contém os dados de treino e teste já preparados:
# - x_census_treinamento: atributos previsores para treino
# - y_census_treinamento: classes alvo para treino
# - x_census_teste: atributos previsores para teste
# - y_census_teste: classes alvo para teste
with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)


# 2. Instanciação e treinamento do modelo Naive Bayes
# GaussianNB é um modelo probabilístico baseado no Teorema de Bayes, eficiente para dados contínuos
naive_census = GaussianNB()
naive_census.fit(x_census_treinamento, y_census_treinamento)


# 3. Realização de previsões nos dados de teste
# O modelo treinado é usado para prever a classe com base nos atributos de teste
previsoes = naive_census.predict(x_census_teste)


# Exibição das métricas
# Exibição da acurácia do modelo
print("Acurácia do modelo:")
print(accuracy_score(y_census_teste, previsoes))

# Exibição da matriz de confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_census_teste, previsoes))

# Exibição do relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_census_teste, previsoes))