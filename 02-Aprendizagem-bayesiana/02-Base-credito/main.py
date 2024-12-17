# Importação das bibliotecas necessárias
import pickle # Para carregar os dados salvos em arquivos
from sklearn.naive_bayes import GaussianNB # Modelo de classificação Naive Bayes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # Métricas de avaliação

# 1. Carregamento dos dados pré-processados
# O arquivo 'credit.pkl' contém os dados de treino e teste já preparados:
# - x_credit_treinamento: atributos previsores para treino
# - y_credit_treinamento: classes alvo para treino
# - x_credit_teste: atributos previsores para teste
# - y_credit_teste: classes alvo para teste
with open('credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)


# 2. Instanciação e treinamento do modelo Naive Bayes
# GaussianNB é um modelo probabilístico baseado no Teorema de Bayes, eficiente para dados contínuos
naive_credit_data = GaussianNB()
naive_credit_data.fit(x_credit_treinamento, y_credit_treinamento)


# 3. Realização de previsões nos dados de teste
# O modelo treinado é usado para prever a classe com base nos atributos de teste
previsoes = naive_credit_data.predict(x_credit_teste)


# Exibição das métricas
print("Acurácia do modelo:")
print(accuracy_score(y_credit_teste, previsoes))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_credit_teste, previsoes))

print("\nRelatório de Classificação:")
print(classification_report(y_credit_teste, previsoes))