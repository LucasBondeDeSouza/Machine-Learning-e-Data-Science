import pandas as pd
from pyod.models.knn import KNN  # Importa o modelo KNN para detecção de outliers
import numpy as np

# Carrega os dados do arquivo CSV contendo informações de crédito
base_credit = pd.read_csv('credit_data.csv')

# Exibe a quantidade de valores ausentes em cada coluna
print(base_credit.isnull().sum())

# Remove as linhas que possuem valores nulos
base_credit.dropna(inplace=True)

# Cria um detector de outliers usando o modelo KNN
detector = KNN()

# Treina o detector usando as colunas 1 a 3 do dataset
detector.fit(base_credit.iloc[:, 1:4])

# Obtém as previsões do modelo (1 = outlier, 0 = normal)
previsoes = detector.labels_

# Mostra quantos valores normais e outliers foram encontrados
print(np.unique(previsoes, return_counts=True))

# Obtém a pontuação de confiança para cada previsão (quanto maior, mais anômalo)
confianca_previsoes = detector.decision_scores_

# Identifica os índices dos outliers detectados
outliers = []
for i in range(len(previsoes)):
    if previsoes[i] == 1:
        outliers.append(i)

# Cria uma lista com os dados dos outliers detectados
lista_outliers = base_credit.iloc[outliers, :]

# Exibe os outliers encontrados
print(lista_outliers)