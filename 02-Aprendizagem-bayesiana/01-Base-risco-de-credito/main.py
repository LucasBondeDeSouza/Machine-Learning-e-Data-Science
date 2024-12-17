import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import GaussianNB

# 1. Lendo o arquivo CVS
base_risco_credito = pd.read_csv('risco_credito.csv')


# 2. Separação dos atributos previsores (X) e da classe alvo (y)
# Atributos previsores: colunas 0 a 3 (história, dívida, garantias, renda)
x_risco_credito = base_risco_credito.iloc[:, 0:4].values

# Classe alvo: coluna 4 (risco de crédito)
y_risco_credito = base_risco_credito.iloc[:, 4].values


# 3. Codificação das variáveis categóricas para valores numéricos
# O LabelEncoder é usado para converter strings em números inteiros
# Exemplo: 'boa' -> 0, 'ruim' -> 1

# Instanciando o LabelEncoder para cada coluna categórica
label_enconder_historia = LabelEncoder()
label_enconder_divida = LabelEncoder()
label_enconder_garantias = LabelEncoder()
label_enconder_renda = LabelEncoder()

# Aplicando o LabelEncoder para cada coluna
x_risco_credito[:, 0] = label_enconder_historia.fit_transform(x_risco_credito[:, 0])
x_risco_credito[:, 1] = label_enconder_divida.fit_transform(x_risco_credito[:, 1])
x_risco_credito[:, 2] = label_enconder_garantias.fit_transform(x_risco_credito[:, 2])
x_risco_credito[:, 3] = label_enconder_renda.fit_transform(x_risco_credito[:, 3])


# 4. Salvando os dados codificados em um arquivo usando pickle
# Isso permite reutilizar os dados transformados posteriormente sem recalcular
with open('risco_cedito.pkl', 'wb') as f:
    pickle.dump([x_risco_credito, y_risco_credito], f)


# 5. Treinamento do modelo Naive Bayes
# Usando o algoritmo GaussianNB para classificação de risco de crédito
naive_risco_credito = GaussianNB()
naive_risco_credito.fit(x_risco_credito, y_risco_credito)


# 6. Realizando previsões em novos dados
# Teste com duas amostras:
# [historia, divida, garantias, renda]
# - Amostra 1: história boa (0), dívida alta (0), garantias nenhuma (1), renda > 35 (2)
# - Amostra 2: história ruim (2), dívida alta (0), garantias adequada (0), renda < 15 (0)
novas_amostras = [[0, 0, 1, 2], [2, 0, 0, 0]]

# Realizando a previsão
previsao = naive_risco_credito.predict(novas_amostras)

print(previsao)