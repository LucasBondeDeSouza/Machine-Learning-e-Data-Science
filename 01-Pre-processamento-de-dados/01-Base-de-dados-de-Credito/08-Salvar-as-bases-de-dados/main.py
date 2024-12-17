import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

scaler_credit = StandardScaler()

# Lendo o arquivo csv
base_credit = pd.read_csv('../credit_data.csv')

# os dados que serão usados como entrada para alguma análise (por exemplo, idade ou salário de pessoas).
x_credit = base_credit.iloc[:, 1:4].values

# Padronizando os dados de x_credit para que todas as colunas (idade, salário, dívida) tenham média 0 e desvio padrão 1.
# Isso é importante para colocar os valores em uma escala uniforme, especialmente útil para algoritmos de Machine Learning.
x_credit = scaler_credit.fit_transform(x_credit)

# o resultado esperado que queremos prever ou analisar (por exemplo, se a pessoa pagou um empréstimo (1) ou não (0)).
y_credit = base_credit.iloc[:, 4].values


# Treinamento e Teste

# Dividindo os dados em duas partes: uma para treinar o modelo (75%) e outra para testar o modelo (25%).
# - "x_credit" são os dados de entrada (como idade, salário e dívida).
# - "y_credit" são os resultados esperados (como se pagou ou não o empréstimo).
# - "test_size = 0.25" significa que 25% dos dados serão usados para teste.
# - "random_state = 0" garante que a divisão dos dados sempre aconteça do mesmo jeito.
x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(
    x_credit, y_credit, test_size = 0.25, random_state = 0
)

with open('credit.pkl', mode = 'wb') as f:
    pickle.dump(
        [x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f
    )