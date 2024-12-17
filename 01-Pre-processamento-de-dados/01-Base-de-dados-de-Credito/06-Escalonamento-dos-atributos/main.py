import pandas as pd
from sklearn.preprocessing import StandardScaler
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

# Exibe qual é a menor renda, a idade da pessoa mais nova e a menor dívida
print(x_credit[:, 0].min(), x_credit[:, 1].min(), x_credit[:, 2].min())

# Exibe qual é a maior renda, a idade da pessoa mais velha e a maior dívida
print(x_credit[:, 0].max(), x_credit[:, 1].max(), x_credit[:, 2].max())