import pandas as pd

# Lendo o arquivo csv
base_credit = pd.read_csv('../credit_data.csv')

# os dados que serão usados como entrada para alguma análise (por exemplo, idade ou salário de pessoas).
x_credit = base_credit.iloc[:, 1:4].values

# o resultado esperado que queremos prever ou analisar (por exemplo, se a pessoa pagou um empréstimo (1) ou não (0)).
y_credit = base_credit.iloc[:, 4].values