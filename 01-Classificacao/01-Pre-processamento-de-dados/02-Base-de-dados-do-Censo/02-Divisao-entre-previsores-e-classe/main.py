import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()

# Lendo o arquivo csv
base_census = pd.read_csv('../census.csv')

# Selecionando apenas as colunas 1 até 13 da tabela (dados de entrada) e guardando em 'x_census'.
x_census = base_census.iloc[:, 1:14].values

# Pegando apenas a última coluna da tabela (coluna 14), que será a "resposta" ou o resultado esperado.
y_census = base_census.iloc[:, 14].values