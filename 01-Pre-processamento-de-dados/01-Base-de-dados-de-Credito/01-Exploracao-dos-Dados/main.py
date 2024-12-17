import pandas as pd

# Lendo o arquivo csv
base_credit = pd.read_csv('../credit_data.csv')

# Mostra os 5 primeiros dados do CSV
print(base_credit.head())

# Mostra os 10 primeiros dados do CSV
print(base_credit.head(10))

# Mostra os 5 útimos dados do CSV
print(base_credit.tail())

# Descreve dos dados do CSV
print(base_credit.describe())

# Filtrando pela Renda os dados do CSV
print(base_credit[base_credit['income'] >= 69995.685578])

# Filtrando pela dívida os dados do CSV
print(base_credit[base_credit['loan'] <= 1.377630])