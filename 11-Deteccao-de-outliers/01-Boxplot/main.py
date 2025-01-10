import pandas as pd
import plotly.express as px

# Carrega os dados do arquivo CSV
base_credit = pd.read_csv('credit_data.csv')

# Verifica quantos valores nulos existem em cada coluna
print(base_credit.isnull().sum())

# Remove as linhas com valores nulos
base_credit.dropna(inplace=True)

# Cria um gráfico de caixa (box plot) para visualizar a distribuição da idade
grafico = px.box(base_credit, y='age')
grafico.show()

# Identifica valores negativos na coluna "age" (dados incorretos)
outliers_age = base_credit[base_credit['age'] < 0]
print(outliers_age)

# Cria um gráfico de caixa para visualizar a distribuição dos empréstimos (loan)
grafico = px.box(base_credit, y='loan')
grafico.show()

# Identifica valores muito altos na coluna "loan" (possíveis outliers)
outliers_loan = base_credit[base_credit['loan'] > 13300]
print(outliers_loan)