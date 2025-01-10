import pandas as pd
import plotly.express as px

# Carrega os dados do arquivo CSV contendo informações de crédito
base_credit = pd.read_csv('credit_data.csv')

# Exibe a quantidade de valores nulos (faltando) em cada coluna
print(base_credit.isnull().sum())

# Remove as linhas que possuem valores nulos
base_credit.dropna(inplace=True)

# Cria um gráfico de dispersão mostrando a relação entre renda (income) e idade (age)
grafico = px.scatter(x=base_credit['income'], y=base_credit['age'])
grafico.show()

# Cria um gráfico de dispersão mostrando a relação entre renda (income) e empréstimo (loan)
grafico = px.scatter(x=base_credit['income'], y=base_credit['loan'])
grafico.show()

# Cria um gráfico de dispersão mostrando a relação entre idade (age) e empréstimo (loan)
grafico = px.scatter(x=base_credit['age'], y=base_credit['loan'])
grafico.show()

# Carrega os dados do arquivo CSV contendo informações do censo
base_census = pd.read_csv('census.csv')

# Cria um gráfico de dispersão mostrando a relação entre idade (age) e peso final (final-weight)
grafico = px.scatter(x=base_census['age'], y=base_census['final-weight'])
grafico.show()