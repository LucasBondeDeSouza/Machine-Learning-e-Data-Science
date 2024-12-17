import pandas as pd

# Lendo o arquivo csv
base_credit = pd.read_csv('../credit_data.csv')

# Retorna os dados do CSV que tem 'age' menor que 0
print(base_credit.loc[base_credit['age'] < 0])

# Apagar a coluna inteira (de todos os registros da base de dados)
base_credit2 = base_credit.drop('age', axis = 1)

# Apagar somente os registros com valores inconsistentes
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)


# Preencher os valores inconsistentes com a média das idades

# Descobrir a média do 'age' sem contar os dados inconsistentes
print(base_credit['age'][base_credit['age'] > 0].mean())
# Define os dados que tem 'age' inconsistente como 40.92
base_credit.loc[base_credit['age'] < 0, 'age' ] = 40.92