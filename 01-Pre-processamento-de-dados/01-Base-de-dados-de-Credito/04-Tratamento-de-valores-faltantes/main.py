import pandas as pd

# Lendo o arquivo csv
base_credit = pd.read_csv('credit_data.csv')

# Procura se não tem nenhum valor faltante, se tiver aparece como 'True' se não aparece como 'False'
print(base_credit.isnull())

# Mostra a quantidade de vezes que o 'True' aparece, ou seja mostra qnts valores faltantes existem
print(base_credit.isnull().sum())

# Mostra onde estão os valores faltantes
print(base_credit.loc[pd.isnull(base_credit['age'])])

# Substitui o valores faltantes pela média dos outros dados que estão na coluna 'age'
base_credit['age'].fillna(base_credit['age'].mean(), inplace = True)
print(base_credit)

# Mostra os valores que anteriormente tinham a coluna 'age' como valor faltante
print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])