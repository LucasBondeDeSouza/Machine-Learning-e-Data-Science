import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Lendo o arquivo csv
base_census = pd.read_csv('../census.csv')

x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

# O LabelEncoder converte variáveis categóricas (como strings) em valores numéricos (como inteiros).
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

# Esse processo é repetido para as colunas marital, occupation, relationship, race, sex e country 
# substituindo os valores originais de texto por valores numéricos correspondentes.
x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

# O LabelEncoder atribui números inteiros diferentes para cada categoria presente nas colunas categóricas, o que é útil quando você precisa treinar um modelo de Machine Learning, pois a maioria dos modelos não pode lidar diretamente com variáveis categóricas em formato de texto.
# Se a coluna workclass contiver os valores ['Private', 'Self-emp', 'Private', 'Government'], o LabelEncoder poderá transformar esses valores em números, por exemplo, [0, 1, 0, 2], onde 0, 1 e 2 são atribuídos às categorias.