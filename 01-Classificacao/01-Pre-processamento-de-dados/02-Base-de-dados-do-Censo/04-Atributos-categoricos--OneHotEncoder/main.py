import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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

# Configurando uma ferramenta que vai converter informações "de texto" em "números"
# Por exemplo, profissões, países, ou estado civil. O computador só entende números.
# - O nome dado a essa ferramenta é 'oneHot'.
# - Escolhemos as colunas de texto que queremos converter (índices 1, 3, 5, 6, 7, 8, 9 e 13).
# - Colunas que já são números (como idade ou horas trabalhadas) serão mantidas como estão.
OneHotEncoder_census = ColumnTransformer(
    transformers = [('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], 
    remainder = 'passthrough' # Mantém as colunas que não precisam de conversão
)

# Aplicando a transformação:
# - As colunas de texto (como profissão ou país) são convertidas em números.
# - Para cada valor de texto, é criada uma nova coluna indicando se a pessoa se encaixa naquele valor.
# - No final, todos os dados, tanto convertidos quanto os que já eram números, são combinados em um único lugar.
# - O resultado final é uma grande tabela só com números.
x_census = OneHotEncoder_census.fit_transform(x_census).toarray()