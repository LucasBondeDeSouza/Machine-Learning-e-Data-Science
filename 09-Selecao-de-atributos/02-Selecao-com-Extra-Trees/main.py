import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carrega os dados do arquivo CSV
base_census = pd.read_csv('census.csv')

# Obtém os nomes das colunas, exceto a última (alvo)
colunas = base_census.columns[:-1]

# Separa os atributos (X) e o alvo (Y)
x_census = base_census.iloc[:, 0:14].values  # Dados de entrada
y_census = base_census.iloc[:, 14].values    # Classe alvo (rótulos)

# Instancia codificadores para transformar texto em números
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

# Aplica a transformação nas colunas categóricas
x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

# Normaliza os valores para um intervalo entre 0 e 1
scaler = MinMaxScaler()
x_census_scaler = scaler.fit_transform(x_census)

# Seleciona as características mais importantes usando ExtraTreesClassifier
selecao = ExtraTreesClassifier()
selecao.fit(x_census_scaler, y_census)

# Obtém a importância de cada característica
importancias = selecao.feature_importances_

# Seleciona apenas as colunas com importância maior ou igual a 0.029
indices = []
for i in range(len(importancias)):
    if importancias[i] >= 0.029:
        indices.append(i)

# Filtra os dados com base nos índices selecionados
x_census_extra = x_census[:, indices]

# Aplica One-Hot Encoding às colunas categóricas selecionadas
onehotencoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7])], remainder='passthrough')
x_census_extra = onehotencoder.fit_transform(x_census_extra).toarray()

# Divide os dados em conjunto de treino (85%) e teste (15%)
x_census_treinamento_extra, x_census_teste_extra, y_census_treinamento_extra, y_census_teste_extra = train_test_split(
    x_census_extra, y_census, test_size=0.15, random_state=0)

# Cria e treina um modelo Random Forest
random_forest_extra = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)
random_forest_extra.fit(x_census_treinamento_extra, y_census_treinamento_extra)

# Faz previsões no conjunto de teste
previsoes = random_forest_extra.predict(x_census_teste_extra)

# Exibe a precisão e relatório de classificação do modelo
print(accuracy_score(y_census_teste_extra, previsoes))
print(classification_report(y_census_teste_extra, previsoes))