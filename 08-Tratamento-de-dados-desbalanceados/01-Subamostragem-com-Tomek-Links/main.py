import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.under_sampling import TomekLinks
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carrega o dataset 'census.csv' em um DataFrame
base_census = pd.read_csv('census.csv')

# Separa os atributos (X) e o rótulo (Y)
x_census = base_census.iloc[:, 0:14].values  # Colunas de características
y_census = base_census.iloc[:, 14].values    # Coluna alvo

# Converte atributos categóricos em números inteiros (Label Encoding)
label_enconder_workclass = LabelEncoder()
label_enconder_education = LabelEncoder()
label_enconder_marital = LabelEncoder()
label_enconder_occupation = LabelEncoder()
label_enconder_relationship = LabelEncoder()
label_enconder_race = LabelEncoder()
label_enconder_sex = LabelEncoder()
label_enconder_country = LabelEncoder()

x_census[:, 1] = label_enconder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_enconder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_enconder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_enconder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_enconder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_enconder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_enconder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_enconder_country.fit_transform(x_census[:, 13])

# Aplica a técnica de undersampling (Tomek Links) para balancear os dados
tl = TomekLinks(sampling_strategy='majority')
x_under, y_under = tl.fit_resample(x_census, y_census)

# Converte variáveis categóricas em variáveis binárias (One-Hot Encoding)
onehotencorder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
x_census = onehotencorder.fit_transform(x_census).toarray()

# Divide os dados em treino (85%) e teste (15%)
x_census_treinamento_under, x_census_teste_under, y_census_treinamento_under, y_census_teste_under = train_test_split(x_census, y_census, test_size=0.15, random_state=0)

# Cria e treina um modelo de Random Forest
random_forest_census = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)
random_forest_census.fit(x_census_treinamento_under, y_census_treinamento_under)

# Faz previsões com o modelo treinado
previsoes = random_forest_census.predict(x_census_teste_under)

# Exibe a acurácia do modelo
print(accuracy_score(y_census_teste_under, previsoes))