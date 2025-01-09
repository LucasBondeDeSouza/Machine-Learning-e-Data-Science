import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Carrega os dados do arquivo 'census.csv'
base_census = pd.read_csv('census.csv')

# Obtém os nomes das colunas, exceto a última (alvo)
colunas = base_census.columns[:-1]

# Separa atributos (X) e classe alvo (Y)
x_census = base_census.iloc[:, 0:14].values  
y_census = base_census.iloc[:, 14].values    

# Instancia codificadores para converter texto em números
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

# Aplica a codificação às colunas categóricas
x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

# Normaliza os dados para ficar entre 0 e 1
scaler = MinMaxScaler()
x_census_scaler = scaler.fit_transform(x_census)

# Exibe a variância de cada coluna após a normalização
for i in range(x_census.shape[1]):
    print(x_census_scaler[:, i].var())

# Remove colunas com baixa variância (pouca informação relevante)
selecao = VarianceThreshold(threshold=0.05)
x_census_variancia = selecao.fit_transform(x_census_scaler)

# Obtém os índices das colunas com variância acima do limite
indices = np.where(selecao.variances_ > 0.05)

# Remove colunas irrelevantes para melhorar o desempenho
base_census_variancia = base_census.drop(columns=['age', 'workclass', 'final-weight', 'education-num', 'race', 
                                                  'capital-gain', 'capital-loos', 'hour-per-week', 'native-country'], axis=1)

# Separa atributos e classe alvo após a filtragem
x_census_variancia = base_census_variancia.iloc[:, 0:5].values
y_census_variancia = base_census_variancia.iloc[:, 5].values

# Aplica a codificação às colunas categóricas restantes
x_census_variancia[:, 0] = label_encoder_education.fit_transform(x_census_variancia[:, 0])
x_census_variancia[:, 1] = label_encoder_marital.fit_transform(x_census_variancia[:, 1])
x_census_variancia[:, 2] = label_encoder_occupation.fit_transform(x_census_variancia[:, 2])
x_census_variancia[:, 3] = label_encoder_relationship.fit_transform(x_census_variancia[:, 3])
x_census_variancia[:, 4] = label_encoder_sex.fit_transform(x_census_variancia[:, 4])

# Aplica One-Hot Encoding para representar categorias como vetores binários
onehotencoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 1, 2, 3, 4])], remainder='passthrough')
x_census_variancia = onehotencoder.fit_transform(x_census_variancia).toarray()

# Normaliza os dados novamente após a transformação
scaler = MinMaxScaler()
x_census_variancia = scaler.fit_transform(x_census_variancia)

# Divide os dados em treino (85%) e teste (15%)
x_census_treinamento_var, x_census_teste_var, y_census_treinamento_var, y_census_teste_var = train_test_split(
    x_census_variancia, y_census_variancia, test_size=0.15, random_state=0)

# Cria um modelo Random Forest para classificação
random_forest_var = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)

# Treina o modelo com os dados de treinamento
random_forest_var.fit(x_census_treinamento_var, y_census_treinamento_var)

# Faz previsões nos dados de teste
previsoes = random_forest_var.predict(x_census_teste_var)

# Exibe a precisão e o relatório de classificação do modelo
print(accuracy_score(y_census_teste_var, previsoes))
print(classification_report(y_census_teste_var, previsoes))