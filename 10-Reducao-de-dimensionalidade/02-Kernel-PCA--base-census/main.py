import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carrega os dados do arquivo CSV
base_census = pd.read_csv('census.csv')

# Separa os atributos (X) e a classe alvo (Y)
x_census = base_census.iloc[:, 0:14].values  # Características dos indivíduos
y_census = base_census.iloc[:, 14].values    # Rótulo (ex: renda alta ou baixa)

# Converte dados categóricos (texto) em números
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

# Aplica a conversão para transformar textos em números
x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

# Normaliza os valores para padronizar os dados
scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)

# Divide os dados em treino (85%) e teste (15%)
x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(
    x_census, y_census, test_size=0.15, random_state=0
)

# Reduz a dimensionalidade dos dados usando Kernel PCA
kpca = KernelPCA(n_components=8, kernel='rbf')
x_census_treinamento_kpca = kpca.fit_transform(x_census_treinamento)
x_census_teste_kpca = kpca.transform(x_census_teste)

# Cria um modelo de floresta aleatória (Random Forest)
random_forest_census_kpca = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
random_forest_census_kpca.fit(x_census_treinamento_kpca, y_census_treinamento)  # Treina o modelo

# Faz previsões nos dados de teste
previsoes = random_forest_census_kpca.predict(x_census_teste_kpca)

# Exibe a acurácia do modelo e um relatório detalhado
print(accuracy_score(y_census_teste, previsoes))  # Mede a precisão
print(classification_report(y_census_teste, previsoes))  # Mostra métricas detalhadas