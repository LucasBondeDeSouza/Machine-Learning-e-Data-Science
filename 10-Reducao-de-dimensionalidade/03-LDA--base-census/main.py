import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carrega os dados do arquivo CSV
base_census = pd.read_csv('census.csv')

# Separa os atributos (X) e o alvo (Y)
x_census = base_census.iloc[:, 0:14].values  # Dados das pessoas (ex: idade, escolaridade)
y_census = base_census.iloc[:, 14].values    # Classe alvo (ex: renda alta ou baixa)

# Converte dados de texto para números
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

# Aplica a conversão em colunas categóricas (transforma textos em números)
x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

# Padroniza os dados para que fiquem na mesma escala
scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)

# Divide os dados em treino (85%) e teste (15%)
x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(
    x_census, y_census, test_size=0.15, random_state=0
)

# Define o número de classes e características
num_classes = len(set(y_census))  # Quantidade de categorias no alvo
num_features = x_census.shape[1]  # Número de colunas em X

# Aplica LDA para reduzir a dimensionalidade dos dados
lda = LinearDiscriminantAnalysis(n_components=min(num_features, num_classes - 1))
x_census_treinamento_lda = lda.fit_transform(x_census_treinamento, y_census_treinamento)
x_census_teste_lda = lda.transform(x_census_teste)

# Cria o modelo de Random Forest para fazer previsões
random_forest_census_lda = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
random_forest_census_lda.fit(x_census_treinamento_lda, y_census_treinamento)

# Faz previsões e avalia o modelo
previsoes = random_forest_census_lda.predict(x_census_teste_lda)
print(accuracy_score(y_census_teste, previsoes))  # Exibe a acurácia do modelo
print(classification_report(y_census_teste, previsoes))  # Mostra métricas detalhadas do modelo