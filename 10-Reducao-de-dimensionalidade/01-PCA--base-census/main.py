import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carrega os dados do arquivo CSV para análise
base_census = pd.read_csv('census.csv')

# Separa os dados em atributos (X) e o que queremos prever (Y)
x_census = base_census.iloc[:, 0:14].values  # Características dos indivíduos
y_census = base_census.iloc[:, 14].values    # Classe alvo (exemplo: renda alta ou baixa)

# Converte colunas com textos para números, pois modelos não trabalham com strings
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

# Aplica a conversão nas colunas categóricas específicas
x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

# Normaliza os dados para manter todas as variáveis na mesma escala
scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)

# Divide os dados em treino (85%) e teste (15%) para avaliar o modelo
x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(
    x_census, y_census, test_size=0.15, random_state=0
)

# Aplica PCA para reduzir a quantidade de variáveis mantendo a informação essencial
pca = PCA(n_components=8)  # Reduz de 14 para 8 variáveis principais

# Transforma os dados de treinamento e teste com PCA
x_census_treinamento_pca = pca.fit_transform(x_census_treinamento)
x_census_teste_pca = pca.transform(x_census_teste)

# Exibe a importância de cada nova variável gerada pelo PCA
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())  # Soma da variância explicada pelas 8 componentes

# Cria um modelo de floresta aleatória para prever a classe alvo
random_forest_census_pca = RandomForestClassifier(n_estimators=40, random_state=0)
random_forest_census_pca.fit(x_census_treinamento_pca, y_census_treinamento)

# Faz previsões no conjunto de teste
previsoes = random_forest_census_pca.predict(x_census_teste_pca)

# Exibe a precisão do modelo e um relatório detalhado do desempenho
print(accuracy_score(y_census_teste, previsoes))
print(classification_report(y_census_teste, previsoes))