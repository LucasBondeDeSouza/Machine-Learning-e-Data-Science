# Importando bibliotecas necessárias
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregando o arquivo CSV com os dados do censo
base_census = pd.read_csv('census.csv')

# Separando as variáveis independentes (X) e dependentes (y)
x_census = base_census.iloc[:, 0:14].values  # Características
y_census = base_census.iloc[:, 14].values   # Classe (salário >50K ou <=50K)

# Codificando variáveis categóricas para números
label_enconder_workclass = LabelEncoder()      # Para 'workclass' (classe de trabalho)
label_enconder_education = LabelEncoder()      # Para 'education' (educação)
label_enconder_marital = LabelEncoder()        # Para 'marital' (estado civil)
label_enconder_occupation = LabelEncoder()     # Para 'occupation' (ocupação)
label_enconder_relationship = LabelEncoder()   # Para 'relationship' (relacionamento)
label_enconder_race = LabelEncoder()           # Para 'race' (raça)
label_enconder_sex = LabelEncoder()            # Para 'sex' (sexo)
label_enconder_country = LabelEncoder()        # Para 'country' (país)

# Aplicando a codificação nas colunas apropriadas
x_census[:, 1] = label_enconder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_enconder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_enconder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_enconder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_enconder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_enconder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_enconder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_enconder_country.fit_transform(x_census[:, 13])

# Aplicando SMOTE para balancear os dados (aumentando as amostras da classe minoritária)
smote = SMOTE(sampling_strategy='minority') 
x_over, y_over = smote.fit_resample(x_census, y_census)

# Dividindo os dados em conjuntos de treinamento e teste (85% para treinamento e 15% para teste)
x_census_treinamento_over, x_census_teste_over, y_census_treinamento_over, y_census_teste_over = train_test_split(x_over, y_over, test_size=0.15, random_state=0)

# Criando um modelo de Random Forest (uma técnica de aprendizado de máquina) para prever o salário
random_forest_census = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=100)

# Treinando o modelo com os dados de treinamento
random_forest_census.fit(x_census_treinamento_over, y_census_treinamento_over)

# Fazendo previsões com o modelo treinado nos dados de teste
previsoes = random_forest_census.predict(x_census_teste_over)

# Avaliando a precisão do modelo com base nas previsões e os resultados reais
print(accuracy_score(y_census_teste_over, previsoes))  # Exibe a precisão do modelo
print(classification_report(y_census_teste_over, previsoes))  # Exibe um relatório completo de avaliação