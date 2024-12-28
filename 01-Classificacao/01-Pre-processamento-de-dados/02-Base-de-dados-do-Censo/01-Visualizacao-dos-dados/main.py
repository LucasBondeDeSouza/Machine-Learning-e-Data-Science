import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()

# Lendo o arquivo csv
base_census = pd.read_csv('census.csv')

# Mostra os valores únicos da coluna 'income' do CSV e quantas vezes cada um aparece
print(np.unique(base_census['income'], return_counts = True))


# Faz a contagem de quantos registros existem na coluna 'income' e gera um gráfico
sns.countplot(x = base_census['income'])
plt.show()


# Cria um histograma que mostra a distribuição de frequência dos valores numéricos presentes em 'age'
plt.hist(x = base_census['age'])
plt.show()


# Cria um histograma que mostra a distribuição de frequência dos valores numéricos presentes em 'education-num'
plt.hist(x = base_census['education-num'])
plt.show()


# Cria um histograma que mostra a distribuição de frequência dos valores numéricos presentes em 'hour-per-week'
plt.hist(x = base_census['hour-per-week'])
plt.show()


# Cria um gráfico que organiza os dados em uma hierarquia com base nas colunas 'workclass' (como primeiro nível) e 'age' (como subnível)
# Cada caixa no gráfico representa uma categoria ou valor e seu tamanho reflete a contagem de ocorrências ou outro critério, caso especificado.
grafico = px.treemap(base_census, path = ['workclass', 'age'])
grafico.show()


# Cria um gráfico que organiza os dados em uma hierarquia com base nas colunas 'occupation' (como primeiro nível), 'relationship' e 'age' (como subnível)
# Cada caixa no gráfico representa uma categoria ou valor e seu tamanho reflete a contagem de ocorrências ou outro critério, caso especificado.
grafico = px.treemap(base_census, path = ['occupation', 'relationship', 'age'])
grafico.show()


# Cria um gráfico que  permite visualizar a relação entre as colunas 'occupation' e 'relationship'
# Cada linha no gráfico representa uma combinação única das categorias das duas variáveis e como elas se conectam.
grafico = px.parallel_categories(base_census, dimensions = ['occupation', 'relationship'])
grafico.show()

grafico = px.parallel_categories(base_census, dimensions = ['workclass', 'occupation', 'income'])
grafico.show()

grafico = px.parallel_categories(base_census, dimensions = ['education', 'income'])
grafico.show()