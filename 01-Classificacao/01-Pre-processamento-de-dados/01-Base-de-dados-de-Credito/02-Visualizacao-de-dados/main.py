import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Lendo o arquivo csv
base_credit = pd.read_csv('../credit_data.csv')

# Mostra os valores únicos da coluna 'default' do CSV
print(np.unique(base_credit['default']))

# Mostra os valores únicos da coluna 'default' do CSV e quantas vezes cada um aparece
print(np.unique(base_credit['default'], return_counts=True))

# Faz a contagem de quantos registros existem em cada uma das classes e gerar um gráfico
print(sns.countplot(x = base_credit['default']))

# Faz a separação dos dados em intervalos de idade
plt.hist(x = base_credit['age'])
plt.show()

# Faz a separação dos dados em intervalos de renda
plt.hist(x = base_credit['income'])
plt.show()

# Faz a separação dos dados em intervalos de dívida
plt.hist(x = base_credit['loan'])
plt.show()

# Essa linha cria um gráfico onde você pode ver como essas informações se relacionam umas com as outras. Além disso, os pontos no desenho têm cores diferentes dependendo de quem está com dívidas ou não
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
grafico.show()