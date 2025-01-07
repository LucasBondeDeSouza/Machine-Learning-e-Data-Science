import pandas as pd  # Biblioteca para trabalhar com dados em formato de tabelas
import string  # Biblioteca para manipulação de strings
import spacy  # Biblioteca de processamento de linguagem natural
import random  # Biblioteca para gerar números aleatórios
import numpy as np  # Biblioteca para manipulação de dados numéricos
from spacy.lang.pt.stop_words import STOP_WORDS  # Lista de palavras comuns que não são importantes para análise
from spacy.training import Example  # Para treinar o modelo de NLP com exemplos
import matplotlib.pyplot as plt  # Biblioteca para criar gráficos
from sklearn.metrics import confusion_matrix, accuracy_score  # Para medir o desempenho do modelo

# Carregar a base de dados de treinamento a partir de um arquivo CSV
base_dados = pd.read_csv('base_treinamento.txt', encoding='utf-8')

# Carregar o modelo treinado previamente para classificar emoções no texto
modelo_carregado = spacy.load('modelo')

# Lista para armazenar as previsões feitas pelo modelo
previsoes = []
for texto in base_dados['texto']:
    # Para cada texto na base de dados, faz uma previsão usando o modelo carregado
    previsao = modelo_carregado(texto)
    previsoes.append(previsao.cats)  # Salva a classificação de emoções (alegria ou medo)

# Lista para armazenar as previsões finais com base na comparação das emoções previstas
previsoes_final = []
for previsao in previsoes:
    # Compara as probabilidades de "ALEGRIA" e "MEDO" e escolhe a maior
    if previsao['ALEGRIA'] > previsao['MEDO']:
        previsoes_final.append('alegria')  # Se a probabilidade de alegria for maior, classifica como 'alegria'
    else:
        previsoes_final.append('medo')  # Caso contrário, classifica como 'medo'

# Converte a lista de previsões para um formato numérico com numpy
previsoes_final = np.array(previsoes_final)

# Obtém as respostas reais (emoções) da base de dados para comparar com as previsões
respostas_reais = base_dados['emocao'].values

# Calcula a precisão do modelo comparando as respostas reais com as previsões
print(accuracy_score(respostas_reais, previsoes_final))  # Exibe a precisão do modelo

# Gera uma matriz de confusão para mostrar como o modelo se saiu em cada categoria
cm = confusion_matrix(respostas_reais, previsoes_final)
print(cm)  # Exibe a matriz de confusão