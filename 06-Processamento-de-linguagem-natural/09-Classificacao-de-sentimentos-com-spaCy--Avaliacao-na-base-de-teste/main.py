import pandas as pd  # Biblioteca para trabalhar com dados em formato de tabelas
import string  # Biblioteca para manipulação de strings
import spacy  # Biblioteca de processamento de linguagem natural
import random  # Biblioteca para gerar números aleatórios
import numpy as np  # Biblioteca para manipulação de dados numéricos
from spacy.lang.pt.stop_words import STOP_WORDS  # Lista de palavras comuns que não são importantes para análise
from spacy.training import Example  # Para treinar o modelo de NLP com exemplos
import matplotlib.pyplot as plt  # Biblioteca para criar gráficos
from sklearn.metrics import confusion_matrix, accuracy_score  # Para medir o desempenho do modelo

# Carregar a base de dados de treinamento a partir de um arquivo
base_dados = pd.read_csv('base_treinamento.txt', encoding='utf-8')

# Definir pontuações e palavras comuns (stopwords) que não ajudam na análise
pontuacoes = string.punctuation  # Pontuações como ., ?, !, etc.
stop_words = STOP_WORDS  # Palavras comuns como 'a', 'de', 'e', que não ajudam na análise
pln = spacy.load('pt_core_news_sm')  # Carregar o modelo do spaCy para português

# Função de pré-processamento do texto
def preprocessamento(texto):
    texto = texto.lower()  # Coloca todo o texto em minúsculas
    documento = pln(texto)  # Processa o texto com o modelo de linguagem spaCy

    lista = []
    for token in documento:  # Para cada palavra no texto
        if token.pos_ == 'PROPN':  # Se for um nome próprio, mantém a palavra
            lista.append(token.text)
        else:  # Se não for nome próprio, usa a forma base da palavra (lematização)
            lista.append(token.lemma_)

    # Remove palavras comuns e pontuações, além de números
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

    return lista  # Retorna o texto limpo

# Carregar o modelo treinado previamente para classificação de emoções
modelo_carregado = spacy.load('modelo')

# Carregar os dados de teste
base_dados_teste = pd.read_csv('base_teste.txt', encoding = 'utf-8')

# Aplicar o pré-processamento no texto de teste
base_dados_teste['texto'] = base_dados_teste['texto'].apply(preprocessamento)

# Lista para armazenar as previsões do modelo
previsoes = []
for texto in base_dados_teste['texto']:
    previsao = modelo_carregado(texto)  # Faz a previsão para o texto
    previsoes.append(previsao.cats)  # Salva a classificação de emoções

# Definir a classe prevista com base na maior probabilidade (ALEGRIA ou MEDO)
previsoes_final = []
for previsao in previsoes:
    if previsao['ALEGRIA'] > previsao['MEDO']:
        previsoes_final.append('alegria')  # Se a probabilidade de ALEGRIA for maior, classifica como alegria
    else:
        previsoes_final.append('medo')  # Caso contrário, classifica como medo

# Converter a lista de previsões para um formato adequado
previsoes_final = np.array(previsoes_final)

# Respostas reais (emocões verdadeiras) dos dados de teste
respostas_reais = base_dados_teste['emocao'].values

# Calcular a precisão do modelo comparando as previsões com as respostas reais
print(accuracy_score(respostas_reais, previsoes_final))  # Exibe a acurácia do modelo

# Gerar e exibir a matriz de confusão para ver como o modelo classificou as emoções
cm = confusion_matrix(respostas_reais, previsoes_final)
print(cm)  # Exibe a matriz de confusão