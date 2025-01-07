import pandas as pd  # Biblioteca para trabalhar com dados em formato de tabelas
import string  # Biblioteca para manipulação de strings
import spacy  # Biblioteca de processamento de linguagem natural
import random  # Biblioteca para gerar números aleatórios
import numpy as np  # Biblioteca para manipulação de dados numéricos
from spacy.lang.pt.stop_words import STOP_WORDS  # Lista de palavras comuns que não são importantes para análise
from spacy.training import Example  # Para treinar o modelo de NLP com exemplos
import matplotlib.pyplot as plt  # Biblioteca para criar gráficos

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

# Exemplo de texto positivo
texto_positivo = 'eu adoro cor dos seu olhos'
texto_positivo = preprocessamento(texto_positivo)  # Pré-processar o texto

# Classificar o texto positivo usando o modelo treinado
previsao = modelo_carregado(texto_positivo)
print(previsao.cats)  # Exibe a previsão de categorias (emoções) para o texto

# Exemplo de texto negativo
texto_negativo = 'estou com medo dele'
# Pré-processar e classificar o texto negativo
previsao = modelo_carregado(preprocessamento(texto_negativo))
print(previsao.cats)  # Exibe a previsão de categorias (emoções) para o texto