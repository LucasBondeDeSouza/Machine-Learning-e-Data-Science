import pandas as pd
import string
import spacy
import random
import seaborn as sns
import numpy as np
from spacy.lang.pt.stop_words import STOP_WORDS

base_dados = pd.read_csv('base_treinamento.txt', encoding='utf-8')

pontuacoes = string.punctuation
stop_words = STOP_WORDS
pln = spacy.load('pt_core_news_sm')

def preprocessamento(texto):
    texto = texto.lower()
    documento = pln(texto)

    lista = []
    for token in documento:
        if token.pos_ == 'PROPN':
            lista.append(token.text)
        else:
            lista.append(token.lemma_)
    
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    
    return lista

base_dados['texto'] = base_dados['texto'].apply(preprocessamento)

exemplo_base_dados = [
    ["este trabalho é agradável", {"ALEGRIA": True, "MEDO": False}],
    ["este lugar continua assustador", {"ALEGRIA": False, "MEDO": True}]
]

base_dados_final = []
for texto, emocao in zip(base_dados['texto'], base_dados['emocao']):
    # print(texto, emocao)
    if emocao == 'alegria':
        dic = ({'ALEGRIA': True, 'MEDO': False})
    elif emocao == 'medo':
        dic = ({'ALEGRIA': False, 'MEDO': True})

    base_dados_final.append([texto, dic.copy()])