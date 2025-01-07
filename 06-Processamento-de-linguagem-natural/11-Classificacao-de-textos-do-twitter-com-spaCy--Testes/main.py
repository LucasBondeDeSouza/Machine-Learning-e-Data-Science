import pandas as pd  # Biblioteca para manipulação de dados tabulares
import string  # Biblioteca para manipulação de strings
import spacy  # Biblioteca para processamento de linguagem natural
import random  # Biblioteca para gerar valores aleatórios
import numpy as np  # Biblioteca para operações matemáticas e manipulação de arrays
import re  # Biblioteca para trabalhar com expressões regulares
from spacy.lang.pt.stop_words import STOP_WORDS  # Palavras irrelevantes que podem ser removidas
from spacy.training import Example  # Para treinar o modelo de NLP com exemplos
import matplotlib.pyplot as plt  # Biblioteca para gerar gráficos
from spacy.util import minibatch  # Para treinar o modelo em pequenos lotes de dados
from sklearn.metrics import accuracy_score, confusion_matrix  # Para avaliar o desempenho do modelo

# Carregar os dados de treinamento e remover colunas desnecessárias
base_treinamento = pd.read_csv('Train50.csv', delimiter=';')
base_treinamento.drop(['id', 'tweet_date', 'query_used'], axis=1, inplace=True)

# Carregar os dados de teste e remover colunas desnecessárias
base_teste = pd.read_csv('Test.csv', delimiter=';')
base_teste.drop(['id', 'tweet_date', 'query_used'], axis=1, inplace=True)

# Carregar o modelo de linguagem do spaCy e definir as palavras irrelevantes (stopwords)
pln = spacy.load('pt_core_news_sm')
stop_words = STOP_WORDS

# Função para limpar e processar o texto
def preprocessamento(texto):
    texto = texto.lower()  # Converter todo o texto para minúsculas
    texto = re.sub(r"@[A-Za-z0-9$-_@.$+]+", ' ', texto)  # Remover menções (@usuario)
    texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)  # Remover links
    texto = re.sub(r" +", ' ', texto)  # Remover espaços em excesso

    # Substituir emojis/textos curtos por palavras que indicam emoção
    lista_emocoes = {':)': 'emocaopositiva', ':d': 'emocaopositiva', ':(': 'emocaonegativa'}
    for emocao in lista_emocoes:
        texto = texto.replace(emocao, lista_emocoes[emocao])

    # Processar o texto com o modelo de NLP
    documento = pln(texto)

    # Criar uma lista apenas com palavras importantes (lematização)
    lista = [token.lemma_ for token in documento if token.lemma_ not in stop_words and token.lemma_ not in string.punctuation]

    return ' '.join([palavra for palavra in lista if not palavra.isdigit()])  # Retornar o texto limpo

# Aplicar o pré-processamento aos dados de treinamento e teste
base_treinamento['tweet_text'] = base_treinamento['tweet_text'].apply(preprocessamento)
base_teste['tweet_text'] = base_teste['tweet_text'].apply(preprocessamento)

# Criar uma estrutura de dados para o treinamento do modelo
base_dados_treinamento_final = []
for texto, emocao in zip(base_treinamento['tweet_text'], base_treinamento['sentiment']):
    dic = {"POSITIVO": emocao == 1, "NEGATIVO": emocao == 0}  # Criar um dicionário para armazenar o sentimento
    base_dados_treinamento_final.append([texto, dic.copy()])  # Adicionar à base de dados

# Criar um modelo NLP vazio do spaCy
modelo = spacy.blank('pt')

# Adicionar um componente de classificação de texto (textcat) ao modelo
if 'textcat' not in modelo.pipe_names:
    textcat = modelo.add_pipe("textcat", last=True)
else:
    textcat = modelo.get_pipe("textcat")

# Definir as categorias que o modelo pode prever
textcat.add_label("POSITIVO")
textcat.add_label("NEGATIVO")

# Carregar um modelo previamente treinado
modelo_carregado = spacy.load('modelo')

# Testar um texto positivo e verificar a previsão do modelo
texto_positivo = base_teste['tweet_text'][21]
previsao = modelo_carregado(texto_positivo)
print(previsao.cats)  # Exibir as probabilidades previstas para cada categoria

# Testar um texto escrito manualmente
texto_positivo = 'eu gosto muito de você'
texto_positivo = preprocessamento(texto_positivo)  # Aplicar pré-processamento
print(modelo_carregado(texto_positivo).cats)  # Exibir a previsão do modelo

# Testar um texto negativo
texto_negativo = base_teste['tweet_text'][4000]
previsao = modelo_carregado(texto_negativo)
print(previsao.cats)  # Exibir a previsão do modelo

# Avaliação do modelo de NLP
previsoes = []
for texto in base_teste['tweet_text']:
    previsao = modelo_carregado(texto)  # Fazer previsão para cada texto de teste
    previsoes.append(previsao.cats)  # Armazenar os resultados

# Converter previsões em rótulos (1 para positivo, 0 para negativo)
previsoes_final = []
for previsao in previsoes:
    if previsao['POSITIVO'] > previsao['NEGATIVO']:
        previsoes_final.append(1)
    else:
        previsoes_final.append(0)

previsoes_final = np.array(previsoes_final)  # Converter a lista em array NumPy

# Obter os rótulos reais do conjunto de teste
respostas_reais = base_teste['sentiment'].values

# Calcular a acurácia do modelo (percentual de respostas corretas)
print(accuracy_score(respostas_reais, previsoes_final))

# Criar uma matriz de confusão para avaliar erros do modelo
cm = confusion_matrix(respostas_reais, previsoes_final)
print(cm)  # Exibir a matriz de confusão