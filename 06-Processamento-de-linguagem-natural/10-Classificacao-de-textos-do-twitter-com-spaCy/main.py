import pandas as pd  # Trabalhar com tabelas de dados
import string  # Manipulação de texto e pontuação
import spacy  # Processamento de linguagem natural (PLN)
import random  # Gerar números aleatórios
import numpy as np  # Operações matemáticas e estatísticas
import re  # Expressões regulares para manipulação de texto
from spacy.lang.pt.stop_words import STOP_WORDS  # Palavras irrelevantes na análise de texto
from spacy.training import Example  # Criar exemplos para treinar o modelo
import matplotlib.pyplot as plt  # Criar gráficos
from spacy.util import minibatch  # Dividir dados em pequenos lotes para treino

# Carregar os dados de treinamento e teste
base_treinamento = pd.read_csv('Train50.csv', delimiter=';')
base_treinamento.drop(['id', 'tweet_date', 'query_used'], axis=1, inplace=True)  # Remover colunas desnecessárias

base_teste = pd.read_csv('Test.csv', delimiter=';')
base_teste.drop(['id', 'tweet_date', 'query_used'], axis=1, inplace=True)

# Carregar modelo de PLN e definir palavras irrelevantes (stopwords)
pln = spacy.load('pt_core_news_sm')
stop_words = STOP_WORDS

# Função para limpar e preparar o texto
def preprocessamento(texto):
    texto = texto.lower()  # Converter para minúsculas
    texto = re.sub(r"@[A-Za-z0-9$-_@.$+]+", ' ', texto)  # Remover menções (@usuario)
    texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)  # Remover links
    texto = re.sub(r" +", ' ', texto)  # Remover espaços extras

    # Substituir emojis por palavras representando emoções
    lista_emocoes = {':)': 'emocaopositiva', ':d': 'emocaopositiva', ':(': 'emocaonegativa'}
    for emocao in lista_emocoes:
        texto = texto.replace(emocao, lista_emocoes[emocao])

    # Processar o texto com spaCy para remover palavras irrelevantes
    documento = pln(texto)
    lista = [token.lemma_ for token in documento if token.lemma_ not in stop_words and token.lemma_ not in string.punctuation]
    
    return ' '.join([palavra for palavra in lista if not palavra.isdigit()])  # Retornar texto limpo

# Aplicar o pré-processamento nos tweets
base_treinamento['tweet_text'] = base_treinamento['tweet_text'].apply(preprocessamento)
base_teste['tweet_text'] = base_teste['tweet_text'].apply(preprocessamento)

# Criar estrutura de dados para treinamento do modelo
base_dados_treinamento_final = []
for texto, emocao in zip(base_treinamento['tweet_text'], base_treinamento['sentiment']):
    dic = {"POSITIVO": emocao == 1, "NEGATIVO": emocao == 0}  # Definir rótulo positivo ou negativo
    base_dados_treinamento_final.append([texto, dic.copy()])  # Adicionar à lista de treinamento

# Criar um novo modelo de PLN do zero
modelo = spacy.blank('pt')  # Modelo em branco para português

# Adicionar um classificador de texto ao modelo
if 'textcat' not in modelo.pipe_names:
    textcat = modelo.add_pipe("textcat", last=True)  # Criar classificador de texto
else:
    textcat = modelo.get_pipe("textcat")

# Definir as categorias que o modelo vai identificar
textcat.add_label("POSITIVO")
textcat.add_label("NEGATIVO")

# Inicializar o treinamento do modelo
modelo.initialize()
historico = []

# Treinar o modelo por 3 épocas (iterações sobre os dados)
for epoca in range(3):
    random.shuffle(base_dados_treinamento_final)  # Misturar os dados para evitar padrões repetitivos
    losses = {}  # Armazenar perdas (erros)

    batches = minibatch(base_dados_treinamento_final, size=512)  # Dividir os dados em lotes pequenos
    for batch in batches:
        examples = [Example.from_dict(modelo.make_doc(texto), {"cats": entities}) for texto, entities in batch]
        modelo.update(examples, losses=losses)  # Atualizar o modelo com novos exemplos
        historico.append(losses)  # Salvar histórico de perdas

# Criar uma lista com os valores do erro ao longo do tempo
historico_loss = [i.get('textcat') for i in historico]
historico_loss = np.array(historico_loss)

# Gerar um gráfico para visualizar a progressão do erro durante o treinamento
plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.show()

# Salvar o modelo treinado para uso futuro
modelo.to_disk('modelo')
