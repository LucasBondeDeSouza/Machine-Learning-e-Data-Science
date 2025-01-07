import pandas as pd  # Manipulação de dados
import string  # Trabalhar com pontuações
import spacy  # Processamento de Linguagem Natural
import random  # Geração de números aleatórios
import numpy as np  # Operações matemáticas
from spacy.lang.pt.stop_words import STOP_WORDS  # Lista de palavras irrelevantes (stopwords)
from spacy.training import Example  # Criar exemplos para treino do modelo
import matplotlib.pyplot as plt  # Criar gráficos

# Carregar a base de dados de treinamento
base_dados = pd.read_csv('base_treinamento.txt', encoding='utf-8')

# Definir pontuações e palavras irrelevantes
pontuacoes = string.punctuation
stop_words = STOP_WORDS
pln = spacy.load('pt_core_news_sm')  # Carrega modelo de PLN em português

# Função para pré-processar o texto (limpar e transformar)
def preprocessamento(texto):
    texto = texto.lower()  # Converte para minúsculas
    documento = pln(texto)  # Processa o texto com o modelo spaCy

    lista = []
    for token in documento:
        if token.pos_ == 'PROPN':  # Se for um nome próprio, mantém como está
            lista.append(token.text)
        else:
            lista.append(token.lemma_)  # Caso contrário, converte para a forma base da palavra
    
    # Remove stopwords e pontuações
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
    
    # Remove números e junta as palavras novamente em uma string
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    
    return lista  # Retorna o texto limpo

# Aplica o pré-processamento a toda a base de dados
base_dados['texto'] = base_dados['texto'].apply(preprocessamento)

# Formatar os dados para treino do modelo
base_dados_final = []
for texto, emocao in zip(base_dados['texto'], base_dados['emocao']):
    if emocao == 'alegria':
        dic = {'ALEGRIA': 1.0, 'MEDO': 0.0}  # Probabilidade para cada emoção
    elif emocao == 'medo':
        dic = {'ALEGRIA': 0.0, 'MEDO': 1.0}
    
    base_dados_final.append((texto, dic.copy()))  # Adiciona à lista de treinamento

# Criar um modelo spaCy vazio para português
modelo = spacy.blank('pt')

# Adicionar um classificador de texto ao modelo
if 'textcat' not in modelo.pipe_names:
    categorias = modelo.add_pipe('textcat', last=True)  # Adiciona o classificador
else:
    categorias = modelo.get_pipe('textcat')

# Definir as categorias do classificador (emoções)
categorias.add_label('ALEGRIA')
categorias.add_label('MEDO')

# Lista para armazenar a evolução do erro
historico = []

# Inicia o treinamento do modelo
optimizer = modelo.begin_training()

for epoca in range(1000):  # Treinar por 1000 épocas
    random.shuffle(base_dados_final)  # Embaralha os dados para melhor aprendizado
    losses = {}  # Dicionário para armazenar as perdas (erros)

    for batch in spacy.util.minibatch(base_dados_final, size=30):  # Divide os dados em pequenos lotes
        examples = []
        for texto, entities in batch:
            doc = modelo.make_doc(texto)  # Transforma o texto em um documento spaCy
            example = Example.from_dict(doc, {"cats": entities})  # Cria o formato correto de exemplo
            examples.append(example)

        modelo.update(examples, losses=losses, sgd=optimizer)  # Atualiza o modelo com os exemplos

    if epoca % 100 == 0:  # A cada 100 épocas, armazena o erro
        historico.append(losses)

# Converter os erros em um array numpy para análise
historico_loss = []
for i in historico:
    historico_loss.append(i.get('textcat'))

historico_loss = np.array(historico_loss)

# Criar gráfico mostrando a evolução do erro ao longo das épocas
plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.show()

# Salvar o modelo treinado para uso futuro
modelo.to_disk("modelo")