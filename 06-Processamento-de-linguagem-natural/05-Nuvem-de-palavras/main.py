import bs4 as bs  # Importa a biblioteca BeautifulSoup para analisar HTML e extrair dados
import urllib.request  # Biblioteca para fazer requisições HTTP
import spacy  # Biblioteca para processamento de linguagem natural (PLN)
from matplotlib.colors import ListedColormap  # Para escolher uma paleta de cores para a nuvem de palavras
from wordcloud import WordCloud  # Biblioteca para gerar nuvem de palavras
import matplotlib.pyplot as plt  # Biblioteca para gerar gráficos e exibir imagens
from spacy.lang.pt.stop_words import STOP_WORDS  # Lista de palavras comuns em português que normalmente são ignoradas

# Faz uma requisição HTTP à página da Wikipédia sobre Inteligência Artificial
dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')
dados = dados.read()  # Lê o conteúdo da página HTML

# Converte o conteúdo HTML em um formato fácil de manipular com BeautifulSoup
dados_html = bs.BeautifulSoup(dados, 'lxml')

# Encontra todos os parágrafos ('p') na página HTML
paragrafos = dados_html.find_all('p')

# Cria uma variável vazia para armazenar o conteúdo dos parágrafos
conteudo = ''
for p in paragrafos:
    conteudo += p.text  # Adiciona o texto de cada parágrafo ao conteúdo

# Converte o texto para minúsculas para facilitar o processamento posterior
conteudo = conteudo.lower()

# Carrega o modelo de linguagem do spaCy para o português
pln = spacy.load('pt_core_news_sm')

# Processa o texto usando o modelo do spaCy para extrair as palavras
doc = pln(conteudo)
lista_token = []
for token in doc:
    lista_token.append(token.text)  # Adiciona cada palavra ao lista_token

# Remove as palavras comuns (como "e", "de", "a") para focar nas palavras mais importantes
sem_stop = []
for palavra in lista_token:
    if pln.vocab[palavra].is_stop == False:  # Verifica se a palavra não é uma palavra comum
        sem_stop.append(palavra)

# Define as cores da nuvem de palavras
color_map = ListedColormap(['orange', 'green', 'red', 'magenta'])

# Gera a nuvem de palavras a partir das palavras restantes
cloud = WordCloud(background_color='white', max_words=100, colormap=color_map)

# Junta as palavras restantes em uma única string e gera a nuvem de palavras
cloud = cloud.generate(' '.join(sem_stop))

# Exibe a nuvem de palavras
plt.figure(figsize=(15, 15))  # Define o tamanho da imagem
plt.imshow(cloud)  # Mostra a imagem gerada
plt.axis('off')  # Remove os eixos
plt.show()  # Exibe a imagem