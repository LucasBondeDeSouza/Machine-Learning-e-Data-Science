import bs4 as bs  # Biblioteca BeautifulSoup para analisar e extrair dados de HTML e XML
import urllib.request  # Biblioteca para fazer requisições HTTP

# Faz uma requisição à página da Wikipédia sobre Inteligência Artificial
dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')
dados = dados.read()  # Lê o conteúdo da página

# Converte o conteúdo HTML para um formato mais fácil de manipular
dados_html = bs.BeautifulSoup(dados, 'lxml')

# Encontra todos os parágrafos ('p') na página
paragrafos = dados_html.find_all('p')

# Inicializa uma variável para armazenar o texto dos parágrafos
conteudo = ''
for p in paragrafos:
    conteudo += p.text  # Adiciona o texto de cada parágrafo ao conteúdo

# Converte todo o texto para letras minúsculas
conteudo = conteudo.lower()