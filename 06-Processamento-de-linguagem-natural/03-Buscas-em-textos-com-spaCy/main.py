import bs4 as bs  # Biblioteca para analisar e extrair dados de HTML e XML
import urllib.request  # Biblioteca para fazer requisições HTTP
import spacy  # Biblioteca avançada para Processamento de Linguagem Natural (PLN)
from spacy.matcher import PhraseMatcher  # Ferramenta do spaCy para procurar padrões de palavras
import webbrowser  # Biblioteca para abrir páginas web no navegador

# Faz uma requisição HTTP à página da Wikipédia sobre Inteligência Artificial
dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')
dados = dados.read()  # Lê o conteúdo da página

# Converte o conteúdo HTML em um formato mais fácil de manipular
dados_html = bs.BeautifulSoup(dados, 'lxml')

# Encontra todos os parágrafos ('p') na página HTML
paragrafos = dados_html.find_all('p')

# Inicializa uma variável vazia para armazenar o texto extraído
conteudo = ''
for p in paragrafos:
    conteudo += p.text  # Adiciona o texto de cada parágrafo ao conteúdo

# Converte todo o texto para letras minúsculas, para facilitar a busca
conteudo = conteudo.lower()

# Cria um modelo do spaCy para o idioma português (sem dados carregados)
pln = spacy.blank('pt')

# Define a palavra que queremos procurar (no caso, "turing")
string = 'turing'
token_pesquisa = pln(string)

# Configura o PhraseMatcher para procurar o token 'turing' no texto
matcher = PhraseMatcher(pln.vocab)
matcher.add('SEARCH', None, token_pesquisa)

# Inicializa uma variável para armazenar o texto de resultado
texto = ''
numero_palavras = 50  # Número de palavras ao redor da palavra encontrada para mostrar no resultado

# Aplica o matcher no conteúdo extraído da página
doc = pln(conteudo)
matches = matcher(doc)  # Encontra todas as ocorrências da palavra 'turing'

# Cria um arquivo HTML para salvar os resultados encontrados
with open("output.html", "w", encoding="utf-8") as f:
    f.write(f'<h1>{string.upper()}</h1>')  # Título com a palavra 'TURING'
    f.write(f"""<p><strong>Resultados encontrados: </strong>{len(matches)}</p>""")  # Mostra a quantidade de ocorrências

    # Para cada ocorrência encontrada, pega as palavras ao redor para mostrar no resultado
    for i in matches:
        inicio = i[1] - numero_palavras  # Define onde começar o trecho a ser exibido
        if inicio < 0:
            inicio = 0  # Não permite que comece antes do início do texto
        texto += str(doc[inicio:i[2] + numero_palavras]).replace(string, f"<mark>{string}</mark>")  # Marca a palavra encontrada
        texto += "<br /><br />"  # Formata o texto para o HTML

    f.write(f"""... {texto} ...""")  # Escreve o conteúdo no arquivo HTML

# Abre o arquivo HTML no navegador
webbrowser.open("output.html")