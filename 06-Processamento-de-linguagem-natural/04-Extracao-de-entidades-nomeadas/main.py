import bs4 as bs  # Biblioteca para analisar e extrair dados de HTML e XML
import urllib.request  # Biblioteca para fazer requisições HTTP (pegar informações da internet)
import spacy  # Biblioteca avançada para Processamento de Linguagem Natural (PLN)
from spacy.matcher import PhraseMatcher  # Ferramenta do spaCy para procurar padrões de palavras
import webbrowser  # Biblioteca para abrir páginas web no navegador
from spacy import displacy  # Biblioteca para visualização de dados do spaCy

# Faz uma requisição HTTP à página da Wikipédia sobre Inteligência Artificial
dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')
dados = dados.read()  # Lê o conteúdo da página (o código HTML)

# Converte o conteúdo HTML em um formato mais fácil de manipular (usando BeautifulSoup)
dados_html = bs.BeautifulSoup(dados, 'lxml')

# Encontra todos os parágrafos ('p') da página HTML
paragrafos = dados_html.find_all('p')

# Cria uma variável para armazenar todo o texto dos parágrafos
conteudo = ''
for p in paragrafos:
    conteudo += p.text  # Adiciona o texto de cada parágrafo à variável 'conteudo'

# Converte todo o texto para letras minúsculas para facilitar a busca
conteudo = conteudo.lower()

# Cria um modelo do spaCy para o idioma português (sem dados carregados)
pln = spacy.load('pt_core_news_sm')

# Define a palavra que queremos procurar na página (no caso, "turing")
string = 'turing'
token_pesquisa = pln(string)

# Configura o PhraseMatcher para procurar a palavra 'turing' no texto
matcher = PhraseMatcher(pln.vocab)
matcher.add('SEARCH', None, token_pesquisa)

# Variável para armazenar o texto dos resultados
texto = ''
numero_palavras = 50  # Define quantas palavras ao redor da palavra encontrada queremos mostrar

# Aplica o PhraseMatcher ao conteúdo da página (agora em português)
doc = pln(conteudo)
matches = matcher(doc)  # Encontra todas as ocorrências da palavra 'turing' no texto

# Cria um arquivo HTML para salvar os resultados encontrados
with open("output.html", "w", encoding="utf-8") as f:
    f.write(f'<h1>{string.upper()}</h1>')  # Escreve o título "TURING"
    f.write(f"""<p><strong>Resultados encontrados: </strong>{len(matches)}</p>""")  # Mostra quantas vezes 'turing' apareceu

    # Para cada ocorrência de 'turing', pega o texto ao redor e exibe no arquivo HTML
    for i in matches:
        inicio = i[1] - numero_palavras  # Define onde começa o trecho de texto a ser mostrado
        if inicio < 0:
            inicio = 0  # Não deixa começar antes do início do texto
        texto += str(doc[inicio:i[2] + numero_palavras]).replace(string, f"<mark>{string}</mark>")  # Marca a palavra encontrada
        texto += "<br /><br />"  # Formata o texto para HTML

    f.write(f"""... {texto} ...""")  # Escreve o conteúdo no arquivo HTML

# Abre o arquivo HTML no navegador para mostrar os resultados
webbrowser.open("output.html")

# Exibe as entidades encontradas no texto, como nomes de pessoas ou organizações
for entidade in doc.ents:
    print(entidade.text, entidade.label_)

# Exibe uma visualização das entidades encontradas no texto usando o displacy
displacy.render(doc, style='ent', jupyter=True)