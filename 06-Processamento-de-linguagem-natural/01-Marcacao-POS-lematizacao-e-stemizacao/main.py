import bs4 as bs  # Biblioteca BeautifulSoup para análise de HTML e XML (não usada no código)
import urllib.request  # Biblioteca para fazer requisições HTTP (não usada no código)
import nltk  # Biblioteca para processamento de linguagem natural
import spacy  # Biblioteca avançada para NLP

# Carrega o modelo de processamento de linguagem natural para português
pln = spacy.load("pt_core_news_sm")

# Texto de exemplo a ser analisado
documento = pln('Estou aprendendo processamento de linguagem natural, curso em Curitiba.')

# Percorre cada palavra (token) do texto e imprime seu tipo gramatical (verbo, substantivo, etc.)
for token in documento:
    print(token.text, token.pos_)

# Lematização (reduz palavras à sua forma base, ex: "aprendendo" → "aprender")
for token in documento:
    print(token.text, token.lemma_)

# Outro exemplo de lematização com palavras flexionadas
doc = pln('encontrei encontraram encontrarão encontrariam cursando curso cursei')
print([token.lemma_ for token in doc])  # Lista com as palavras na forma base

# Baixa o stemmer do NLTK para português
nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()

# Exemplo de stemização (reduz palavras à raiz, ex: "aprendendo" → "aprend")
print(stemmer.stem('aprender'))

# Imprime cada palavra, sua forma base (lematização) e sua raiz (stemização)
for token in documento:
    print(token.text, token.lemma_, stemmer.stem(token.text))
