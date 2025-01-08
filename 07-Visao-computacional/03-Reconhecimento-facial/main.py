# Importando as bibliotecas necessárias
import cv2
from PIL import Image
import numpy as np
import zipfile
import os

# TREINAMENTO

# Definindo o caminho do arquivo zip que contém as imagens
path = 'yalefaces.zip'

# Abrindo o arquivo zip que contém as imagens de treinamento
# zip_object = zipfile.ZipFile(file = path, mode = 'r')

# Extraindo o conteúdo do arquivo zip para a pasta atual
# zip_object.extractall('./')

# Fechando o arquivo zip
# zip_object.close()

# Função para carregar as imagens de treinamento
def dados_imagem():
    # Pegando todos os caminhos das imagens da pasta 'train'
    caminhos = [os.path.join('yalefaces/train', f) for f in os.listdir('yalefaces/train')]
    
    faces = []  # Lista para armazenar as imagens
    ids = []    # Lista para armazenar os ids das pessoas nas imagens

    # Para cada caminho de imagem encontrado
    for caminho in caminhos:
        # Abrindo a imagem e convertendo para tons de cinza (escala de cinza)
        imagem = Image.open(caminho).convert('L')
        
        # Convertendo a imagem para um array numpy (para manipulação)
        imagem_np = np.array(imagem, 'uint8')
        
        # Pegando o id da pessoa (extraído do nome do arquivo)
        id = int(os.path.split(caminho)[1].split('.')[0].replace('subject', ''))
        
        # Adicionando o id e a imagem às respectivas listas
        ids.append(id)
        faces.append(imagem_np)
    
    # Retornando os ids e as imagens como arrays numpy
    return np.array(ids), faces

# Chamando a função para carregar os dados de imagem e os ids
ids, faces = dados_imagem()

# Criando um reconhecedor de faces utilizando o algoritmo LBPH (Local Binary Patterns Histograms)
lbph = cv2.face.LBPHFaceRecognizer_create()

# Treinando o modelo com as imagens e seus respectivos ids
lbph.train(faces, ids)

# Salvando o classificador treinado em um arquivo 'classificadorLBPH.yml'
# lbph.write('classificadorLBPH.yml')



# CLASSIFICAÇÃO

# Criando um novo reconhecedor de faces
reconhecedor = cv2.face.LBPHFaceRecognizer_create()

# Carregando o classificador treinado previamente
reconhecedor.read('classificadorLBPH.yml')

# Definindo o caminho da imagem de teste
imagem_teste = 'yalefaces/test/subject01.gif'

# Abrindo a imagem de teste e convertendo para escala de cinza
imagem = Image.open(imagem_teste).convert('L')

# Convertendo a imagem para um array numpy
imagem_np = np.array(imagem, 'uint8')

# Usando o reconhecedor para prever a identidade na imagem de teste
idprevisto, _ = reconhecedor.predict(imagem_np)

# Pegando o id correto a partir do nome do arquivo de teste
idcorreto = int(os.path.split(imagem_teste)[1].split('.')[0].replace('subject', ''))

# Carregando o classificador de faces para detectar faces na imagem
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detectando faces na imagem
faces_detectadas = face_cascade.detectMultiScale(imagem_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Para cada face detectada, exibindo o id previsto e o id correto na imagem
for (x, y, w, h) in faces_detectadas:
    # Adicionando texto à imagem com o id previsto e o id correto
    cv2.putText(imagem_np, 'P: ' + str(idprevisto), (x - 70, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
    cv2.putText(imagem_np, 'C: ' + str(idcorreto), (x - 70, y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

# Exibindo a imagem com as anotações de identificação
cv2.imshow('Imagem', imagem_np)

# Aguardando até pressionar uma tecla para fechar a imagem
cv2.waitKey(0)

# Fechando a janela de exibição
cv2.destroyAllWindows()