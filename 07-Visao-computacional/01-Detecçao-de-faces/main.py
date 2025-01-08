import cv2  # Importa a biblioteca OpenCV para processamento de imagens

# Carrega a imagem do arquivo
imagem = cv2.imread('workplace-1245776_1920.jpg')

# Carrega o classificador pré-treinado para detecção de rostos
detector_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Converte a imagem para escala de cinza (melhora a detecção)
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detecta rostos na imagem (parâmetros ajustam a precisão da detecção)
deteccoes = detector_face.detectMultiScale(imagem_cinza, scaleFactor=1.3, minSize=(30, 30))

# Desenha retângulos ao redor dos rostos detectados
for (x, y, l, a) in deteccoes:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)  # Verde com borda de 2px

# Exibe a imagem com os rostos detectados
cv2.imshow('Imagem', imagem)
cv2.waitKey(0)  # Aguarda o usuário pressionar uma tecla para fechar a janela
cv2.destroyAllWindows()  # Fecha todas as janelas abertas pelo OpenCV