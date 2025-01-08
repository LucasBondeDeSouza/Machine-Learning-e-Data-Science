import cv2  # Importa a biblioteca OpenCV para processamento de imagens

# Carrega a imagem do arquivo
imagem = cv2.imread('pessoas.jpg')

# Carrega o classificador para detectar corpos inteiros
detector_corpo = cv2.CascadeClassifier('fullbody.xml')

# Converte a imagem para tons de cinza (melhora a detecção)
image_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detecta corpos na imagem (ajustando fator de escala e tamanho mínimo)
detections = detector_corpo.detectMultiScale(image_gray, scaleFactor=1.1, minSize=(50, 50))

# Desenha um retângulo verde ao redor de cada corpo detectado
for (x, y, l, a) in detections:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

# Exibe a imagem com as detecções
cv2.imshow('Imagem', imagem)
cv2.waitKey(0)  # Aguarda uma tecla para fechar a janela
cv2.destroyAllWindows()  # Fecha todas as janelas abertas