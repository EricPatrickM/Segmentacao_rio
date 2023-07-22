import cv2
import numpy as np
import matplotlib.pyplot as plt

k = cv2.imread('./river4.jpg')
imagem = cv2.cvtColor(k, cv2.COLOR_BGR2HSV).astype("uint16")

h,s,v = cv2.split(imagem)
mapa = h * s

_, binary_image = cv2.threshold(mapa, 50, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


def bwareaopen(img, area_threshold):
    # Verifica se a imagem é binária (escala de cinza)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normaliza a imagem para o intervalo [0, 255]
    img = (img / np.max(img) * 255).astype(np.uint8)

    # Encontra os contornos dos objetos na imagem binária
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Cria uma máscara para armazenar os objetos maiores ou iguais ao limiar de área
    mask = np.zeros_like(img)

    # Percorre os contornos e mantém apenas os objetos com área maior ou igual ao limiar
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= area_threshold:
            print(area)
            cv2.drawContours(mask, [contour], 0, 255, -1)

    return mask

img_limpa = bwareaopen(binary_image, 167000)

plt.figure(figsize=(12, 12))
plt.subplot(231)
plt.imshow(cv2.cvtColor(k, cv2.COLOR_BGR2RGB))
plt.title("Imagem original")
plt.axis('off')

plt.subplot(232)
plt.imshow(imagem, cmap='hsv')
plt.title("Imagem HSV")
plt.axis('off')

plt.subplot(233)
plt.imshow(mapa, cmap='gray')
plt.title("Produto (H x S)")
plt.axis('off')

plt.subplot(234)
plt.imshow(binary_image, cmap='gray')
plt.title("Segmentação HSV")
plt.axis('off')

plt.subplot(235)
plt.imshow(img_limpa, cmap='gray')
plt.title("Segmentação HSV")
plt.axis('off')


plt.show()
