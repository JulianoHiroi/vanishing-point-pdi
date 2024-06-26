import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem em escala de cinza
img = cv2.imread('sua_imagem.jpg', cv2.IMREAD_GRAYSCALE)

# Definir os valores de minVal e maxVal
minVal = 50
maxVal = 150

# Aplicar o detector de bordas de Canny
edges = cv2.Canny(img, minVal, maxVal)

# Verificar a forma (shape) da imagem original e da imagem resultante
print(f"Forma da imagem original: {img.shape}")
print(f"Forma da imagem com bordas detectadas: {edges.shape}")

# Exibir a imagem original e a imagem com as bordas detectadas
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Bordas Detectadas')
plt.axis('off')

plt.tight_layout()
plt.show()