import cv2
import numpy as np
import random
import os

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    area = gray.shape[0] * gray.shape[1]
    print(area)
    print(np.sum(gray))
    blur = cv2.GaussianBlur(gray, (0, 0), 1)
    melhor = 1000
    melhor_i = 0
    melhor_j = 0
    for i in range(100 , 501 , 50):
        for j in range(100 , 201 , 50):
            if(i <= j):
                continue
            edges = cv2.Canny(blur, j, i, apertureSize=3 , L2gradient=False)
            soma_edge  = np.sum(edges)/255
            print("A imagem" + str(j) + "_" + str(i) + "tem porcentagem de borda : " + str(soma_edge/area))
            distancia = np.abs((soma_edge)/area - 0.04)
            print("distancia: " + str(distancia)   )
            if(melhor > distancia):
                melhor = distancia
                melhor_i = i
                melhor_j = j
                img_edges = edges
            #cv2.imwrite('resultados2/image_edges_' + str(j) + "_" + str(i) + '.jpg' , edges)
    cv2.imwrite('resultados2/melhor_image_edges_' + str(melhor_j) + "_" + str(melhor_i) + '.jpg' , img_edges)
    return img_edges

def detect_lines(edges):


    # Fazer uma função que detecta as linhas da imagem
    # 


    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150 , None, 0, 0)
    # faça um assert para garantir que tenha mais de 2 linhas e manda msg se não tiver
    assert lines is not None, "Não foi possível detectar linhas na imagem"
    return lines

def compute_intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])

    det_A = np.linalg.det(A)
    if det_A != 0:
        intersection = np.linalg.solve(A, b)
        return intersection
    else:
        return None

def distance_point_to_line(point, equation_line):
    rho, theta = equation_line[0]
    x0, y0 = point

    a = np.cos(theta)
    b = np.sin(theta)

    return np.abs(a * x0 + b * y0 - rho) / np.sqrt(a**2 + b**2)



def ransac_vanishing_point(lines, num_iterations, threshold , width , restrictAreaFunc = None, restrict = False):

    best_intersection = None
    best_count = 0

    for _ in range(num_iterations):
        line1, line2 = random.sample(list(lines), 2)
        if restrict == False:
            intersection = compute_intersection(line1, line2)
        else: 
            intersection = compute_intersection(line1, line2)
            if intersection is None:
                continue
            if(restrictAreaFunc(intersection, width) == False):
                continue
        if intersection is None:
            continue

        count = 0
        for line in lines:
            if distance_point_to_line(intersection, line) < threshold:
                count += 1

        if count > best_count:
            best_count = count
            best_intersection = intersection
    print (best_intersection)
    print (best_count)
    assert best_intersection is not None, "Não foi possível encontrar o ponto de fuga"
    return best_intersection 

#restrição para a direita
def restrictRight(intersection, width):
    if intersection[0] > width//2 :
        return True
    return False

# restringe para a esquerda
def restrictLeft(intersection, width):
    if intersection[0] < width//2:
        return True
    return False

# restringe para cima
def restrictTop(intersection, height):
    if intersection[1] < height//2:
        return True
    return False
#restringe para baixo
def restrictBottom(intersection, height):
    if intersection[1] > height//2:
        return True
    return False


def vanishing_point(filePath):
    # Carregar imagem
    imagelines = cv2.imread("imagens2/" +  filePath)
    image = imagelines.copy()
    
    # Detectar bordas
    edges = detect_edges(image)
    # Detectar linhas
    lines = detect_lines(edges)
    cv2.imwrite('resultados2/image_edges_'+ filePath , edges)
    #Desenha as linhas na imagem
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(imagelines, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite('resultados2/image_lines_'+ filePath , imagelines) 
    
    # Definir parâmetros do RANSAC
    num_iterations = 100
    threshold = 5

    # Adicionar margem branca
    
    # Encontrar ponto de fuga
    print("Vanishing Point 1")
    vanishing_point = ransac_vanishing_point(lines, num_iterations, threshold , image.shape[0] )
   
    cv2.circle(image, (int(vanishing_point[0][0]), int(vanishing_point[1][0])), 10, (0, 0, 255), -1)
    cv2.imwrite('resultados2/ponto1' + filePath , image)

    # verifica se o vanishing_point está na esquerda da imagem, se sim, restringe para a direita
    print("Vanishing Point 2")
    if vanishing_point[0] < image.shape[1]//2:
        vanishing_point2 = ransac_vanishing_point(lines, num_iterations, threshold , image.shape[0] , restrictRight, True)
    if vanishing_point[0] > image.shape[1]//2:
        vanishing_point2 = ransac_vanishing_point(lines, num_iterations, threshold , image.shape[0] , restrictLeft, True)
    
    cv2.circle(image, (int(vanishing_point2[0][0]), int(vanishing_point2[1][0])), 10, (0, 0, 255), -1)
    cv2.imwrite('resultados2/ponto2' + filePath , image)
    # Faz imagem com borda branca

    imageBorda = cv2.copyMakeBorder(image, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cv2.circle(imageBorda, (int(vanishing_point[0][0] + 200), int(vanishing_point[1][0] + 200)), 10, (0, 0, 255), -1)
    cv2.circle(imageBorda, (int(vanishing_point2[0][0] + 200), int(vanishing_point2[1][0] + 200)), 10, (0, 0, 255), -1)
    cv2.imwrite('resultados2/image_borda_pontos' + filePath , imageBorda)
    



def main ():

    try:
        os.mkdir('resultados2')
    except OSError as e:
        pass

    images = ["image7.jpg"]
    #images = ["ponto_fuga_2.webp"]

    # crie o diretório resultados2



    for(image) in images:
        vanishing_point(image)
    

if __name__ == '__main__':

    main()